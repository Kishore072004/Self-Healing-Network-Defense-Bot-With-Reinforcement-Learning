"""
RL environment — agent defends Ubuntu (victim) from Kali (attacker).

Flow:
  Kali (192.168.100.5) → attacks Ubuntu (192.168.100.10)
  Windows agent monitors Ubuntu metrics API → classifies traffic via LR+LGB ensemble
  Agent picks action → executes iptables rules on Ubuntu via SSH

Actions (5):
  0 — Allow             (do nothing)
  1 — Block IP          (iptables DROP src IP)
  2 — Rate-limit        (iptables DROP src subnet)
  3 — Restart service   (systemctl restart networking)
  4 — Flush iptables    (clear RL_DEFENSE chain)

Reward: derived DIRECTLY from the ML model's probability outputs.
  - No hardcoded class × action lookup tables.
  - No alert-count thresholds.
  - The ML model decides what the traffic is; reward logic follows.
"""
import logging
import numpy as np
import gymnasium as gym
from monitor.state_builder import StateBuilder

logger = logging.getLogger(__name__)

ACTION_NAMES = {
    0: "Allow",
    1: "Block IP",
    2: "Rate-limit",
    3: "Restart service",
    4: "Flush iptables",
}

# 5 classes — must match model/metadata.json class_names order
CLASS_NAMES = ["BENIGN", "Bot", "DDoS", "PortScan", "Brute"]


def compute_reward(probs: np.ndarray, action: int) -> float:
    """
    100% ML-driven reward function.

    Uses ONLY the 5-class probability vector from the ensemble model.
    No hardcoded thresholds, no alert counts, no heuristic overrides.

    probs: 5-dim output from the ensemble [BENIGN, Bot, DDoS, PortScan, Brute]
    action: 0-4

    Reward logic:
      - Allow during attack → large negative (missed defense)
      - Block/Rate-limit during attack → large positive (correct defense)
      - Block during benign → negative (false positive, service disruption)
      - Allow during benign → positive (correct pass-through)
    """
    p_safe  = float(probs[0])   # BENIGN
    p_bot   = float(probs[1])   # Bot
    p_ddos  = float(probs[2])   # DDoS
    p_scan  = float(probs[3])   # PortScan
    p_brute = float(probs[4]) if len(probs) > 4 else 0.0  # Brute

    # Total attack probability (complement of benign)
    p_attack = 1.0 - p_safe

    if action == 0:   # Allow — good if benign, bad if attack
        reward = p_safe * 1.0 - p_attack * 5.0

    elif action == 1:  # Block IP — good for scan/brute/bot, ok for ddos
        reward = (p_scan + p_bot + p_brute) * 5.0 + p_ddos * 2.0 - p_safe * 3.0

    elif action == 2:  # Rate-limit — best for DDoS, ok for other attacks
        reward = p_ddos * 6.0 + p_bot * 2.0 + p_brute * 2.0 + p_scan * 1.0 - p_safe * 3.0

    elif action == 3:  # Restart networking — nuclear option, costly
        reward = (p_bot + p_ddos + p_brute) * 4.0 - p_safe * 2.0 - p_scan * 1.0

    elif action == 4:  # Flush iptables — recovery action
        reward = p_attack * 3.0 - p_safe * 4.0

    else:
        reward = 0.0

    return round(float(reward), 4)


class NetworkDefenseEnv(gym.Env):
    """
    RL environment: Kali → Ubuntu ← Windows agent.

    Observation (17-dim) — ALL ML-driven:
      0  p_benign       (ML probability of BENIGN)
      1  p_bot          (ML probability of Bot)
      2  p_ddos         (ML probability of DDoS)
      3  p_scan         (ML probability of PortScan)
      4  p_brute        (ML probability of Brute)
      5  threat_score   (ML ensemble threat score)
      6  confidence     (ML model's max class probability)
      7  class_idx      (normalised class index)
      8  models_agree   (LR and LGB agree: 1.0 or 0.0)
      9  cpu_percent    (Ubuntu CPU load)
      10 mem_percent    (Ubuntu memory)
      11 bytes_recv     (inbound bytes, normalised)
      12 bytes_sent     (outbound bytes, normalised)
      13 one_hot_BENIGN
      14 one_hot_Bot
      15 one_hot_DDoS
      16 one_hot_PortScan
    """

    metadata = {"render_modes": []}

    def __init__(self, config: dict, simulator=None):
        super().__init__()
        self.config    = config
        self.simulator = simulator
        self.sb        = StateBuilder(config, simulator=simulator)
        self.max_steps = config["training"]["max_steps_per_episode"]
        self.sim_mode  = simulator is not None

        # Live mode: real Ubuntu iptables via SSH
        if not self.sim_mode:
            from defense.ubuntu_firewall import UbuntuFirewallDefender
            self.fw = UbuntuFirewallDefender(config)
            logger.info("Live mode: Ubuntu iptables over SSH ✓")
        else:
            self.fw = None

        self.observation_space = gym.spaces.Box(
            low=0.0, high=1.0,
            shape=(config["training"]["obs_dim"],),
            dtype=np.float32
        )
        self.action_space = gym.spaces.Discrete(config["training"]["n_actions"])

        self.step_count  = 0
        self.episode_reward = 0.0
        self._last_probs = np.array([1.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32)  # 5 classes

    # ------------------------------------------------------------------
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        if self.fw:
            try:
                self.fw.cleanup_all_rules()
            except Exception as e:
                logger.warning(f"reset cleanup: {e}")
        self.step_count     = 0
        self.episode_reward = 0.0
        self._last_probs    = np.array([1.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32)
        obs = self.sb.build_observation()
        return obs, {}

    # ------------------------------------------------------------------
    def step(self, action: int):
        metrics = self.sb.fetch_metrics()

        fw_result = self._execute_action(action, metrics)

        # Notify simulator of defensive action (reduces traffic intensity)
        if self.simulator and action in [1, 2, 3, 4]:
            self.simulator.mark_defended()

        obs    = self.sb.build_observation(metrics)
        reward = self._compute_reward(action)
        self.episode_reward += reward
        self.step_count     += 1
        done = self.step_count >= self.max_steps

        # In live mode, wait between steps so the API observes changing traffic
        if not self.sim_mode:
            import time
            time.sleep(1.5)

        class_name = _decode_class(obs)

        if self.sim_mode and fw_result == "simulated":
            fw_result = f"{ACTION_NAMES.get(action, str(action))} ({class_name})"

        blocked_ips = self.fw.get_blocked() if self.fw else []

        info = {
            "action_name":  ACTION_NAMES[action],
            "threat_class": class_name,
            "confidence":   float(obs[6]),
            "threat_score": float(obs[5]),
            "blocked_ips":  blocked_ips,
            "fw_result":    fw_result,
        }
        logger.info(
            f"Step {self.step_count:03d} | "
            f"action={ACTION_NAMES[action]:<18} | "
            f"class={class_name:<10} | "
            f"threat={float(obs[5]):.2f} | "
            f"conf={float(obs[6]):.2f} | "
            f"reward={reward:+.3f}"
        )
        return obs, reward, done, False, info

    # ------------------------------------------------------------------
    def _execute_action(self, action: int, metrics: dict) -> str:
        if self.sim_mode:
            return "simulated"

        ip     = metrics.get("top_attacker", "")
        subnet = f"{ip}/32" if ip else ""

        try:
            if action == 1 and ip:
                ok = self.fw.block_ip(ip)
                return f"Blocked {ip}" if ok else f"Block failed: {ip}"

            elif action == 2 and subnet:
                ok = self.fw.rate_limit_subnet(subnet)
                return f"Rate-limited {subnet}" if ok else "Rate-limit failed"

            elif action == 3:
                ok = self.fw.restart_networking()
                return "Networking restarted" if ok else "Restart failed"

            elif action == 4:
                self.fw.cleanup_all_rules()
                return "Flushed iptables"

        except Exception as e:
            logger.warning(f"_execute_action error: {e}")
            return f"error: {e}"

        return "no-op"

    # ------------------------------------------------------------------
    #  100% ML-driven reward — uses ONLY model probability output
    # ------------------------------------------------------------------
    def _compute_reward(self, action: int) -> float:
        """
        Compute reward using ONLY ML probability vector.
        No metrics dict, no alert counts, no hardcoded thresholds.
        """
        probs = self.sb.get_last_probs()
        reward = compute_reward(probs, action)
        logger.debug(
            f"Reward: probs={np.round(probs, 3)} "
            f"action={ACTION_NAMES[action]} reward={reward:+.4f}"
        )
        return reward


# ------------------------------------------------------------------
#  Helper: decode class name from obs one-hot bits [13-16]
# ------------------------------------------------------------------
def _decode_class(obs: np.ndarray) -> str:
    """Decode class from one-hot bits: 13=BENIGN, 14=Bot, 15=DDoS, 16=PortScan."""
    if   obs[15] > 0.5: return "DDoS"       # check attacks first
    elif obs[16] > 0.5: return "PortScan"
    elif obs[14] > 0.5: return "Bot"
    elif obs[13] > 0.5: return "BENIGN"
    # Fallback: use class index from obs[7]
    idx = int(round(float(obs[7]) * (len(CLASS_NAMES) - 1)))
    idx = max(0, min(idx, len(CLASS_NAMES) - 1))
    return CLASS_NAMES[idx]
