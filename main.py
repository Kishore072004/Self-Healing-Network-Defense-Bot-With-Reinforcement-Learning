"""
RL Network Defense Bot — main entry point.

Usage:
    python main.py              # Simulation mode (synthetic traffic, no VMs)
    python main.py --live       # Full VM mode: auto-starts Ubuntu + Kali, real iptables
    python main.py --live -n 50 # Live mode, 50 episodes
    python main.py --sim  -n 50 # Simulation mode, 50 episodes
"""
import sys, io
# Force UTF-8 on Windows console
if sys.stdout.encoding and sys.stdout.encoding.lower() != 'utf-8':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

import argparse
import logging
import signal
import threading
import time
import yaml
import numpy as np
from pathlib import Path
from collections import defaultdict

from agent.environment import NetworkDefenseEnv
from agent.simulator    import TrafficSimulator
from dashboard.app      import run_dashboard, update

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[
        logging.FileHandler("defense_bot.log", encoding="utf-8"),
        logging.StreamHandler(),
    ]
)
logger = logging.getLogger(__name__)


# ══════════════════════════════════════════════════════════════════════
#  Model file validation
# ══════════════════════════════════════════════════════════════════════
def check_models(config: dict):
    """Verify the 5 IDS model files exist in model/ folder."""
    mcfg = config["model"]
    needed = {
        "LightGBM (pkl)":        mcfg["lgb_path"],
        "Logistic Regression":   mcfg["lr_path"],
        "Scaler":                mcfg["scaler_path"],
        "Feature Selector":      mcfg["selector_path"],
        "Label Encoder":         mcfg["label_enc_path"],
    }
    missing = {k: v for k, v in needed.items() if not Path(v).exists()}
    if missing:
        logger.error("Missing IDS model files:")
        for label, path in missing.items():
            logger.error(f"  {label}: {path}")
        raise FileNotFoundError("Model files missing. Place them in the model/ folder.")
    logger.info("All 5 IDS model files found [OK]")


# ══════════════════════════════════════════════════════════════════════
#  Q-table agent (lightweight — no GPU / PyTorch needed)
# ══════════════════════════════════════════════════════════════════════
class QTableAgent:
    """
    Q-learning agent with compact state space.

    Uses 6 key observation dimensions for the Q-table key:
      dim 0 — p_benign     (ML probability of BENIGN)
      dim 2 — p_ddos       (ML probability of DDoS)
      dim 3 — p_scan       (ML probability of PortScan)
      dim 5 — threat_score (ML ensemble threat score)
      dim 6 — confidence   (ML model confidence)
      dim 8 — models_agree (LR/LGB agreement)
    This gives 4^6 = 4096 states × 5 actions = 20480 Q-values max.
    """
    # Which obs dims to use for Q-table key — ALL ML-driven
    _STATE_DIMS = [0, 2, 3, 5, 6, 8]

    def __init__(self, obs_dim: int, n_actions: int,
                 lr: float = 0.3, gamma: float = 0.90,
                 epsilon_start: float = 1.0, epsilon_end: float = 0.05,
                 epsilon_decay: float = 0.92):
        self.n_actions     = n_actions
        self.lr            = lr         # 0.3 — faster convergence
        self.gamma         = gamma
        self.epsilon       = epsilon_start
        self.epsilon_end   = epsilon_end
        self.epsilon_decay = epsilon_decay  # 0.92 — decays to 0.05 in ~35 eps
        self.obs_bins      = 4          # 4 bins per dim (coarser but denser)
        self.obs_dim       = obs_dim
        self.q_table       = defaultdict(lambda: np.zeros(n_actions))

    def _discretise(self, obs: np.ndarray) -> tuple:
        # Only use the 6 key ML dimensions
        key_obs = obs[self._STATE_DIMS]
        clipped = np.clip(key_obs, 0.0, 1.0)
        binned  = np.floor(clipped * (self.obs_bins - 1)).astype(int)
        return tuple(binned.tolist())

    def choose_action(self, obs: np.ndarray, deterministic: bool = False) -> int:
        state = self._discretise(obs)
        if not deterministic and np.random.random() < self.epsilon:
            return np.random.randint(self.n_actions)
        return int(np.argmax(self.q_table[state]))

    def learn(self, obs, action, reward, next_obs, done):
        s  = self._discretise(obs)
        s2 = self._discretise(next_obs)
        best_next = np.max(self.q_table[s2]) if not done else 0.0
        td_target = reward + self.gamma * best_next
        self.q_table[s][action] += self.lr * (td_target - self.q_table[s][action])

    def decay_epsilon(self):
        self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)

    @property
    def n_states_seen(self):
        return len(self.q_table)

    def save(self, path: str):
        import pickle
        with open(path, "wb") as f:
            pickle.dump(dict(self.q_table), f)
        logger.info(f"Q-table saved → {path}")

    def load(self, path: str):
        import pickle
        with open(path, "rb") as f:
            data = pickle.load(f)
        self.q_table.update(data)
        logger.info(f"Q-table loaded ← {path}")


# ══════════════════════════════════════════════════════════════════════
#  Dashboard state builder helper
# ══════════════════════════════════════════════════════════════════════
def _build_dashboard_payload(episode, step, total_reward, action,
                              info, obs, env, blocked_ips=None,
                              step_reward: float = 0.0) -> dict:
    """Build JSON-safe payload — all numpy types cast to native Python.

    Obs dims (17) — NEW ML-driven layout:
      0=p_benign  1=p_bot  2=p_ddos  3=p_scan  4=p_brute
      5=threat_score  6=confidence  7=class_idx  8=models_agree
      9=cpu  10=mem  11=bytes_recv  12=bytes_sent
      13=one_hot_BENIGN  14=one_hot_Bot  15=one_hot_DDoS  16=one_hot_PortScan
    """
    last_signal  = getattr(getattr(env, "sb", None), "_last_reward_signal", {}) or {}
    last_metrics = getattr(getattr(env, "sb", None), "_last_metrics", {}) or {}

    # ML class probabilities for dashboard display
    ml_probs = {
        "p_benign": float(obs[0]),
        "p_bot":    float(obs[1]),
        "p_ddos":   float(obs[2]),
        "p_scan":   float(obs[3]),
        "p_brute":  float(obs[4]),
    }

    return {
        "mode":            "SIMULATION" if getattr(env, "sim_mode", True) else "LIVE MODE",
        "episode":         int(episode),
        "step":            int(step),
        "total_reward":    round(float(total_reward), 2),
        "step_reward":     round(float(step_reward), 4),
        "action":          int(action),
        "action_name":     info.get("action_name", "Allow"),
        "threat_level":    float(obs[5]),     # threat_score
        "threat_class":    info.get("threat_class", "BENIGN"),
        "confidence":      float(obs[6]),     # confidence
        "cpu":             float(obs[9] * 100),
        "memory":          float(obs[10] * 100),
        "bytes_recv":      float(obs[11]),
        "bytes_sent":      float(obs[12]),
        "is_attacking":    bool(obs[5] > 0.3),  # threat_score > 0.3
        "top_attacker_ip": str(env.sb.get_top_attacker_ip()),
        "blocked_ips":     list(blocked_ips or info.get("blocked_ips", [])),
        "fw_result":       str(info.get("fw_result", "")),
        # ML probability vector
        **ml_probs,
        # IDS Monitor fields
        "ids_final_class":    str(last_signal.get("ids_class", "Normal Traffic")),
        "ids_lr_class":       str(last_signal.get("lr_class", "Normal Traffic")),
        "ids_lgb_class":      str(last_signal.get("lgb_class", "Normal Traffic")),
        "ids_models_agree":   bool(last_signal.get("models_agree", True)),
        "ids_flow_count":     int(last_metrics.get("_flows", 0)),
        "ids_packet_count":   int(last_metrics.get("_packets", 0)),
        "ids_nonzero_features": int(last_metrics.get("_nonzero", 0)),
    }


# ══════════════════════════════════════════════════════════════════════
#  SIMULATION MODE  (no VMs)
# ══════════════════════════════════════════════════════════════════════
def run_simulation(config: dict, total_episodes: int):
    """VM-free simulation mode using Q-table agent + synthetic traffic."""
    sim   = TrafficSimulator()
    env   = NetworkDefenseEnv(config, simulator=sim)
    t     = config["training"]
    agent = QTableAgent(
        t["obs_dim"], t["n_actions"],
        lr=t.get("learning_rate", 0.1),
        gamma=t["gamma"]
    )

    Path("checkpoints").mkdir(exist_ok=True)

    for episode in range(1, total_episodes + 1):
        attack_type = sim.curriculum(episode - 1)
        sim.set_attack(attack_type)
        obs, _ = env.reset()
        total_reward = 0.0
        step = 0
        done = False

        logger.info(f"Episode {episode}/{total_episodes} | "
                    f"attack={attack_type} | ε={agent.epsilon:.3f}")

        while not done:
            action = agent.choose_action(obs)
            next_obs, reward, done, _, info = env.step(action)
            agent.learn(obs, action, reward, next_obs, done)
            obs = next_obs
            total_reward += reward
            step += 1

            update(_build_dashboard_payload(
                episode, step, total_reward, action, info, obs, env,
                step_reward=reward,
            ))

        agent.decay_epsilon()
        logger.info(f"Episode {episode} done | "
                    f"attack={attack_type} | reward={total_reward:.2f}")

        if episode % 25 == 0:
            agent.save(f"checkpoints/q_table_ep{episode}.pkl")

    agent.save("checkpoints/q_table_final.pkl")
    logger.info("Simulation training complete.")
    print("\n=== DONE — Q-table saved to checkpoints/q_table_final.pkl ===")


# ══════════════════════════════════════════════════════════════════════
#  LIVE MODE  (real VMs, real iptables, real attacks)
# ══════════════════════════════════════════════════════════════════════
def run_live(config: dict, total_episodes: int):
    """
    Fully automatic live mode:
      1. Auto-start Ubuntu + Kali VMs via vmrun
      2. Deploy metrics API to Ubuntu
      3. Q-table agent trains against real Kali attacks
      4. Agent blocks/unblocks IPs via Ubuntu iptables over SSH
    """
    from agent.vm_manager    import VMManager
    from agent.attack_launcher import AttackLauncher

    # ── Boot VMs ──────────────────────────────────────────────────────
    vm_mgr = VMManager(config)
    logger.info("Starting VM boot sequence …")
    vms_ready = vm_mgr.startup()
    if not vms_ready:
        logger.error("VMs failed to start. Aborting live mode.")
        print("\n[ERROR] Could not reach Ubuntu or Kali VM via SSH.")
        print("  • Check your VM IPs in config.yaml (network section)")
        print("  • Check your VMX paths in config.yaml (vmware section)")
        print("  • Or run:  python main.py --sim   to use simulation mode")
        return

    # ── Setup environment & verify API ───────────────────────────────
    env = NetworkDefenseEnv(config)          # live mode, no simulator

    # Block until Ubuntu API returns non-zero features
    logger.info("Verifying Ubuntu metrics API connection …")
    if env.sb.warmup(max_wait=30):
        logger.info("API warmup OK — features are live ✓")
    else:
        logger.warning("API warmup incomplete — some features may be zero initially")

    attacker = AttackLauncher(config)
    attacker.deploy()

    # ── Q-table agent ────────────────────────────────────────────────
    t     = config["training"]
    agent = QTableAgent(
        t["obs_dim"], t["n_actions"],
        lr=t.get("learning_rate", 0.3),
        gamma=t.get("gamma", 0.90),
        epsilon_start=1.0,
        epsilon_end=0.05,
        epsilon_decay=0.92,   # decays to ~0.05 in ~35 episodes
    )

    # Load checkpoint if available
    ckpt = Path("checkpoints/q_table_final.pkl")
    if ckpt.exists():
        agent.load(str(ckpt))
        logger.info("Resumed from existing checkpoint.")

    Path("checkpoints").mkdir(exist_ok=True)

    # ── Graceful Ctrl+C handler ───────────────────────────────────────
    _stop = threading.Event()

    def _sigint(sig, frame):
        logger.info("Interrupt received — cleaning up …")
        _stop.set()

    signal.signal(signal.SIGINT, _sigint)

    # ── Training loop ─────────────────────────────────────────────────
    try:
        for episode in range(1, total_episodes + 1):
            if _stop.is_set():
                break

            # Rotate through all attack types every 4 episodes so the
            # agent sees every class from episode 1 (not just scan for 20 eps)
            LIVE_CYCLE = ["scan", "ddos", "brute", "ddos", "scan", "web", "brute", "ddos"]
            attack_type = LIVE_CYCLE[(episode - 1) % len(LIVE_CYCLE)]
            logger.info(f"Episode {episode}/{total_episodes} | "
                        f"attack={attack_type} | states={agent.n_states_seen} | "
                        f"e={agent.epsilon:.3f}")

            # Launch Kali attack for this episode
            from dashboard.app import get_launch_mode
            if get_launch_mode() == "AUTO":
                logger.info(f"[AUTO MODE] Launching Kali attack ({attack_type})")
                attacker.launch(attack_type)
            else:
                logger.info(f"[MANUAL MODE] Skipping auto-attack. Waiting 15s for manual Kali attack (expected: {attack_type})")

            time.sleep(15)          # let attack ramp up + tshark capture traffic

            obs, _ = env.reset()
            total_reward = 0.0
            step = 0
            done = False

            while not done and not _stop.is_set():
                action = agent.choose_action(obs)
                next_obs, reward, done, _, info = env.step(action)
                agent.learn(obs, action, reward, next_obs, done)
                obs = next_obs
                total_reward += reward
                step += 1

                update(_build_dashboard_payload(
                    episode, step, total_reward, action, info, obs, env,
                    step_reward=reward,
                ))

            # Stop current attack between episodes
            attacker.stop()
            agent.decay_epsilon()
            logger.info(
                f"Episode {episode} done | attack={attack_type} | "
                f"reward={total_reward:.2f} | states={agent.n_states_seen} | "
                f"e={agent.epsilon:.3f} | lr={agent.lr}"
            )

            # Checkpoint
            if episode % 10 == 0:
                agent.save(f"checkpoints/q_table_ep{episode}.pkl")

            # Rest between episodes
            if not _stop.is_set():
                time.sleep(3)

    finally:
        # Cleanup: unblock all IPs, optionally shut down VMs
        logger.info("Cleaning up firewall rules on Ubuntu …")
        try:
            env.fw.cleanup_all_rules()
        except Exception:
            pass

        attacker.stop()
        agent.save("checkpoints/q_table_final.pkl")
        logger.info("Live training complete. Q-table saved.")

        vm_mgr.shutdown()
        print("\n=== DONE — Q-table saved to checkpoints/q_table_final.pkl ===")


# ══════════════════════════════════════════════════════════════════════
#  Entry point
# ══════════════════════════════════════════════════════════════════════
def main():
    parser = argparse.ArgumentParser(description="RL Network Defense Bot")
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--live", action="store_true",
                       help="Full VM mode: auto-starts Ubuntu+Kali, real iptables")
    group.add_argument("--sim",  action="store_true",
                       help="Simulation mode — no VMs needed (default)")
    parser.add_argument("-n", "--episodes", type=int, default=100,
                        help="Number of training episodes (default: 100)")
    args = parser.parse_args()

    with open("config.yaml") as f:
        config = yaml.safe_load(f)

    # Determine mode
    if args.live:
        config["simulation"] = False
    else:
        # --sim or bare run → simulation
        config["simulation"] = True

    sim_mode = config.get("simulation", True)

    # Verify model files
    check_models(config)

    # Create dirs
    Path("checkpoints").mkdir(exist_ok=True)
    Path("logs").mkdir(exist_ok=True)

    # Start dashboard
    dash_cfg = config["dashboard"]
    logger.info(f"Starting dashboard at http://localhost:{dash_cfg['port']}")
    threading.Thread(
        target=run_dashboard,
        kwargs={"host": dash_cfg["host"], "port": dash_cfg["port"]},
        daemon=True
    ).start()
    time.sleep(2)

    # Banner
    mode_str = "SIMULATION (no VMs)" if sim_mode else "LIVE  (Kali attacker → Ubuntu victim)"
    sep = "═" * 60
    print(f"\n{sep}")
    print(f"  RL Network Defense Bot  —  {mode_str}")
    print(sep)
    print(f"  Agent      : Q-table (epsilon-greedy, ML-only state)")
    print(f"  Engine     : LR-primary + LGB ensemble (52->30 features)")
    print(f"  Detection  : 100% ML — no hardcoded rules or thresholds")
    print(f"  Classes    : BENIGN / Bot / DDoS / PortScan / Brute")
    print(f"  RL Actions : Allow / Block IP / Rate-limit / Restart / Flush")
    print(f"  Reward     : Pure ML probability vector (no alert counts)")
    print(f"  Firewall   : {'Ubuntu victim only -- iptables via SSH' if not sim_mode else 'Simulated'}")
    print(f"  Attacker   : {'Kali Linux (192.168.100.5)' if not sim_mode else 'Synthetic traffic'}")
    print(f"  Episodes   : {args.episodes}")
    print(f"  Dashboard  : http://localhost:{dash_cfg['port']}")
    print(f"  Log file   : defense_bot.log")
    print(f"{sep}\n")

    if sim_mode:
        run_simulation(config, args.episodes)
    else:
        run_live(config, args.episodes)

    # Keep dashboard alive after training so the browser stays connected
    print(f"\nTraining done. Dashboard still live at "
          f"http://localhost:{dash_cfg['port']}")
    print("Press Ctrl+C to shut down.\n")
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\nShutting down.")


if __name__ == "__main__":
    main()
