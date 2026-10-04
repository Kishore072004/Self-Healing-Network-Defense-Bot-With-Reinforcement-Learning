"""
monitor/state_builder.py

State builder — fetches metrics from Ubuntu API and builds the 17-dim
RL observation vector.

Classification: PURELY the ML engine.
  LR-primary ensemble (LR 80% + LGB 20%), 52-feature input -> 30 selected -> 7 classes.
  NO hardcoded rules. NO threshold overrides. NO dataset substitution.
  The model decides everything from REAL live features.

Observation vector (17-dim) — ALL ML-driven:
  0  p_benign       (ML probability of BENIGN)
  1  p_bot          (ML probability of Bot)
  2  p_ddos         (ML probability of DDoS)
  3  p_scan         (ML probability of PortScan)
  4  p_brute        (ML probability of Brute)
  5  threat_score   (ML ensemble threat score)
  6  confidence     (ML model's max class probability)
  7  class_idx      (normalised class index)
  8  models_agree   (LR and LGB agree: 1.0 or 0.0)
  9  cpu_percent    (Ubuntu CPU load, normalised)
  10 mem_percent    (Ubuntu memory, normalised)
  11 bytes_recv     (inbound bytes, normalised)
  12 bytes_sent     (outbound bytes, normalised)
  13 one_hot_BENIGN
  14 one_hot_Bot
  15 one_hot_DDoS
  16 one_hot_PortScan
"""
import logging
import time
import numpy as np
import requests
import warnings
from classifier.intrusion_predictor import IntrusionPredictor

log = logging.getLogger(__name__)

# RL env class ordering (must match environment.py CLASS_NAMES)
RL_CLASS_NAMES = ["BENIGN", "Bot", "DDoS", "PortScan", "Brute"]


class StateBuilder:
    def __init__(self, config: dict, simulator=None):
        self.simulator   = simulator
        self.obs_dim     = config["training"]["obs_dim"]

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self.ids = IntrusionPredictor(model_dir="model")

        # Expose for warmup() compatibility
        self.feature_names = self.ids.feature_names    # 52 names
        self.n_features    = self.ids.n_features        # 52

        self._last_metrics = {}
        self._last_probs   = np.array([1.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32)
        self._last_reward_signal = {}
        self._consecutive_failures = 0

        # HTTP session for keep-alive (reduces TCP overhead)
        self._session = requests.Session()
        self._session.headers.update({
            "Connection": "keep-alive",
            "Accept": "application/json",
        })

        if simulator is None:
            ip   = config["network"]["ubuntu_ip"]
            port = config["ubuntu"]["metrics_port"]
            self.url = f"http://{ip}:{port}/state"
        else:
            self.url = None

    # ── Fetch with retries ────────────────────────────────────────────
    def fetch_metrics(self) -> dict:
        if self.simulator is not None:
            m = self.simulator.generate()
            self._last_metrics = m
            return m

        for attempt in range(5):
            try:
                timeout = 15 + attempt * 10  # 15s, 25s, 35s, 45s, 55s
                r = self._session.get(self.url, timeout=timeout)
                r.raise_for_status()
                m = r.json()
                if not isinstance(m, dict) or len(m) < 5:
                    log.warning("API returned unexpected data")
                    continue
                self._last_metrics = m
                self._consecutive_failures = 0
                return m
            except Exception as e:
                self._consecutive_failures += 1
                if attempt < 4:
                    wait = 2 + attempt * 2
                    log.warning(f"fetch_metrics attempt {attempt+1} failed: {e} (retry in {wait}s)")
                    time.sleep(wait)
                else:
                    log.error(f"fetch_metrics FAILED after 5 attempts: {e}")

        # After 10 consecutive failures, recreate the session (fresh TCP)
        if self._consecutive_failures >= 10:
            log.warning("Recreating HTTP session after 10 consecutive failures")
            self._session.close()
            self._session = requests.Session()
            self._session.headers.update({
                "Connection": "keep-alive",
                "Accept": "application/json",
            })
            self._consecutive_failures = 0

        if self._last_metrics:
            log.warning("Using cached metrics from last successful fetch")
            return dict(self._last_metrics)
        return {}

    # ── Warmup: block until API returns non-zero features ─────────────
    def warmup(self, max_wait: int = 60) -> bool:
        log.info(f"Warming up API connection ({self.url}) ...")
        deadline = time.time() + max_wait

        while time.time() < deadline:
            try:
                m = self.fetch_metrics()
                if not m:
                    time.sleep(3)
                    continue
                nonzero = sum(1 for f in self.feature_names if float(m.get(f, 0.0)) != 0.0)
                cpu = m.get("cpu_percent", 0)
                mem = m.get("mem_percent", 0)
                if nonzero >= 3 and (cpu > 0 or mem > 0):
                    log.info(
                        f"API warm: {nonzero}/{self.n_features} features non-zero, "
                        f"cpu={cpu:.1f}% mem={mem:.1f}%"
                    )
                    return True
                log.info(f"API not ready: {nonzero} non-zero — waiting...")
                time.sleep(3)
            except Exception as e:
                log.warning(f"Warmup attempt failed: {e}")
                time.sleep(3)

        log.error(f"API warmup timed out after {max_wait}s")
        return False

    # ── Build 17-dim observation — 100% ML-driven ─────────────────────
    def build_observation(self, metrics: dict = None) -> np.ndarray:
        if not metrics:
            metrics = self.fetch_metrics()

        # Pure ML classification — no rules, no overrides, no data substitution
        rl_class, confidence, threat_score, rl_probs, models_agree = self._classify(metrics)
        self._last_probs = rl_probs

        class_id = RL_CLASS_NAMES.index(rl_class) if rl_class in RL_CLASS_NAMES else 0

        obs = np.array([
            float(rl_probs[0]),                                    # 0  p_benign
            float(rl_probs[1]),                                    # 1  p_bot
            float(rl_probs[2]),                                    # 2  p_ddos
            float(rl_probs[3]),                                    # 3  p_scan
            float(rl_probs[4]),                                    # 4  p_brute
            threat_score,                                          # 5  threat_score
            confidence,                                            # 6  confidence
            class_id / max(len(RL_CLASS_NAMES) - 1, 1),           # 7  class_idx (normalised)
            1.0 if models_agree else 0.0,                          # 8  models_agree
            metrics.get("cpu_percent",  0.0) / 100.0,             # 9  cpu
            metrics.get("mem_percent",  0.0) / 100.0,             # 10 mem
            min(metrics.get("bytes_recv", 0) / 1e8, 1.0),         # 11 bytes_recv
            min(metrics.get("bytes_sent", 0) / 1e8, 1.0),         # 12 bytes_sent
            1.0 if rl_class == "BENIGN"   else 0.0,               # 13 one-hot
            1.0 if rl_class == "Bot"      else 0.0,               # 14 one-hot
            1.0 if rl_class == "DDoS"     else 0.0,               # 15 one-hot
            1.0 if rl_class == "PortScan" else 0.0,               # 16 one-hot
        ], dtype=np.float32)

        log.info(
            f"[OBS] ml_class={rl_class} conf={confidence:.3f} threat={threat_score:.3f} | "
            f"probs=[B={rl_probs[0]:.2f} Bot={rl_probs[1]:.2f} "
            f"DDoS={rl_probs[2]:.2f} Scan={rl_probs[3]:.2f} Brute={rl_probs[4]:.2f}] | "
            f"cpu={metrics.get('cpu_percent', 0):.0f}%"
        )
        return obs

    # ── Pure ML classification — NO hardcoded rules, NO data swap ─────
    def _classify(self, metrics: dict):
        """
        100% ML-driven classification via the IDS engine.
        Uses ONLY the live features from Ubuntu API (or simulator).
        LR-primary ensemble: LogisticRegression (80%) + LightGBM (20%).
        No alert-count thresholds, no packet-rate rules, no dataset substitution.

        Returns: (rl_class, confidence, threat_score, rl_probs_5, models_agree)

        NOTE: predict_raw() is called ONCE here. rl_probs are built from the
        returned probs_7 via build_rl_probs_from_vec() to avoid double inference.
        """
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                # Single inference — capture the 7-class probability vector
                ids_class, rl_family, confidence, threat_score, probs_7, both_agree = \
                    self.ids.predict_raw(metrics)
                # Build 5-class RL vector from already-computed probs_7 (no second call)
                rl_probs = self.ids.build_rl_probs_from_vec(probs_7)

            # Store reward signal for dashboard
            self._last_reward_signal = {
                "ids_class": ids_class,
                "rl_family": rl_family,
                "lgb_class": self.ids._last_lgb_class,
                "lr_class":  self.ids._last_lr_class,
                "models_agree": self.ids._last_agree,
                "confidence": confidence,
                "threat_score": threat_score,
            }

            # Store raw metrics for dashboard display (NOT for RL observation)
            self._last_metrics.update({
                "_ml_class": rl_family,
                "_ml_confidence": confidence,
                "_ml_threat": threat_score,
            })

            log.info(
                f"[ML] IDS: LR={self._last_reward_signal.get('lr_class','?')} "
                f"LGB={self._last_reward_signal.get('lgb_class','?')} "
                f"-> {ids_class}({confidence:.3f}) RL={rl_family} "
                f"threat={threat_score:.3f} agree={both_agree}"
            )
            return rl_family, confidence, threat_score, rl_probs, both_agree

        except Exception as e:
            log.error(f"_classify error: {e}", exc_info=True)
            return "BENIGN", 0.5, 0.0, np.array([1, 0, 0, 0, 0], dtype=np.float32), True

    # ── Accessors ──────────────────────────────────────────────────────
    def get_last_probs(self) -> np.ndarray:
        return self._last_probs.copy()

    def get_top_attacker_ip(self) -> str:
        if self.simulator is not None:
            return getattr(self.simulator, "_top_ip", "")
        return self._last_metrics.get("top_attacker", "")

    def get_last_reward_signal(self) -> dict:
        """Return the last reward signal for dashboard display."""
        return dict(self._last_reward_signal)
