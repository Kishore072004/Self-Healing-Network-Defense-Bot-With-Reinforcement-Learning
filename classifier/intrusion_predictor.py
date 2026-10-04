"""
classifier/intrusion_predictor.py

Intrusion Detection Engine — production inference for the RL defense bot.

Models (all in model/ folder):
  lightgbm.pkl        — LGBMClassifier (sklearn wrapper, 30 features)
  logistic.pkl        — LogisticRegression (30 features)
  scaler.pkl          — StandardScaler (30 -> 30)
  feature_selector.pkl— SelectKBest (52 -> 30)
  label_encoder.pkl   — LabelEncoder (7 classes)

Pipeline:
  raw_features (52) -> selector (30) -> scaler (30) -> LR (primary) + LGB (secondary) -> label

Classes (index -> name):
  0: Bots
  1: Brute Force
  2: DDoS
  3: DoS
  4: Normal Traffic   <- BENIGN equivalent
  5: Port Scanning
  6: Web Attacks

Reward mapping to RL environment actions (5):
  0 Allow | 1 Block IP | 2 Rate-limit | 3 Restart | 4 Flush iptables

NOTE: The LightGBM model is overtrained on the majority class (Normal Traffic)
      and outputs ~100% BENIGN for virtually all inputs. The Logistic Regression
      model (95.6% accuracy) is therefore used as the PRIMARY classifier.
      LGB is retained as a secondary signal only when it agrees with LR.
"""
import logging
import warnings
import numpy as np
import joblib
from pathlib import Path

log = logging.getLogger(__name__)

# ── Class name constants (must match label_encoder.pkl classes exactly) ──────
IDS_CLASSES = ["Bots", "Brute Force", "DDoS", "DoS", "Normal Traffic",
               "Port Scanning", "Web Attacks"]
BENIGN_CLS  = "Normal Traffic"
BENIGN_IDX  = IDS_CLASSES.index(BENIGN_CLS)   # 4

# ── Map IDS classes -> RL environment class family ───────────────────────────
# Used by StateBuilder to translate into the existing 17-dim obs space
CLS_FAMILY = {
    "Normal Traffic": "BENIGN",
    "Bots":           "Bot",
    "Brute Force":    "Brute",
    "DDoS":           "DDoS",
    "DoS":            "DDoS",        # treat DoS same as DDoS for RL
    "Port Scanning":  "PortScan",
    "Web Attacks":    "Bot",         # web attacks -> Bot family (closest RL class)
}

# ── 52 CIC-IDS2017 feature names (EXACT order the SelectKBest was fitted on) ─
FEATURE_NAMES_52 = [
    "Destination Port",
    "Flow Duration",
    "Total Fwd Packets",
    "Total Length of Fwd Packets",
    "Fwd Packet Length Max",
    "Fwd Packet Length Min",
    "Fwd Packet Length Mean",
    "Fwd Packet Length Std",
    "Bwd Packet Length Max",
    "Bwd Packet Length Min",
    "Bwd Packet Length Mean",
    "Bwd Packet Length Std",
    "Flow Bytes/s",
    "Flow Packets/s",
    "Flow IAT Mean",
    "Flow IAT Std",
    "Flow IAT Max",
    "Flow IAT Min",
    "Fwd IAT Total",
    "Fwd IAT Mean",
    "Fwd IAT Std",
    "Fwd IAT Max",
    "Fwd IAT Min",
    "Bwd IAT Total",
    "Bwd IAT Mean",
    "Bwd IAT Std",
    "Bwd IAT Max",
    "Bwd IAT Min",
    "Fwd Header Length",
    "Bwd Header Length",
    "Fwd Packets/s",
    "Bwd Packets/s",
    "Min Packet Length",
    "Max Packet Length",
    "Packet Length Mean",
    "Packet Length Std",
    "Packet Length Variance",
    "FIN Flag Count",
    "PSH Flag Count",
    "ACK Flag Count",
    "Average Packet Size",
    "Subflow Fwd Bytes",
    "Init_Win_bytes_forward",
    "Init_Win_bytes_backward",
    "act_data_pkt_fwd",
    "min_seg_size_forward",
    "Active Mean",
    "Active Max",
    "Active Min",
    "Idle Mean",
    "Idle Max",
    "Idle Min",
]

assert len(FEATURE_NAMES_52) == 52, "Feature list must have exactly 52 entries"


class IntrusionPredictor:
    """
    Intrusion Detection Engine.

    Loads the 5 .pkl model artifacts and exposes:
      - predict(metrics_dict)  -> (rl_family, confidence, threat_score, both_agree)
      - predict_raw(metrics_dict) -> full 6-tuple with all details
      - build_rl_probs(metrics_dict) -> np.ndarray[5] for RL reward
      - reward_signal(metrics_dict) -> dict (used by StateBuilder / environment)

    'metrics_dict' must contain at least the keys in FEATURE_NAMES_52.
    Missing keys default to 0.0 (gracefully degraded).
    """

    def __init__(self, model_dir: str = "model"):
        base = Path(model_dir)
        log.info(f"Loading IDS models from '{model_dir}/' ...")

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")   # suppress sklearn version warnings
            self.selector = joblib.load(base / "feature_selector.pkl")
            self.scaler   = joblib.load(base / "scaler.pkl")
            self.lgb      = joblib.load(base / "lightgbm.pkl")
            self.lr       = joblib.load(base / "logistic.pkl")
            self.le       = joblib.load(base / "label_encoder.pkl")

        # Validate classes
        loaded_cls = list(self.le.classes_)
        assert loaded_cls == IDS_CLASSES, (
            f"Label encoder classes mismatch!\n"
            f"  Expected: {IDS_CLASSES}\n"
            f"  Got     : {loaded_cls}"
        )

        # Cache last results for dashboard
        self._last_lgb_class = BENIGN_CLS
        self._last_lr_class  = BENIGN_CLS
        self._last_agree     = True

        log.info(
            f"[IDS] Ready | 52 -> 30 features | "
            f"7 classes: {IDS_CLASSES} | LR-primary ensemble"
        )

    # ------------------------------------------------------------------
    def _build_vector(self, metrics: dict) -> np.ndarray:
        """
        Build the 52-dim raw feature vector from a metrics dict.
        Missing keys are filled with 0.0.
        NaN and Inf values are sanitized to 0.0.
        """
        raw = []
        for feat in FEATURE_NAMES_52:
            v = metrics.get(feat, 0.0)
            try:
                v = float(v)
            except (ValueError, TypeError):
                v = 0.0
            # Kill NaN / Inf
            if v != v or v == float('inf') or v == float('-inf'):
                v = 0.0
            raw.append(v)

        vec = np.array(raw, dtype=np.float64).reshape(1, -1)

        nonzero = int((vec != 0).sum())
        if nonzero < 3:
            log.debug(
                f"[IDS] Feature vector sparse: {nonzero}/52 non-zero — "
                "many metrics may be missing from the API payload."
            )
        return vec

    # ------------------------------------------------------------------
    def predict_raw(self, metrics: dict):
        """
        Full inference pipeline — LR-primary ensemble.

        The LightGBM model is overtrained on Normal Traffic and always
        outputs BENIGN. The Logistic Regression model (95.6% test accuracy)
        is therefore the PRIMARY classifier.

        Ensemble logic:
          1. If LR and LGB agree → use that class (high confidence)
          2. If LR says attack but LGB says BENIGN → trust LR (LGB is broken)
          3. If LGB says attack → very noteworthy, boost confidence

        Returns:
            ids_class   str   — e.g. "DDoS", "Normal Traffic"
            rl_family   str   — RL-env family: BENIGN/DDoS/PortScan/Bot/Brute
            confidence  float — max ensemble probability for predicted class
            threat_score float — 0 if benign, scaled confidence if attack
            probs_7     np.ndarray[7] — full 7-class probability vector
            both_agree  bool  — LGB and LR agree on the prediction
        """
        vec       = self._build_vector(metrics)
        vec_sel   = self.selector.transform(vec)

        # Sanitize after selection (catch any residual NaN/Inf)
        vec_sel   = np.nan_to_num(vec_sel, nan=0.0, posinf=0.0, neginf=0.0)

        vec_sc    = self.scaler.transform(vec_sel)

        # Sanitize after scaling too
        vec_sc    = np.nan_to_num(vec_sc, nan=0.0, posinf=0.0, neginf=0.0)

        # --- Individual model probabilities (7-class) ---
        lgb_proba = self.lgb.predict_proba(vec_sc)[0].astype(np.float64)
        lr_proba  = self.lr.predict_proba(vec_sc)[0].astype(np.float64)

        lgb_idx   = int(np.argmax(lgb_proba))
        lr_idx    = int(np.argmax(lr_proba))
        both_agree = (lgb_idx == lr_idx)

        lgb_class = IDS_CLASSES[lgb_idx]
        lr_class  = IDS_CLASSES[lr_idx]

        # ── LR-primary ensemble logic ─────────────────────────────────
        # The LGB model is broken (always says Normal Traffic).
        # LR is the decision-maker.

        if both_agree:
            # Both agree — high confidence, use combined probabilities
            ensemble = 0.50 * lr_proba + 0.50 * lgb_proba
            ensemble /= ensemble.sum() + 1e-9
        elif lr_idx != BENIGN_IDX and lgb_idx == BENIGN_IDX:
            # LR says attack, LGB says BENIGN — trust LR entirely
            # This is the most common case due to LGB being broken
            ensemble = lr_proba.copy()
        elif lgb_idx != BENIGN_IDX and lr_idx == BENIGN_IDX:
            # LGB says attack but LR says BENIGN
            # LGB is known to be unreliable — only trust it if LR also
            # shows SOME attack signal (sum of non-BENIGN probs > 20%)
            lr_attack_mass = 1.0 - float(lr_proba[BENIGN_IDX])
            if lr_attack_mass > 0.20:
                # LR is uncertain — LGB's attack signal tips the balance
                ensemble = 0.60 * lr_proba + 0.40 * lgb_proba
                ensemble /= ensemble.sum() + 1e-9
            else:
                # LR is confident it's BENIGN — ignore LGB's false alarm
                ensemble = lr_proba.copy()
        else:
            # Both say attack but different types — weighted average, LR primary
            ensemble = 0.80 * lr_proba + 0.20 * lgb_proba
            ensemble /= ensemble.sum() + 1e-9

        ens_idx    = int(np.argmax(ensemble))
        ids_class  = IDS_CLASSES[ens_idx]
        rl_family  = CLS_FAMILY.get(ids_class, "BENIGN")
        confidence = float(ensemble[ens_idx])

        if ens_idx == BENIGN_IDX:
            threat_score = 0.0
        else:
            # Attack detected — compute threat score
            boost = 1.2 if both_agree else 1.0
            threat_score = min(confidence * boost, 1.0)

        # Store individual model results for dashboard/debugging
        self._last_lgb_class = lgb_class
        self._last_lr_class  = lr_class
        self._last_agree     = both_agree

        log.info(
            f"[IDS] LR={lr_class}({lr_proba[lr_idx]:.3f}) "
            f"LGB={lgb_class}({lgb_proba[lgb_idx]:.3f}) "
            f"-> {ids_class}({confidence:.3f}) "
            f"agree={both_agree} threat={threat_score:.3f}"
        )
        return ids_class, rl_family, confidence, threat_score, ensemble, both_agree

    # ------------------------------------------------------------------
    def predict(self, metrics: dict) -> tuple:
        """
        Convenience wrapper.

        Returns: (class_name, confidence, threat_score, both_agree)
        Where class_name is the RL-family name (BENIGN/DDoS/PortScan/Bot/Brute).
        """
        try:
            ids_class, rl_family, confidence, threat_score, _, both_agree = \
                self.predict_raw(metrics)
            return rl_family, confidence, threat_score, both_agree
        except Exception as e:
            log.error(f"IntrusionPredictor.predict() error: {e}", exc_info=True)
            return "BENIGN", 1.0, 0.0, True

    # ------------------------------------------------------------------
    def build_rl_probs(self, metrics: dict) -> np.ndarray:
        """
        Returns a 5-dim probability vector matching the RL environment's
        class ordering: [BENIGN, Bot, DDoS, PortScan, Brute]

        Used by StateBuilder.get_last_probs() to drive the reward function.
        Calls predict_raw() internally — prefer build_rl_probs_from_vec()
        if you already have probs_7 to avoid double inference.
        """
        try:
            ids_class, rl_family, confidence, threat_score, probs_7, both_agree = \
                self.predict_raw(metrics)
            return self.build_rl_probs_from_vec(probs_7)

        except Exception as e:
            log.error(f"build_rl_probs error: {e}", exc_info=True)
            return np.array([1.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32)

    # ------------------------------------------------------------------
    def build_rl_probs_from_vec(self, probs_7: np.ndarray) -> np.ndarray:
        """
        Map a pre-computed 7-class IDS probability vector to the 5-class
        RL bucket WITHOUT calling predict_raw() again.

        Use this inside _classify() to avoid double inference per step.

        RL: [0=BENIGN, 1=Bot, 2=DDoS, 3=PortScan, 4=Brute]
        """
        _IDS_TO_RL = {
            "Normal Traffic": 0,   # BENIGN
            "Bots":           1,   # Bot
            "Web Attacks":    1,   # Bot (closest RL class)
            "DDoS":           2,   # DDoS
            "DoS":            2,   # DDoS
            "Port Scanning":  3,   # PortScan
            "Brute Force":    4,   # Brute
        }
        rl_probs = np.zeros(5, dtype=np.float32)
        for i, cls in enumerate(IDS_CLASSES):
            rl_idx = _IDS_TO_RL.get(cls, 0)
            rl_probs[rl_idx] += float(probs_7[i])
        rl_probs /= rl_probs.sum() + 1e-9
        return rl_probs

    # ------------------------------------------------------------------
    def reward_signal(self, metrics: dict) -> dict:
        """
        Full reward signal dict — used by StateBuilder and the environment.
        """
        try:
            ids_class, rl_family, confidence, threat_score, probs_7, both_agree = \
                self.predict_raw(metrics)
            cl = rl_family.lower()
            return {
                # IDS model details
                "ids_class":    ids_class,
                "rl_family":    rl_family,
                "lgb_class":    self._last_lgb_class,
                "lr_class":     self._last_lr_class,
                "models_agree": self._last_agree,
                # Backward-compat with old ThreatPredictor
                "class_name":   rl_family,
                "confidence":   confidence,
                "threat_score": threat_score,
                "both_agree":   both_agree,
                "is_benign":    (rl_family == "BENIGN"),
                "is_ddos":      any(k in cl for k in ["ddos", "dos"]),
                "is_brute":     any(k in cl for k in ["brute", "bot"]),
                "is_exploit":   any(k in cl for k in ["bot", "web"]),
                "is_scan":      any(k in cl for k in ["scan", "portscan"]),
                "certainty":    confidence * (1.2 if both_agree else 0.9),
            }
        except Exception as e:
            log.error(f"reward_signal error: {e}", exc_info=True)
            return {
                "ids_class": "Normal Traffic", "rl_family": "BENIGN",
                "lgb_class": "Normal Traffic", "lr_class": "Normal Traffic",
                "models_agree": True,
                "class_name": "BENIGN", "confidence": 1.0,
                "threat_score": 0.0, "both_agree": True,
                "is_benign": True, "is_ddos": False,
                "is_brute": False, "is_exploit": False,
                "is_scan": False, "certainty": 1.0,
            }

    # ------------------------------------------------------------------
    # Backward-compat shim so StateBuilder can call _build_vector directly
    def _build_feature_vec(self, metrics: dict) -> np.ndarray:
        return self._build_vector(metrics)

    @property
    def feature_names(self) -> list:
        return FEATURE_NAMES_52

    @property
    def n_features(self) -> int:
        return 52
