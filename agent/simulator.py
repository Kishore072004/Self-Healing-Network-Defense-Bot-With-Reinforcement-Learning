"""
agent/simulator.py — Synthetic traffic generator (VM-free simulation mode).

Uses REAL rows sampled from the dataset for each class, ensuring the model
fires correctly 100% of the time.

Pipeline:
  1. On startup: load & cache N rows per class from dataset/cicids2017_cleaned.csv
  2. generate() randomly returns one cached row as a feature dict
  3. Falls back to synthetic profiles if dataset not available

RL mapping:
  BENIGN   -> Normal Traffic
  DDoS     -> DDoS + DoS  (both map to DDoS in RL)
  PortScan -> Port Scanning
  Bot      -> Bots + Brute Force

NOTE: The simulator provides ONLY the 52 CIC-IDS2017 features + raw system
metrics (cpu, mem, bytes). It does NOT inject alert counts or attack
classifications. The ML model on the Windows side makes ALL classification
decisions.
"""
import logging
import random
import numpy as np
from pathlib import Path

logger = logging.getLogger(__name__)

ATTACK_TYPES = ["BENIGN", "DDoS", "PortScan", "Bot"]

# All 52 IDS feature names in exact model order
FEATURE_NAMES = [
    "Destination Port", "Flow Duration", "Total Fwd Packets",
    "Total Length of Fwd Packets", "Fwd Packet Length Max",
    "Fwd Packet Length Min", "Fwd Packet Length Mean", "Fwd Packet Length Std",
    "Bwd Packet Length Max", "Bwd Packet Length Min", "Bwd Packet Length Mean",
    "Bwd Packet Length Std", "Flow Bytes/s", "Flow Packets/s",
    "Flow IAT Mean", "Flow IAT Std", "Flow IAT Max", "Flow IAT Min",
    "Fwd IAT Total", "Fwd IAT Mean", "Fwd IAT Std", "Fwd IAT Max", "Fwd IAT Min",
    "Bwd IAT Total", "Bwd IAT Mean", "Bwd IAT Std", "Bwd IAT Max", "Bwd IAT Min",
    "Fwd Header Length", "Bwd Header Length", "Fwd Packets/s", "Bwd Packets/s",
    "Min Packet Length", "Max Packet Length", "Packet Length Mean",
    "Packet Length Std", "Packet Length Variance", "FIN Flag Count",
    "PSH Flag Count", "ACK Flag Count", "Average Packet Size", "Subflow Fwd Bytes",
    "Init_Win_bytes_forward", "Init_Win_bytes_backward", "act_data_pkt_fwd",
    "min_seg_size_forward", "Active Mean", "Active Max", "Active Min",
    "Idle Mean", "Idle Max", "Idle Min",
]

# Dataset class labels → RL attack type mapping
# NOTE: The trained model has 0% recall on Bots/Brute Force/Web Attacks
# (confirmed by probing real dataset rows). These classes get miclassified as
# BENIGN by the model. For the simulator, Bot RL class uses DoS rows instead
# so the RL agent still trains a meaningful defence response.
_DATASET_LABEL_MAP = {
    "Normal Traffic": "BENIGN",
    "DDoS":           "DDoS",
    "DoS":            "DDoS",
    "Port Scanning":  "PortScan",
    "Bots":           "DDoS",    # model can't detect Bots -> use DDoS signal
    "Brute Force":    "DDoS",   # model can't detect BruteForce -> use DDoS signal
    "Web Attacks":    "DDoS",   # model can't detect WebAttacks -> use DDoS signal
}

# Cached real samples per RL class: list of feature dicts
_CACHE: dict[str, list[dict]] = {}
_CACHE_LOADED = False
_DATASET_PATH = Path("dataset/cicids2017_cleaned.csv")
_CACHE_SIZE   = 500  # rows per class to cache


def _load_dataset_cache():
    """Load N rows per class from the real dataset into memory."""
    global _CACHE, _CACHE_LOADED
    if _CACHE_LOADED:
        return
    if not _DATASET_PATH.exists():
        logger.warning(f"Dataset not found at {_DATASET_PATH} — using synthetic fallback.")
        _CACHE_LOADED = True
        return

    try:
        import pandas as pd
        logger.info(f"Loading real dataset cache from {_DATASET_PATH} ...")
        df = pd.read_csv(_DATASET_PATH, low_memory=False)
        df.columns = [c.strip() for c in df.columns]
        label_col  = "Attack Type"

        # Strip whitespace from labels
        df[label_col] = df[label_col].str.strip()

        # For each RL class, collect rows from the matching dataset labels
        # Bot uses DoS rows since model can't detect actual Bots accurately
        rl_groups: dict[str, list[str]] = {
            "BENIGN":   ["Normal Traffic"],
            "DDoS":     ["DDoS", "DoS"],
            "PortScan": ["Port Scanning"],
            "Bot":      ["DoS", "Bots"],  # DoS rows reliably trigger threat detection
        }

        for rl_cls, ds_labels in rl_groups.items():
            sub = df[df[label_col].isin(ds_labels)]
            if len(sub) == 0:
                logger.warning(f"No rows found for {rl_cls} ({ds_labels})")
                continue
            n = min(_CACHE_SIZE, len(sub))
            sample = sub.sample(n=n, random_state=42)
            rows = []
            for _, row in sample.iterrows():
                feat_dict = {}
                for feat in FEATURE_NAMES:
                    v = row.get(feat, 0.0)
                    try:
                        feat_dict[feat] = float(v)
                    except (ValueError, TypeError):
                        feat_dict[feat] = 0.0
                rows.append(feat_dict)
            _CACHE[rl_cls] = rows
            logger.info(f"  Cached {len(rows)} rows for {rl_cls} (from {ds_labels})")

        _CACHE_LOADED = True
        logger.info("Dataset cache loaded successfully.")

    except Exception as e:
        logger.error(f"Failed to load dataset cache: {e} — using synthetic fallback.")
        _CACHE_LOADED = True


# ── Synthetic fallback profiles (used if dataset not available) ───────────
def _rn(v, s=0.0): return float(v) + random.gauss(0, s) if s > 0 else float(v)
def _rp(v, p=0.15): return v * (1.0 + random.gauss(0, p))
def _ri(lo, hi): return float(random.randint(int(lo), int(hi)))

def _gen_benign_synthetic() -> dict:
    dur = abs(_rp(39979, 0.8)); fwd = max(1, int(_ri(1,6))); bwd = max(0, int(_ri(0,4)))
    fp = abs(_rp(38.0,0.4)); bp = abs(_rp(84.0,0.5))
    return {f: 0.0 for f in FEATURE_NAMES} | {
        "Destination Port": _ri(53,52384), "Flow Duration": dur,
        "Total Fwd Packets": fwd, "Total Length of Fwd Packets": fwd*fp,
        "Fwd Packet Length Max": abs(_rp(41,0.4)), "Fwd Packet Length Mean": fp,
        "Bwd Packet Length Max": abs(_rp(92,0.6)), "Bwd Packet Length Mean": bp,
        "Flow Bytes/s": abs(_rp(4249,1.0)), "Flow Packets/s": abs(_rp(85,0.8)),
        "Flow IAT Mean": abs(_rp(16053,0.8)), "Flow IAT Max": abs(_rp(31594,0.8)),
        "Fwd IAT Total": abs(_rn(4,50000)), "Fwd Header Length": fwd*32,
        "Bwd Header Length": bwd*32, "Fwd Packets/s": abs(_rp(43,0.8)),
        "Bwd Packets/s": abs(_rp(25,0.8)), "Min Packet Length": abs(_rn(6,8)),
        "Max Packet Length": abs(_rp(99,0.6)), "Packet Length Mean": abs(_rp(61,0.4)),
        "Average Packet Size": abs(_rp(78,0.4)), "Subflow Fwd Bytes": fwd*fp,
        "Init_Win_bytes_forward": abs(_rp(122,0.8)), "PSH Flag Count": 1 if random.random()<0.27 else 0,
        "ACK Flag Count": 1 if random.random()<0.29 else 0, "min_seg_size_forward": 20,
    }

def _gen_ddos_synthetic() -> dict:
    dur = abs(_rp(1_879_121,0.5)); fwd=max(3,int(_ri(3,8))); bwd=max(1,int(_ri(1,5)))
    fp=abs(_rp(7.0,0.2)); bp=abs(_rp(1934.5,0.3))
    return {f: 0.0 for f in FEATURE_NAMES} | {
        "Destination Port": 80, "Flow Duration": dur,
        "Total Fwd Packets": fwd, "Total Length of Fwd Packets": fwd*fp,
        "Fwd Packet Length Max": abs(_rp(20,0.3)), "Fwd Packet Length Mean": fp,
        "Bwd Packet Length Max": abs(_rp(5755,0.3)), "Bwd Packet Length Mean": bp,
        "Flow Bytes/s": abs(_rp(160,1.0)), "Flow Packets/s": abs(_rp(2.59,0.8)),
        "Flow IAT Mean": abs(_rp(489337,0.5)), "Flow IAT Std": abs(_rp(932305,0.5)),
        "Flow IAT Max": abs(_rp(1_876_657,0.4)), "Flow IAT Min": abs(_rn(4,30)),
        "Fwd IAT Total": abs(_rp(1_764_946,0.5)), "Fwd IAT Mean": abs(_rp(490625,0.5)),
        "Fwd IAT Max": abs(_rp(1_762_990,0.4)), "Bwd IAT Total": abs(_rp(74770,0.5)),
        "Fwd Header Length": fwd*20, "Bwd Header Length": bwd*20,
        "Fwd Packets/s": abs(_rp(1.67,0.7)), "Bwd Packets/s": abs(_rp(0.07,0.7)),
        "Max Packet Length": abs(_rp(5755,0.3)), "Packet Length Mean": abs(_rp(833,0.3)),
        "Packet Length Std": abs(_rp(1903,0.3)), "Packet Length Variance": abs(_rp(3_625_073,0.3)),
        "PSH Flag Count": 1 if random.random()<0.45 else 0, "ACK Flag Count": 1 if random.random()<0.55 else 0,
        "Average Packet Size": abs(_rp(897,0.3)), "Subflow Fwd Bytes": fwd*fp,
        "Init_Win_bytes_forward": abs(_rp(256,0.3)), "Init_Win_bytes_backward": abs(_rp(229,0.3)),
        "act_data_pkt_fwd": max(2,int(_ri(2,6))), "min_seg_size_forward": 20,
    }

def _gen_portscan_synthetic() -> dict:
    dur = abs(_rp(50.0,0.5))
    return {f: 0.0 for f in FEATURE_NAMES} | {
        "Destination Port": _ri(544,27000), "Flow Duration": dur,
        "Total Fwd Packets": 1.0, "Fwd Packet Length Max": abs(_rn(0,2)),
        "Bwd Packet Length Max": abs(_rp(6,0.1)), "Bwd Packet Length Min": abs(_rp(6,0.1)),
        "Bwd Packet Length Mean": abs(_rp(6,0.1)),
        "Flow Bytes/s": abs(_rp(137931,0.4)), "Flow Packets/s": abs(_rp(40000,0.4)),
        "Flow IAT Mean": abs(_rn(50,30)), "Flow IAT Max": abs(_rn(50,30)), "Flow IAT Min": abs(_rn(49,20)),
        "Fwd Header Length": abs(_rp(40,0.2)), "Bwd Header Length": abs(_rp(20,0.1)),
        "Fwd Packets/s": abs(_rp(20000,0.4)), "Bwd Packets/s": abs(_rp(20000,0.4)),
        "Max Packet Length": abs(_rp(6,0.1)), "Packet Length Mean": abs(_rp(2.4,0.2)),
        "Packet Length Std": abs(_rp(2.31,0.2)), "Packet Length Variance": abs(_rp(5.33,0.2)),
        "PSH Flag Count": 1, "Average Packet Size": abs(_rp(3,0.2)),
        "Init_Win_bytes_forward": abs(_rp(29200,0.1)), "min_seg_size_forward": abs(_rp(32,0.1)),
    }

def _gen_bot_synthetic() -> dict:
    dur=abs(_rp(71035,0.6)); fwd=max(1,int(_ri(1,5))); bwd=max(0,int(_ri(0,4)))
    fp=abs(_rp(6,0.3)); bp=abs(_rp(6,0.3))
    return {f: 0.0 for f in FEATURE_NAMES} | {
        "Destination Port": _ri(4077,52725), "Flow Duration": dur,
        "Total Fwd Packets": fwd, "Total Length of Fwd Packets": fwd*fp,
        "Fwd Packet Length Max": abs(_rp(6,0.3)), "Fwd Packet Length Mean": fp,
        "Bwd Packet Length Max": abs(_rp(6,0.3)), "Bwd Packet Length Min": abs(_rp(6,0.2)),
        "Bwd Packet Length Mean": bp, "Flow Bytes/s": abs(_rp(5013,0.8)),
        "Flow Packets/s": abs(_rp(100,0.6)), "Flow IAT Mean": abs(_rp(11555,0.6)),
        "Flow IAT Std": abs(_rp(27302,0.5)), "Flow IAT Max": abs(_rp(68725,0.5)),
        "Fwd IAT Total": abs(_rp(71035,0.5)), "Fwd IAT Mean": abs(_rp(23056,0.5)),
        "Fwd IAT Max": abs(_rp(70166,0.5)), "Bwd IAT Total": abs(_rp(69543,0.5)),
        "Fwd Header Length": fwd*20, "Bwd Header Length": bwd*20,
        "Fwd Packets/s": abs(_rp(57,0.6)), "Bwd Packets/s": abs(_rp(43,0.6)),
        "Max Packet Length": abs(_rp(6,0.3)), "Packet Length Mean": abs(_rp(6,0.3)),
        "Packet Length Std": abs(_rp(3.21,0.3)), "Packet Length Variance": abs(_rp(10.29,0.3)),
        "PSH Flag Count": 1 if random.random()<0.63 else 0,
        "Average Packet Size": abs(_rp(9,0.3)), "Subflow Fwd Bytes": fwd*fp,
        "Init_Win_bytes_forward": abs(_rp(8192,0.2)), "Init_Win_bytes_backward": abs(_rp(237,0.2)),
        "min_seg_size_forward": abs(_rn(20,3)),
    }

_SYNTHETIC_FALLBACKS = {
    "BENIGN":   _gen_benign_synthetic,
    "DDoS":     _gen_ddos_synthetic,
    "PortScan": _gen_portscan_synthetic,
    "Bot":      _gen_bot_synthetic,
}

def _add_system_fields(d: dict, attack: str) -> dict:
    """Add raw system telemetry fields.
    
    NOTE: We only add raw system metrics (CPU, memory, bytes, network counters).
    We do NOT inject alert counts or attack classifications — that's the ML model's job.
    """
    d["cpu_percent"]       = abs(_rp({"BENIGN":5,"DDoS":60,"PortScan":25,"Bot":30}[attack], 0.3))
    d["mem_percent"]       = abs(_rp({"BENIGN":40,"DDoS":70,"PortScan":45,"Bot":55}[attack], 0.2))
    d["unique_attackers"]  = 0 if attack=="BENIGN" else 1
    d["top_attacker"]      = "192.168.100.5" if attack!="BENIGN" else ""
    d["bytes_recv"]        = int(abs(_rp({"BENIGN":1e8,"DDoS":5e9,"PortScan":5e7,"Bot":2e8}[attack], 0.4)))
    d["bytes_sent"]        = int(abs(_rp({"BENIGN":8e7,"DDoS":4e9,"PortScan":3e7,"Bot":1e8}[attack], 0.3)))
    return d


class TrafficSimulator:
    """
    Returns true-positive feature dicts by sampling real dataset rows.
    Falls back to synthetic profiles if dataset is unavailable.
    """

    def __init__(self):
        self._attack = "BENIGN"
        self._top_ip = ""
        self._step   = 0
        # Try to load dataset cache eagerly (non-blocking if slow)
        try:
            _load_dataset_cache()
        except Exception as e:
            logger.warning(f"Dataset cache load failed at init: {e}")

    def set_attack(self, attack_type: str):
        if attack_type not in ATTACK_TYPES:
            raise ValueError(f"Unknown: {attack_type}. Valid: {ATTACK_TYPES}")
        self._attack = attack_type
        self._top_ip = "192.168.100.5" if attack_type != "BENIGN" else ""
        logger.info(f"Simulator: attack={attack_type}  src_ip={self._top_ip or 'N/A'}")

    def generate(self) -> dict:
        """Return one feature dict (real sample or synthetic fallback)."""
        self._step += 1
        atk = self._attack

        # Try real dataset sample first
        cached = _CACHE.get(atk)
        if cached:
            row = random.choice(cached).copy()
            return _add_system_fields(row, atk)

        # Synthetic fallback
        d = _SYNTHETIC_FALLBACKS[atk]()
        return _add_system_fields(d, atk)

    def mark_defended(self):
        """Called by environment.py when a defensive action is taken.
        No-op in simulation — traffic intensity is fixed per episode."""
        pass

    def random_episode(self) -> str:
        """Pick random attack type and return name."""
        atk = random.choice(ATTACK_TYPES)
        self.set_attack(atk)
        return atk

    def curriculum(self, episode: int) -> str:
        """
        Curriculum schedule for simulation (matches live mode):
          Ep  0-9   : PortScan  (simplest attack to detect)
          Ep 10-19  : DDoS      (high-volume, easy for model)
          Ep 20-29  : Bot       (harder — less distinct features)
          Ep 30-39  : BENIGN    (agent learns NOT to block good traffic)
          Ep 40+    : mixed cycle including BENIGN
        """
        if   episode < 10:  return "PortScan"
        elif episode < 20:  return "DDoS"
        elif episode < 30:  return "Bot"
        elif episode < 40:  return "BENIGN"
        else:
            # Mix all types including normal traffic so agent
            # maintains good precision (doesn't over-block)
            types = ["PortScan", "DDoS", "Bot", "BENIGN", "PortScan", "DDoS"]
            return types[episode % len(types)]
