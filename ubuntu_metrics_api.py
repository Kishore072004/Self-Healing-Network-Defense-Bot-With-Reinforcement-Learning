#!/usr/bin/env python3
"""
ubuntu_metrics_api.py — Live CIC-IDS2017 feature extractor.
Runs ON Ubuntu VM at port 8080.

Uses continuous tshark subprocess for real-time packet capture.
Accumulates packets in a 30-second rolling buffer to produce
statistically meaningful flows matching CIC-IDS2017 conventions.

Pipeline:  tshark -> rolling buffer -> per-flow features -> pick most suspicious -> JSON API
"""
import os, sys, time, math, threading, subprocess, re, traceback
from collections import defaultdict, Counter
from datetime import datetime

# ── Dependency install (robust for Ubuntu 22.04 / 24.04) ─────────
def _ensure_deps():
    """Install flask + psutil if missing."""
    missing = []
    try:
        import flask
    except ImportError:
        missing.append("flask")
    try:
        import psutil
    except ImportError:
        missing.append("psutil")
    if missing:
        pkgs = " ".join(missing)
        print(f"[*] Installing missing packages: {pkgs}")
        # Try multiple install methods for compatibility
        for cmd in [
            f"{sys.executable} -m pip install {pkgs} --quiet --break-system-packages",
            f"{sys.executable} -m pip install {pkgs} --quiet",
            f"pip3 install {pkgs} --quiet --break-system-packages",
            f"pip3 install {pkgs} --quiet",
        ]:
            rc = os.system(cmd)
            if rc == 0:
                print(f"[*] Installed OK: {pkgs}")
                break
        else:
            print(f"[ERROR] Could not install {pkgs}. Trying anyway...")

_ensure_deps()

from flask import Flask, jsonify, make_response
import psutil

app = Flask(__name__)
_lock = threading.Lock()
_state = {}

# ── Rolling packet buffer (thread-safe) ──────────────────────────
_buffer_lock = threading.Lock()
_packet_buffer = []
_BUFFER_WINDOW = 30.0   # seconds — increased from 10s for better flow stats

# ── Detect network interface ─────────────────────────────────────
def _get_interface():
    try:
        out = subprocess.check_output(
            "ip route | grep default | awk '{print $5}' | head -1",
            shell=True, timeout=5
        ).decode().strip()
        if out:
            return out
    except Exception:
        pass
    # Fallback: try common interface names
    for iface in ["ens33", "eth0", "ens160", "enp0s3"]:
        try:
            subprocess.check_output(f"ip link show {iface}", shell=True,
                                    timeout=3, stderr=subprocess.DEVNULL)
            return iface
        except Exception:
            continue
    return "ens33"

def _get_local_ip(iface):
    try:
        out = subprocess.check_output(
            f"ip -4 addr show {iface} | grep -oP '(?<=inet\\s)\\d+(\\.\\d+){{3}}'",
            shell=True, timeout=5
        ).decode().strip().split('\n')[0]
        if out:
            return out
    except Exception:
        pass
    return "192.168.100.10"

IFACE = _get_interface()
LOCAL_IP = _get_local_ip(IFACE)

# ── Helpers ───────────────────────────────────────────────────────
def _safe(val, default=0.0):
    try:
        v = str(val).strip()
        if not v or v.lower() in ("", "nan", "inf", "-inf", "infinity"):
            return default
        if v == "True": return 1.0
        if v == "False": return 0.0
        return float(v)
    except (ValueError, TypeError):
        return default

def _sanitize(v):
    if v != v:
        return 0.0
    if v == float('inf') or v == float('-inf'):
        return 0.0
    return float(v)

def _std(values, mean):
    if len(values) < 2:
        return 0.0
    return math.sqrt(sum((v - mean) ** 2 for v in values) / len(values))

def _iats_us(times):
    """Inter-arrival time stats in MICROSECONDS."""
    if len(times) < 2:
        return 0.0, 0.0, 0.0, 0.0, 0.0
    sorted_t = sorted(times)
    iats = [(sorted_t[i+1] - sorted_t[i]) * 1e6 for i in range(len(sorted_t)-1)]
    total = sum(iats)
    mean = total / len(iats)
    std = _std(iats, mean)
    return total, mean, std, max(iats), min(iats)


# ── Continuous tshark capture ─────────────────────────────────────
_tshark_proc = None

def _start_tshark():
    """Start tshark as a continuous live-capture subprocess."""
    global _tshark_proc
    try:
        # Kill any existing tshark processes first
        subprocess.run("pkill -f 'tshark.*-T fields' 2>/dev/null || true",
                       shell=True, timeout=5)
        time.sleep(0.5)

        if _tshark_proc is not None:
            try:
                _tshark_proc.terminate()
                _tshark_proc.wait(timeout=3)
            except Exception:
                try:
                    _tshark_proc.kill()
                except Exception:
                    pass

        # Check if tshark exists
        tshark_path = subprocess.check_output(
            "which tshark 2>/dev/null || echo ''",
            shell=True, timeout=5
        ).decode().strip()

        if not tshark_path:
            print("[ERROR] tshark not found!")
            sys.stdout.flush()
            return False

        print(f"[*] Starting continuous tshark on {IFACE} (path: {tshark_path})")
        sys.stdout.flush()

        cmd = [
            "sudo", tshark_path, "-i", IFACE, "-l",
            "-T", "fields",
            "-e", "frame.time_epoch",
            "-e", "ip.src", "-e", "ip.dst",
            "-e", "ip.proto", "-e", "ip.len",
            "-e", "frame.len",
            "-e", "tcp.srcport", "-e", "tcp.dstport",
            "-e", "tcp.flags.syn", "-e", "tcp.flags.ack",
            "-e", "tcp.flags.fin", "-e", "tcp.flags.push",
            "-e", "tcp.flags.reset", "-e", "tcp.flags.urg",
            "-e", "tcp.window_size_value",
            "-e", "udp.srcport", "-e", "udp.dstport",
            "-E", "separator=|",
            "-Y", "ip",
        ]

        # Use preexec_fn only if available (Linux)
        kwargs = dict(
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,  # capture stderr for debugging
            bufsize=1,
            universal_newlines=True,
        )
        try:
            # Try setpgrp to prevent tshark from being killed when parent exits
            os.setpgrp  # just check it exists
            kwargs['preexec_fn'] = os.setpgrp
        except AttributeError:
            pass

        _tshark_proc = subprocess.Popen(cmd, **kwargs)

        # Wait briefly and check if it crashed immediately
        time.sleep(1)
        if _tshark_proc.poll() is not None:
            stderr_out = _tshark_proc.stderr.read() if _tshark_proc.stderr else ""
            print(f"[ERROR] tshark exited immediately! rc={_tshark_proc.returncode}")
            print(f"[ERROR] stderr: {stderr_out}")
            sys.stdout.flush()
            _tshark_proc = None
            return False

        print(f"[*] tshark started successfully (PID {_tshark_proc.pid})")
        sys.stdout.flush()
        return True

    except FileNotFoundError:
        print("[ERROR] tshark binary not found")
        sys.stdout.flush()
        return False
    except Exception as e:
        print(f"[tshark error] {e}")
        traceback.print_exc()
        sys.stdout.flush()
        return False


def _tshark_reader():
    """Background thread: read tshark stdout lines into rolling buffer."""
    global _tshark_proc
    while True:
        proc = _tshark_proc
        if proc is None or proc.poll() is not None:
            time.sleep(2)
            continue
        try:
            line = proc.stdout.readline()
            if not line:
                time.sleep(0.1)
                continue
            parts = line.strip().split('|')
            if len(parts) < 6:
                continue
            pkt = {
                "frame.time_epoch":      parts[0] or "0",
                "ip.src":                parts[1] or "",
                "ip.dst":                parts[2] or "",
                "ip.proto":              parts[3] or "6",
                "ip.len":                parts[4] or "0",
                "frame.len":             parts[5] or "0",
                "tcp.srcport":           parts[6] if len(parts) > 6 else "0",
                "tcp.dstport":           parts[7] if len(parts) > 7 else "0",
                "tcp.flags.syn":         parts[8] if len(parts) > 8 else "0",
                "tcp.flags.ack":         parts[9] if len(parts) > 9 else "0",
                "tcp.flags.fin":         parts[10] if len(parts) > 10 else "0",
                "tcp.flags.psh":         parts[11] if len(parts) > 11 else "0",
                "tcp.flags.rst":         parts[12] if len(parts) > 12 else "0",
                "tcp.flags.urg":         parts[13] if len(parts) > 13 else "0",
                "tcp.window_size_value": parts[14] if len(parts) > 14 else "0",
                "udp.srcport":           parts[15] if len(parts) > 15 else "0",
                "udp.dstport":           parts[16] if len(parts) > 16 else "0",
            }
            with _buffer_lock:
                _packet_buffer.append(pkt)
        except Exception:
            time.sleep(0.1)


def _get_buffered_packets():
    """Get packets from the rolling buffer, pruning old ones."""
    now = time.time()
    cutoff = now - _BUFFER_WINDOW
    with _buffer_lock:
        fresh = [p for p in _packet_buffer
                 if _safe(p.get("frame.time_epoch"), 0) > cutoff]
        _packet_buffer.clear()
        _packet_buffer.extend(fresh)
        return list(fresh)


# ── CIC-IDS2017 Feature Computation ──────────────────────────────
def _compute_cicids_features(packets):
    """
    Compute CIC-IDS2017-style features from tshark packets.
    Returns dict with exact 52 feature names matching the trained model.
    """
    if not packets:
        return _zero_state()

    # Group into flows (5-tuple)
    flows = defaultdict(list)
    for pkt in packets:
        src   = pkt.get("ip.src", "")
        dst   = pkt.get("ip.dst", "")
        sport = pkt.get("tcp.srcport") or pkt.get("udp.srcport") or "0"
        dport = pkt.get("tcp.dstport") or pkt.get("udp.dstport") or "0"
        proto = pkt.get("ip.proto", "6")
        if (src, sport) < (dst, dport):
            key = (src, dst, sport, dport, proto)
        else:
            key = (dst, src, dport, sport, proto)
        flows[key].append(pkt)

    if not flows:
        return _zero_state()

    # Compute features PER FLOW
    flow_features = []

    for flow_key, pkts in flows.items():
        # Accept single-packet flows too (port scans are 1-2 packets)
        if len(pkts) < 1:
            continue

        # Determine forward direction: first packet's source is "forward"
        first_src = pkts[0].get("ip.src", "")
        fwd_times, bwd_times = [], []
        fwd_lens, bwd_lens = [], []
        fwd_win, bwd_win = None, None
        syn_cnt = fin_cnt = psh_cnt = ack_cnt = urg_cnt = rst_cnt = 0
        fwd_hdr_total = bwd_hdr_total = 0

        for p in pkts:
            t  = _safe(p.get("frame.time_epoch"), 0.0)
            ln = _safe(p.get("ip.len"), 0.0)
            if ln == 0:
                ln = _safe(p.get("frame.len"), 0.0)
                if ln > 14:
                    ln -= 14

            is_fwd = (p.get("ip.src", "") == first_src)

            if is_fwd:
                fwd_times.append(t)
                fwd_lens.append(ln)
                fwd_hdr_total += 20
                if fwd_win is None:
                    w = _safe(p.get("tcp.window_size_value"), 0)
                    if w > 0:
                        fwd_win = w
            else:
                bwd_times.append(t)
                bwd_lens.append(ln)
                bwd_hdr_total += 20
                if bwd_win is None:
                    w = _safe(p.get("tcp.window_size_value"), 0)
                    if w > 0:
                        bwd_win = w

            if _safe(p.get("tcp.flags.syn")) == 1: syn_cnt += 1
            if _safe(p.get("tcp.flags.fin")) == 1: fin_cnt += 1
            if _safe(p.get("tcp.flags.psh")) == 1: psh_cnt += 1
            if _safe(p.get("tcp.flags.ack")) == 1: ack_cnt += 1
            if _safe(p.get("tcp.flags.urg")) == 1: urg_cnt += 1
            if _safe(p.get("tcp.flags.rst")) == 1: rst_cnt += 1

        n_fwd = len(fwd_lens)
        n_bwd = len(bwd_lens)
        n_total = n_fwd + n_bwd
        if n_total < 1:
            continue

        fwd_bytes = sum(fwd_lens)
        bwd_bytes = sum(bwd_lens)
        total_bytes = fwd_bytes + bwd_bytes

        all_times = fwd_times + bwd_times
        if len(all_times) >= 2:
            dur_s = max(all_times) - min(all_times)
            dur_s = max(dur_s, 0.0)
            dur_us = dur_s * 1e6
        else:
            dur_s = 0.0  # single packet → 0 duration
            dur_us = 0.0

        fwd_iat_total, fwd_iat_mean, fwd_iat_std, fwd_iat_max, fwd_iat_min = _iats_us(fwd_times)
        bwd_iat_total, bwd_iat_mean, bwd_iat_std, bwd_iat_max, bwd_iat_min = _iats_us(bwd_times)
        _, flow_iat_mean, flow_iat_std, flow_iat_max, flow_iat_min = _iats_us(all_times)

        fwd_mean = fwd_bytes / n_fwd if n_fwd > 0 else 0.0
        fwd_std_ = _std(fwd_lens, fwd_mean)
        fwd_max  = max(fwd_lens) if fwd_lens else 0.0
        fwd_min  = min(fwd_lens) if fwd_lens else 0.0
        bwd_mean = bwd_bytes / n_bwd if n_bwd > 0 else 0.0
        bwd_std_ = _std(bwd_lens, bwd_mean)
        bwd_max  = max(bwd_lens) if bwd_lens else 0.0
        bwd_min  = min(bwd_lens) if bwd_lens else 0.0

        all_lens = fwd_lens + bwd_lens
        pkt_mean = total_bytes / n_total if n_total > 0 else 0.0
        pkt_std  = _std(all_lens, pkt_mean)
        pkt_var  = pkt_std ** 2
        pkt_max  = max(all_lens) if all_lens else 0.0
        pkt_min  = min(all_lens) if all_lens else 0.0

        flow_bps = (total_bytes / dur_s) if dur_s > 0.0 else 0.0
        flow_pps = (n_total / dur_s) if dur_s > 0.0 else 0.0
        fwd_pps  = (n_fwd / dur_s) if dur_s > 0.0 else 0.0
        bwd_pps  = (n_bwd / dur_s) if dur_s > 0.0 else 0.0

        # Active/Idle
        active_periods, idle_periods = [], []
        if len(all_times) >= 2:
            st = sorted(all_times)
            active_start = st[0]
            for i in range(len(st) - 1):
                if st[i+1] - st[i] > 1.0:
                    active_periods.append(st[i] - active_start)
                    idle_periods.append(st[i+1] - st[i])
                    active_start = st[i+1]
            active_periods.append(st[-1] - active_start)

        active_mean = (sum(active_periods) / len(active_periods) * 1e6) if active_periods else 0.0
        active_max  = (max(active_periods) * 1e6) if active_periods else 0.0
        active_min  = (min(active_periods) * 1e6) if active_periods else 0.0
        idle_mean   = (sum(idle_periods) / len(idle_periods) * 1e6) if idle_periods else 0.0
        idle_max    = (max(idle_periods) * 1e6) if idle_periods else 0.0
        idle_min    = (min(idle_periods) * 1e6) if idle_periods else 0.0

        src_ip, dst_ip, s_port, d_port, proto = flow_key
        dest_port = int(_safe(d_port if first_src == src_ip else s_port, 0))

        # Init_Win_bytes: use actual captured value, or 0 if none seen
        # DO NOT default to 65535 — that signals "healthy normal" to the model
        # 0 means "unknown/not captured" which is safer
        init_win_fwd = fwd_win if fwd_win is not None else 0
        init_win_bwd = bwd_win if bwd_win is not None else 0

        suspicion = (flow_pps * 0.5 + syn_cnt * 10 + rst_cnt * 5
                     + (50 if pkt_mean < 100 and n_total > 5 else 0) + n_total * 0.1)

        flow_features.append({
            "dest_port": dest_port, "dur_us": dur_us,
            "n_fwd": n_fwd, "n_bwd": n_bwd,
            "fwd_bytes": fwd_bytes, "bwd_bytes": bwd_bytes,
            "fwd_max": fwd_max, "fwd_min": fwd_min,
            "fwd_mean": fwd_mean, "fwd_std": fwd_std_,
            "bwd_max": bwd_max, "bwd_min": bwd_min,
            "bwd_mean": bwd_mean, "bwd_std": bwd_std_,
            "flow_bps": flow_bps, "flow_pps": flow_pps,
            "flow_iat_mean": flow_iat_mean, "flow_iat_std": flow_iat_std,
            "flow_iat_max": flow_iat_max, "flow_iat_min": flow_iat_min,
            "fwd_iat_total": fwd_iat_total, "fwd_iat_mean": fwd_iat_mean,
            "fwd_iat_std": fwd_iat_std, "fwd_iat_max": fwd_iat_max,
            "fwd_iat_min": fwd_iat_min,
            "bwd_iat_total": bwd_iat_total, "bwd_iat_mean": bwd_iat_mean,
            "bwd_iat_std": bwd_iat_std, "bwd_iat_max": bwd_iat_max,
            "bwd_iat_min": bwd_iat_min,
            "fwd_hdr": fwd_hdr_total, "bwd_hdr": bwd_hdr_total,
            "fwd_pps": fwd_pps, "bwd_pps": bwd_pps,
            "pkt_min": pkt_min, "pkt_max": pkt_max,
            "pkt_mean": pkt_mean, "pkt_std": pkt_std, "pkt_var": pkt_var,
            "fin_cnt": fin_cnt, "psh_cnt": psh_cnt, "ack_cnt": ack_cnt,
            "avg_pkt_size": pkt_mean,
            "subfwd_bytes": fwd_bytes,
            "fwd_win": init_win_fwd, "bwd_win": init_win_bwd,
            "act_data_fwd": n_fwd, "min_seg_fwd": 20,
            "active_mean": active_mean, "active_max": active_max,
            "active_min": active_min,
            "idle_mean": idle_mean, "idle_max": idle_max, "idle_min": idle_min,
            "syn_cnt": syn_cnt, "rst_cnt": rst_cnt,
            "n_total": n_total, "src": first_src, "suspicion": suspicion,
        })

    if not flow_features:
        return _zero_state()

    # ── Pure ML metrics compilation ───────────────────────────────────
    all_dst_ports = set(f["dest_port"] for f in flow_features)
    all_src_ips = set(f["src"] for f in flow_features if f["src"] != LOCAL_IP)
    total_syn = sum(f["syn_cnt"] for f in flow_features)
    total_pkts = sum(f["n_total"] for f in flow_features)
    n_flows = len(flow_features)

    # ── Weighted average across ALL flows ─────────────────────────────
    # Instead of picking one flow (which biases toward DDoS), compute
    # the average feature vector weighted by packet count.
    # This way, if 90% of flows are 1-packet SYN scans, the average
    # correctly represents a scan pattern, not a DDoS pattern.
    numeric_keys = [
        "dest_port", "dur_us", "n_fwd", "n_bwd", "fwd_bytes", "bwd_bytes",
        "fwd_max", "fwd_min", "fwd_mean", "fwd_std",
        "bwd_max", "bwd_min", "bwd_mean", "bwd_std",
        "flow_bps", "flow_pps",
        "flow_iat_mean", "flow_iat_std", "flow_iat_max", "flow_iat_min",
        "fwd_iat_total", "fwd_iat_mean", "fwd_iat_std", "fwd_iat_max", "fwd_iat_min",
        "bwd_iat_total", "bwd_iat_mean", "bwd_iat_std", "bwd_iat_max", "bwd_iat_min",
        "fwd_hdr", "bwd_hdr", "fwd_pps", "bwd_pps",
        "pkt_min", "pkt_max", "pkt_mean", "pkt_std", "pkt_var",
        "fin_cnt", "psh_cnt", "ack_cnt", "avg_pkt_size",
        "subfwd_bytes", "fwd_win", "bwd_win", "act_data_fwd", "min_seg_fwd",
        "active_mean", "active_max", "active_min",
        "idle_mean", "idle_max", "idle_min",
    ]

    # Compute weighted average (weight = packet count per flow)
    total_weight = sum(f["n_total"] for f in flow_features)
    if total_weight < 1:
        total_weight = len(flow_features)  # fallback: equal weight

    avg = {}
    for key in numeric_keys:
        weighted_sum = sum(f.get(key, 0.0) * f["n_total"] for f in flow_features)
        avg[key] = weighted_sum / total_weight

    # For Destination Port: use the most common port across flows (not average)
    port_counter = Counter(f["dest_port"] for f in flow_features)
    most_common_port = port_counter.most_common(1)[0][0]
    avg["dest_port"] = most_common_port

    # For Init_Win_bytes: use the most common non-zero value
    fwd_wins = [f["fwd_win"] for f in flow_features if f["fwd_win"] > 0]
    bwd_wins = [f["bwd_win"] for f in flow_features if f["bwd_win"] > 0]
    avg["fwd_win"] = Counter(fwd_wins).most_common(1)[0][0] if fwd_wins else 0
    avg["bwd_win"] = Counter(bwd_wins).most_common(1)[0][0] if bwd_wins else 0

    # ── Raw structural telemetry — NO attack classification labels ──────
    # These are raw network metrics. The ML model on the Windows side
    # makes all classification decisions. We do NOT label anything as
    # "scan", "ddos", or "brute" here.
    unique_dst_ports = len(all_dst_ports)
    total_syn_packets = total_syn
    ssh_push_flows = sum(1 for f in flow_features if f["dest_port"] == 22 and f["psh_cnt"] > 0)
    unique_src  = len(all_src_ips)
    top_ip = (Counter(f["src"] for f in flow_features
                      if f["src"] != LOCAL_IP).most_common(1)[0][0]
              if all_src_ips else "")

    net = psutil.net_io_counters()
    cpu = psutil.cpu_percent(interval=None)
    mem = psutil.virtual_memory().percent

    S = avg
    feats = {
        "Destination Port": S["dest_port"], "Flow Duration": S["dur_us"],
        "Total Fwd Packets": S["n_fwd"],
        "Total Length of Fwd Packets": S["fwd_bytes"],
        "Fwd Packet Length Max": S["fwd_max"], "Fwd Packet Length Min": S["fwd_min"],
        "Fwd Packet Length Mean": S["fwd_mean"], "Fwd Packet Length Std": S["fwd_std"],
        "Bwd Packet Length Max": S["bwd_max"], "Bwd Packet Length Min": S["bwd_min"],
        "Bwd Packet Length Mean": S["bwd_mean"], "Bwd Packet Length Std": S["bwd_std"],
        "Flow Bytes/s": S["flow_bps"], "Flow Packets/s": S["flow_pps"],
        "Flow IAT Mean": S["flow_iat_mean"], "Flow IAT Std": S["flow_iat_std"],
        "Flow IAT Max": S["flow_iat_max"], "Flow IAT Min": S["flow_iat_min"],
        "Fwd IAT Total": S["fwd_iat_total"], "Fwd IAT Mean": S["fwd_iat_mean"],
        "Fwd IAT Std": S["fwd_iat_std"], "Fwd IAT Max": S["fwd_iat_max"],
        "Fwd IAT Min": S["fwd_iat_min"],
        "Bwd IAT Total": S["bwd_iat_total"], "Bwd IAT Mean": S["bwd_iat_mean"],
        "Bwd IAT Std": S["bwd_iat_std"], "Bwd IAT Max": S["bwd_iat_max"],
        "Bwd IAT Min": S["bwd_iat_min"],
        "Fwd Header Length": S["fwd_hdr"], "Bwd Header Length": S["bwd_hdr"],
        "Fwd Packets/s": S["fwd_pps"], "Bwd Packets/s": S["bwd_pps"],
        "Min Packet Length": S["pkt_min"], "Max Packet Length": S["pkt_max"],
        "Packet Length Mean": S["pkt_mean"], "Packet Length Std": S["pkt_std"],
        "Packet Length Variance": S["pkt_var"],
        "FIN Flag Count": S["fin_cnt"], "PSH Flag Count": S["psh_cnt"],
        "ACK Flag Count": S["ack_cnt"],
        "Average Packet Size": S["avg_pkt_size"],
        "Subflow Fwd Bytes": S["subfwd_bytes"],
        "Init_Win_bytes_forward": S["fwd_win"],
        "Init_Win_bytes_backward": S["bwd_win"],
        "act_data_pkt_fwd": S["act_data_fwd"],
        "min_seg_size_forward": S["min_seg_fwd"],
        "Active Mean": S["active_mean"], "Active Max": S["active_max"],
        "Active Min": S["active_min"],
        "Idle Mean": S["idle_mean"], "Idle Max": S["idle_max"],
        "Idle Min": S["idle_min"],
    }

    # Sanitize all features
    for k, v in feats.items():
        if isinstance(v, (int, float)):
            feats[k] = _sanitize(v)

    # Raw system telemetry — classification is done by the ML model
    feats.update({
        "unique_attackers": unique_src, "top_attacker": top_ip,
        "cpu_percent": cpu, "mem_percent": mem,
        "bytes_recv": net.bytes_recv, "bytes_sent": net.bytes_sent,
    })

    nonzero = sum(1 for k, v in feats.items()
                  if not k.startswith("_") and isinstance(v, (int, float)) and v != 0.0)
    feats["_nonzero"] = nonzero
    feats["_flows"]   = n_flows
    feats["_packets"] = total_pkts
    return feats


def _zero_state():
    """Return zero-valued features when no traffic is captured."""
    try:
        net = psutil.net_io_counters()
        cpu = psutil.cpu_percent(interval=None)
        mem = psutil.virtual_memory().percent
    except Exception:
        class _N:
            bytes_recv = 0; bytes_sent = 0
        net = _N(); cpu = 0.0; mem = 0.0
    return {
        "Destination Port":0,"Flow Duration":0,"Total Fwd Packets":0,
        "Total Length of Fwd Packets":0,"Fwd Packet Length Max":0,
        "Fwd Packet Length Min":0,"Fwd Packet Length Mean":0,
        "Fwd Packet Length Std":0,"Bwd Packet Length Max":0,
        "Bwd Packet Length Min":0,"Bwd Packet Length Mean":0,
        "Bwd Packet Length Std":0,"Flow Bytes/s":0,"Flow Packets/s":0,
        "Flow IAT Mean":0,"Flow IAT Std":0,"Flow IAT Max":0,"Flow IAT Min":0,
        "Fwd IAT Total":0,"Fwd IAT Mean":0,"Fwd IAT Std":0,"Fwd IAT Max":0,
        "Fwd IAT Min":0,"Bwd IAT Total":0,"Bwd IAT Mean":0,"Bwd IAT Std":0,
        "Bwd IAT Max":0,"Bwd IAT Min":0,"Fwd Header Length":0,
        "Bwd Header Length":0,"Fwd Packets/s":0,"Bwd Packets/s":0,
        "Min Packet Length":0,"Max Packet Length":0,"Packet Length Mean":0,
        "Packet Length Std":0,"Packet Length Variance":0,"FIN Flag Count":0,
        "PSH Flag Count":0,"ACK Flag Count":0,"Average Packet Size":0,
        "Subflow Fwd Bytes":0,"Init_Win_bytes_forward":0,
        "Init_Win_bytes_backward":0,"act_data_pkt_fwd":0,
        "min_seg_size_forward":20,"Active Mean":0,"Active Max":0,
        "Active Min":0,"Idle Mean":0,"Idle Max":0,"Idle Min":0,
        "unique_attackers":0,"top_attacker":"",
        "cpu_percent":cpu,"mem_percent":mem,
        "bytes_recv":net.bytes_recv,"bytes_sent":net.bytes_sent,
        "_nonzero":0,"_flows":0,"_packets":0,
    }


# ── Background loops ─────────────────────────────────────────────
def _compute_loop():
    """Every 2 seconds: grab buffered packets -> features -> state."""
    print("[*] Feature computation loop started")
    sys.stdout.flush()
    cycle = 0
    while True:
        try:
            pkts = _get_buffered_packets()
            try:
                feats = _compute_cicids_features(pkts)
            except Exception as e:
                print(f"[feature error] {e}")
                traceback.print_exc()
                feats = _zero_state()

            with _lock:
                _state.clear()
                _state.update(feats)

            cycle += 1
            if cycle % 5 == 0 or feats.get('_packets', 0) > 50:
                print(f"[cycle #{cycle}] buf={len(pkts)} pkts={feats.get('_packets',0)} "
                      f"flows={feats.get('_flows',0)} nz={feats.get('_nonzero',0)}/52 "
                      f"dur={feats.get('Flow Duration',0):.0f}us "
                      f"bps={feats.get('Flow Bytes/s',0):.0f} "
                      f"win_fwd={feats.get('Init_Win_bytes_forward',0)} "
                      f"win_bwd={feats.get('Init_Win_bytes_backward',0)}")
                sys.stdout.flush()
        except Exception as e:
            print(f"[compute error] {e}")
            traceback.print_exc()
            sys.stdout.flush()
        time.sleep(2)


def _watchdog():
    """Restart tshark if it dies."""
    global _tshark_proc
    while True:
        time.sleep(15)  # Check every 15s (was 30s)
        if _tshark_proc is None or _tshark_proc.poll() is not None:
            print("[WATCHDOG] tshark died! Restarting...")
            sys.stdout.flush()
            _start_tshark()


# ── Flask routes ─────────────────────────────────────────────────
@app.route("/state")
def state():
    try:
        with _lock:
            data = dict(_state) if _state else _zero_state()
        resp = make_response(jsonify(data))
        resp.headers['Connection'] = 'keep-alive'
        resp.headers['Keep-Alive'] = 'timeout=60, max=1000'
        return resp
    except Exception as e:
        return jsonify(_zero_state())

@app.route("/health")
def health():
    try:
        with _lock:
            s = dict(_state) if _state else _zero_state()
        data = {
            "status": "ok",
            "nonzero": s.get("_nonzero", 0),
            "packets": s.get("_packets", 0),
            "flows":   s.get("_flows", 0),
            "bps_in":  s.get("Flow Bytes/s", 0),
            "pps":     s.get("Flow Packets/s", 0),
            "cpu":     s.get("cpu_percent", 0),
            "iface":   IFACE,
            "local_ip": LOCAL_IP,
            "timestamp": datetime.utcnow().isoformat(),
        }
        resp = make_response(jsonify(data))
        resp.headers['Connection'] = 'keep-alive'
        return resp
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)})

@app.route("/debug")
def debug():
    try:
        with _lock:
            s = dict(_state) if _state else _zero_state()
        return jsonify(s)
    except Exception as e:
        return jsonify({"error": str(e)})


# ── Main ─────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("=" * 60)
    print("  Ubuntu Metrics API")
    print(f"  Interface: {IFACE}")
    print(f"  Local IP:  {LOCAL_IP}")
    print(f"  Buffer:    {_BUFFER_WINDOW}s rolling window")
    print(f"  Python:    {sys.executable} ({sys.version.split()[0]})")
    print("=" * 60)
    sys.stdout.flush()

    # Initialize state with zeros
    with _lock:
        _state.update(_zero_state())

    # Start tshark capture
    print("[*] Starting tshark capture...")
    sys.stdout.flush()
    capture_ok = _start_tshark()

    if not capture_ok:
        print("[WARNING] tshark capture failed - will return zero features")
        print("[WARNING] Make sure tshark is installed: apt-get install tshark")
        sys.stdout.flush()

    # Start background threads
    print("[*] Starting background threads...")
    sys.stdout.flush()
    threading.Thread(target=_tshark_reader, daemon=True, name="TsharkReader").start()
    time.sleep(2)
    threading.Thread(target=_compute_loop, daemon=True, name="ComputeLoop").start()
    threading.Thread(target=_watchdog, daemon=True, name="Watchdog").start()

    # Start Flask (this blocks)
    print("[*] Starting Flask on 0.0.0.0:8080 ...")
    sys.stdout.flush()
    try:
        app.run(host="0.0.0.0", port=8080, threaded=True,
                debug=False, use_reloader=False)
    except Exception as e:
        print(f"[FATAL] Flask failed to start: {e}")
        traceback.print_exc()
        sys.stdout.flush()
        sys.exit(1)
