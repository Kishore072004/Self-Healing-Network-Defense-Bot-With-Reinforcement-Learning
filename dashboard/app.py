import logging
import threading
import time
from flask import Flask, render_template, jsonify
from flask_socketio import SocketIO, emit
from flask_cors import CORS

logger   = logging.getLogger(__name__)
app      = Flask(__name__)
CORS(app)
socketio = SocketIO(app, cors_allowed_origins="*", async_mode="threading")

_lock    = threading.Lock()
_history = []
_fw_log  = []          # firewall action log (last 50)
_episode_rewards = []  # cumulative reward per episode for learning curve
_class_counts = {"BENIGN": 0, "Bot": 0, "DDoS": 0, "PortScan": 0, "Brute": 0}
_last_episode = 0

_state = {
    "mode": "Initializing...",
    "episode": 0, "step": 0, "total_reward": 0.0, "step_reward": 0.0,
    "action": 0, "action_name": "No-op",
    "threat_level": 0.0, "threat_class": "BENIGN",
    "confidence": 0.0,
    "cpu": 0.0, "memory": 0.0,
    "is_attacking": False, "reward_history": [],
    "top_attacker_ip": "", "blocked_ips": [], "fw_result": "",
    "fw_log": [],
    # ML probability vector
    "p_benign": 1.0, "p_bot": 0.0, "p_ddos": 0.0,
    "p_scan": 0.0, "p_brute": 0.0,
    # IDS monitor fields
    "ids_final_class": "Normal Traffic",
    "ids_lr_class": "Normal Traffic",
    "ids_lgb_class": "Normal Traffic",
    "ids_models_agree": True,
    "ids_flow_count": 0,
    "ids_packet_count": 0,
    "ids_nonzero_features": 0,
    # Learning curve
    "episode_rewards": [],
    "class_counts": {"BENIGN": 0, "Bot": 0, "DDoS": 0, "PortScan": 0, "Brute": 0},
    "launch_mode": "AUTO",
}


def update(data: dict):
    global _last_episode
    with _lock:
        _state.update(data)

        # Reward history (rolling window for chart)
        _history.append(data.get("step_reward", 0.0))
        if len(_history) > 300:
            _history.pop(0)
        _state["reward_history"] = list(_history[-60:])

        # Track episode-level rewards for learning curve
        ep = data.get("episode", 0)
        if ep > _last_episode and _last_episode > 0:
            _episode_rewards.append(data.get("total_reward", 0.0))
            if len(_episode_rewards) > 200:
                _episode_rewards.pop(0)
        _last_episode = ep
        _state["episode_rewards"] = list(_episode_rewards)

        # Track class distribution
        tc = data.get("threat_class", "BENIGN")
        if tc in _class_counts:
            _class_counts[tc] += 1
        _state["class_counts"] = dict(_class_counts)

        # Firewall action log
        fw_result = data.get("fw_result", "")
        if fw_result and fw_result not in ("simulated", "no-op", ""):
            entry = {
                "time":   time.strftime("%H:%M:%S"),
                "ep":     data.get("episode", 0),
                "action": data.get("action_name", ""),
                "result": fw_result,
                "ip":     data.get("top_attacker_ip", ""),
                "class":  data.get("threat_class", ""),
            }
            _fw_log.insert(0, entry)
            if len(_fw_log) > 50:
                _fw_log.pop()
        _state["fw_log"] = list(_fw_log[:20])


def _push_loop():
    while True:
        with _lock:
            payload = dict(_state)
        socketio.emit("state_update", payload)
        time.sleep(1)


@app.route("/")
def index():
    return render_template("soc_dashboard.html")


@app.route("/api/state")
def api_state():
    with _lock:
        return jsonify(dict(_state))


@app.route("/api/history")
def api_history():
    with _lock:
        return jsonify(_history[-200:])


@app.route("/api/blocked")
def api_blocked():
    with _lock:
        return jsonify({
            "blocked_ips": _state.get("blocked_ips", []),
            "fw_log":      _fw_log[:20],
        })


@app.route("/api/learning_curve")
def api_learning_curve():
    with _lock:
        return jsonify(_episode_rewards[-200:])


@app.route("/api/class_dist")
def api_class_dist():
    with _lock:
        return jsonify(dict(_class_counts))


@app.route("/api/reset", methods=["POST"])
def api_reset():
    update({"episode": 0, "step": 0, "total_reward": 0.0})
    return jsonify({"status": "reset"})


@app.route("/api/set_launch_mode", methods=["POST"])
def api_set_launch_mode():
    from flask import request
    data = request.json or {}
    mode = data.get("mode", "AUTO")
    with _lock:
        _state["launch_mode"] = mode
    logger.info(f"Launch mode changed to: {mode}")
    return jsonify({"status": "ok", "launch_mode": mode})


def get_launch_mode():
    with _lock:
        return _state.get("launch_mode", "AUTO")


@socketio.on("connect")
def on_connect():
    with _lock:
        emit("state_update", dict(_state))


def run_dashboard(host: str = "0.0.0.0", port: int = 5000):
    threading.Thread(target=_push_loop, daemon=True).start()
    socketio.run(
        app,
        host=host,
        port=port,
        use_reloader=False,
        allow_unsafe_werkzeug=True,
    )