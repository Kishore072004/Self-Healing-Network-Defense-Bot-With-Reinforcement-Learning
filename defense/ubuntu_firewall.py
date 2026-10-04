"""
Ubuntu victim firewall (iptables) over SSH only.
"""
import logging
import paramiko

logger = logging.getLogger(__name__)


class UbuntuFirewallDefender:
    CHAIN = "RL_DEFENSE"

    def __init__(self, config: dict):
        net  = config["network"]
        ucfg = config["ubuntu"]
        vcfg = config.get("vmware", {})

        self.host     = net["ubuntu_ip"]
        self.user     = ucfg["user"]
        self.password = ucfg["password"]
        self.simulate = bool(ucfg.get("firewall_simulate", False))

        base_port = int(vcfg.get("ssh_port", 22))
        self._ssh_port = int(ucfg["ssh_port"]) if "ssh_port" in ucfg else base_port
        self._ssh_timeout = int(vcfg.get("ssh_connect_timeout", 20))
        self._ssh_banner = int(vcfg.get("ssh_banner_timeout", 30))

        self.blocked_ips  = set()
        self.rate_limited = set()
        self._chain_ready = False

    # ─────────────────────────────────────────────
    # SSH helpers
    # ─────────────────────────────────────────────
    def _ssh(self):
        if hasattr(self, "_ssh_client") and self._ssh_client is not None:
            try:
                transport = self._ssh_client.get_transport()
                if transport and transport.is_active():
                    return self._ssh_client
            except Exception:
                pass

        c = paramiko.SSHClient()
        c.set_missing_host_key_policy(paramiko.AutoAddPolicy())
        c.connect(
            self.host,
            port=self._ssh_port,
            username=self.user,
            password=self.password,
            timeout=self._ssh_timeout,
            banner_timeout=self._ssh_banner,
            auth_timeout=self._ssh_timeout,
            allow_agent=False,
            look_for_keys=False,
        )
        self._ssh_client = c
        return c

    def _run(self, ssh, cmd, retry=True):
        if cmd.strip().startswith("sudo"):
            full = f"echo {self.password} | sudo -S {cmd.lstrip('sudo').lstrip()}"
        else:
            full = cmd

        try:
            _, stdout, stderr = ssh.exec_command(full, timeout=15)
            rc  = stdout.channel.recv_exit_status()
            out = stdout.read().decode(errors="replace").strip()
            err = stderr.read().decode(errors="replace").strip()
            return rc, out, err
        except Exception as e:
            if retry:
                self._ssh_client = None
                return self._run(self._ssh(), cmd, retry=False)
            raise e

    # ─────────────────────────────────────────────
    # Chain setup
    # ─────────────────────────────────────────────
    def _ensure_chain(self, ssh):
        if self._chain_ready:
            return

        self._run(ssh, f"sudo iptables -N {self.CHAIN} 2>/dev/null || true")
        self._run(ssh, f"sudo iptables -C INPUT -j {self.CHAIN} 2>/dev/null "
                       f"|| sudo iptables -I INPUT 1 -j {self.CHAIN}")

        self._chain_ready = True

    # ─────────────────────────────────────────────
    # Firewall Actions
    # ─────────────────────────────────────────────
    def block_ip(self, ip):
        # Whitelist critical infrastructure IPs to prevent accidental lockout
        if ip in ("127.0.0.1", "192.168.100.1", "192.168.100.2", self.host):
            logger.warning(f"Prevented block of critical IP: {ip}")
            return False

        if not ip or ip in self.blocked_ips:
            return False

        try:
            if self.simulate:
                self.blocked_ips.add(ip)
                return True

            ssh = self._ssh()
            self._ensure_chain(ssh)

            rc, _, _ = self._run(ssh, f"sudo iptables -I {self.CHAIN} 1 -s {ip} -j DROP")

            if rc == 0:
                self.blocked_ips.add(ip)
                return True

        except Exception as e:
            logger.error(e)

        return False

    def unblock_ip(self, ip):
        if ip not in self.blocked_ips:
            return False

        try:
            if self.simulate:
                self.blocked_ips.discard(ip)
                return True

            ssh = self._ssh()
            rc, _, _ = self._run(ssh, f"sudo iptables -D {self.CHAIN} -s {ip} -j DROP")

            if rc == 0:
                self.blocked_ips.discard(ip)
                return True

        except Exception as e:
            logger.error(e)

        return False

    def get_blocked(self):
        return list(self.blocked_ips)

    def rate_limit_subnet(self, subnet):
        # Prevent rate-limiting gateway
        if any(subnet.startswith(p) for p in ("127.0.0.1", "192.168.100.1", "192.168.100.2")):
            return False

        if not subnet or subnet in self.rate_limited:
            return False

        try:
            if self.simulate:
                self.rate_limited.add(subnet)
                return True

            ssh = self._ssh()
            self._ensure_chain(ssh)

            rc, _, _ = self._run(ssh, f"sudo iptables -I {self.CHAIN} 1 -s {subnet} -j DROP")

            if rc == 0:
                self.rate_limited.add(subnet)
                return True

        except Exception as e:
            logger.error(e)

        return False

    # ─────────────────────────────────────────────
    def restart_networking(self):
        """Flush defense rules only — does NOT kill SSH or API connections.
        Only clears the RL_DEFENSE chain and resets blocked IP tracking."""
        try:
            if self.simulate:
                logger.info("[SIM] Restart networking")
                return True

            ssh = self._ssh()
            # Only flush our defense chain (not all iptables or conntrack)
            self._run(ssh, f"sudo iptables -F {self.CHAIN} 2>/dev/null || true")

            self.blocked_ips.clear()
            self.rate_limited.clear()
            logger.info("[FIREWALL] Defense rules flushed (safe restart)")
            return True

        except Exception as e:
            logger.error(f"restart_networking error: {e}")

        return False

    # ─────────────────────────────────────────────
    def cleanup_all_rules(self):
        try:
            if not self.simulate:
                ssh = self._ssh()
                self._run(ssh, f"sudo iptables -F {self.CHAIN} 2>/dev/null || true")

            self.blocked_ips.clear()
            self.rate_limited.clear()
            self._chain_ready = False

            return True

        except Exception as e:
            logger.error(e)

        return False