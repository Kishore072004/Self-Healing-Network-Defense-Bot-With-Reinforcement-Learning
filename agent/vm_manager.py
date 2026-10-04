"""
VM Manager — auto-starts Ubuntu and Kali VMs via vmrun.exe,
waits for SSH, deploys ubuntu_metrics_api.py, and sets up sudo.

If vmrun is not found or VMX paths are placeholders, it skips the
boot step and just waits for existing SSH connections.
"""
import logging
import os
import subprocess
import time
from concurrent.futures import ThreadPoolExecutor

import paramiko
from pathlib import Path

logger = logging.getLogger(__name__)

# Common Workstation installs (config vmrun_path is tried first)
_VMRUN_CANDIDATES = (
    r"C:\Program Files\VMware\VMware Workstation\vmrun.exe",
    r"C:\Program Files (x86)\VMware\VMware Workstation\vmrun.exe",
)


def _resolve_vmrun_path(configured: str) -> str:
    """Use configured path if it exists; otherwise try standard install locations."""
    configured = (configured or "").strip()
    if configured and Path(configured).exists():
        return configured
    for alt in _VMRUN_CANDIDATES:
        if Path(alt).exists():
            if configured:
                logger.warning(
                    "vmware.vmrun_path not found (%s) — using %s",
                    configured,
                    alt,
                )
            else:
                logger.info("vmware.vmrun_path unset — using %s", alt)
            return alt
    return configured


class VMManager:
    """Manages VMware VM lifecycle and SSH deployment."""

    def __init__(self, config: dict):
        self.config   = config
        vcfg          = config.get("vmware", {})
        net           = config["network"]
        ucfg          = config["ubuntu"]
        kcfg          = config["kali"]

        self.vmrun       = _resolve_vmrun_path(vcfg.get("vmrun_path", ""))
        self.ubuntu_vmx  = vcfg.get("ubuntu_vmx", "")
        self.kali_vmx    = vcfg.get("kali_vmx", "")
        self.boot_timeout = int(vcfg.get("boot_timeout", 120))
        self.shutdown_after = bool(vcfg.get("shutdown_after", False))
        self.ssh_port = int(vcfg.get("ssh_port", 22))
        self.ssh_connect_timeout = int(vcfg.get("ssh_connect_timeout", 20))
        self.ssh_banner_timeout = int(vcfg.get("ssh_banner_timeout", 30))
        self._run_vms = set(vcfg.get("run_vms") or ["ubuntu", "kali"])

        self.ubuntu_ip   = net["ubuntu_ip"]
        self.kali_ip     = net["kali_ip"]
        self.ubuntu_user = ucfg["user"]
        self.ubuntu_pass = ucfg["password"]
        self.kali_user   = kcfg["user"]
        self.kali_pass   = kcfg["password"]
        self.ubuntu_ssh_port = int(ucfg.get("ssh_port", self.ssh_port))
        self.kali_ssh_port = int(kcfg.get("ssh_port", self.ssh_port))

    # ─────────────────────────────────────────────────────────────────────
    #  vmrun helpers
    # ─────────────────────────────────────────────────────────────────────
    def _vmrun_available(self) -> bool:
        return bool(self.vmrun) and Path(self.vmrun).exists()

    def _vmx_valid(self, vmx: str) -> bool:
        """Accept if path looks real (not placeholder text)."""
        if not vmx:
            return False
        if "YourName" in vmx or "placeholder" in vmx.lower():
            return False
        # It may not exist yet if VM is on another drive — try anyway
        return True

    def _start_vm(self, vmx: str, label: str):
        key = label.lower()
        if key not in self._run_vms:
            logger.info(f"Skipping vmrun start for {label} (not in vmware.run_vms).")
            return
        if not self._vmrun_available():
            logger.warning(f"vmrun not found — skipping auto-start for {label}. "
                           "Make sure the VM is already running.")
            return
        if not self._vmx_valid(vmx):
            logger.warning(f"VMX path for {label} looks like a placeholder. "
                           "Update 'vmware.{label}_vmx' in config.yaml.")
            return

        # Check if already running
        try:
            result = subprocess.run(
                [self.vmrun, "list"],
                capture_output=True, text=True, timeout=10
            )
            if vmx in result.stdout:
                logger.info(f"{label} VM is already running.")
                return
        except Exception:
            pass

        logger.info(f"Starting {label} VM: {vmx}")
        try:
            subprocess.Popen(
                [self.vmrun, "-T", "ws", "start", vmx, "nogui"],
                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
            )
            logger.info(f"{label} VM start command issued (nogui).")
        except Exception as e:
            logger.error(f"Failed to start {label} VM: {e}")

    # ─────────────────────────────────────────────────────────────────────
    #  SSH wait
    # ─────────────────────────────────────────────────────────────────────
    def _wait_for_ssh(
        self,
        ip: str,
        user: str,
        password: str,
        label: str,
        timeout: int,
        port: int,
    ) -> bool:
        logger.info(
            f"Waiting for {label} SSH ({ip}:{port}) — up to {timeout}s …"
        )
        # Short probes while VMs boot; full ssh_*_timeout is for deploy/sudo after SSH is up.
        poll_connect = min(10, max(5, self.ssh_connect_timeout))
        poll_banner = min(18, max(8, self.ssh_banner_timeout))
        poll_interval = 5
        deadline = time.time() + timeout
        attempt = 0
        last_err = None
        while time.time() < deadline:
            attempt += 1
            try:
                c = paramiko.SSHClient()
                c.set_missing_host_key_policy(paramiko.AutoAddPolicy())
                # Windows: avoid slow/hanging agent + default key scan (often breaks "worked before")
                c.connect(
                    ip,
                    port=port,
                    username=user,
                    password=password,
                    timeout=poll_connect,
                    banner_timeout=poll_banner,
                    auth_timeout=poll_connect,
                    allow_agent=False,
                    look_for_keys=False,
                )
                c.close()
                logger.info(f"{label} SSH ready ✓ (attempt {attempt})")
                return True
            except Exception as e:
                last_err = e
                remaining = max(0, int(deadline - time.time()))
                err_msg = f"{type(e).__name__}: {e}"
                if attempt <= 3 or attempt % 5 == 0:
                    logger.warning(
                        f"  {label} SSH not ready ({remaining}s left) — {err_msg}"
                    )
                else:
                    logger.info(f"  {label} not ready yet … {remaining}s left")
                time.sleep(poll_interval)
        logger.error(
            f"{label} SSH not reachable after {timeout}s (last error: {last_err}). "
            f"Verify IP {ip}, port {port}, user/password, and that sshd is running."
        )
        return False

    # ─────────────────────────────────────────────────────────────────────
    #  Ubuntu setup
    # ─────────────────────────────────────────────────────────────────────
    def _deploy_metrics_api(self):
        """Copy ubuntu_metrics_api.py to Ubuntu and start it.
        Uses tcpdump for stable packet capture."""
        local_script = Path("ubuntu_metrics_api.py")
        if not local_script.exists():
            logger.error("ubuntu_metrics_api.py not found locally — skipping deploy.")
            return

        try:
            c = paramiko.SSHClient()
            c.set_missing_host_key_policy(paramiko.AutoAddPolicy())
            c.connect(
                self.ubuntu_ip,
                port=self.ubuntu_ssh_port,
                username=self.ubuntu_user,
                password=self.ubuntu_pass,
                timeout=self.ssh_connect_timeout,
                banner_timeout=self.ssh_banner_timeout,
                auth_timeout=self.ssh_connect_timeout,
                allow_agent=False,
                look_for_keys=False,
            )

            # 1. Kill old API instances
            c.exec_command("pkill -9 -f ubuntu_metrics_api 2>/dev/null || true")
            time.sleep(2)

            # 3. Upload API script
            sftp = c.open_sftp()
            remote_path = f"/home/{self.ubuntu_user}/ubuntu_metrics_api.py"
            sftp.put(str(local_script), remote_path)

            # 4. Upload install helper (avoids shell quote-escaping issues)
            #    Handles Ubuntu 22.04+ PEP 668 "externally-managed-environment"
            install_code = "\n".join([
                "import subprocess, sys",
                "# Skip if already installed",
                "try:",
                "    import flask, psutil; print('ALREADY_INSTALLED'); sys.exit(0)",
                "except ImportError:",
                "    pass",
                "cmds = [",
                "    [sys.executable,'-m','pip','install','flask','psutil','--quiet','--break-system-packages'],",
                "    ['pip3','install','flask','psutil','--quiet','--break-system-packages'],",
                "    ['sudo','apt-get','install','-y','-q','python3-flask','python3-psutil'],",
                "    [sys.executable,'-m','pip','install','flask','psutil','--user','--quiet'],",
                "]",
                "for cmd in cmds:",
                "    try:",
                "        r=subprocess.run(cmd,capture_output=True,timeout=90)",
                "        print(('OK' if r.returncode==0 else 'FAIL')+': '+' '.join(cmd[:5]))",
                "        if r.returncode==0: sys.exit(0)",
                "    except Exception as e:",
                "        print('ERR:',e)",
                "print('ALL_FAILED: flask may not be available')",
            ])
            with sftp.open("/tmp/_install_deps.py", "w") as fh:
                fh.write(install_code + "\n")
            sftp.close()
            logger.info(f"Uploaded metrics API → {remote_path}")

            # 5. Run install helper
            logger.info("Installing flask + psutil on Ubuntu (Ubuntu 22.04 compatible) ...")
            _, out_ch, _ = c.exec_command(
                "python3 /tmp/_install_deps.py; rm -f /tmp/_install_deps.py",
                timeout=120,
            )
            pip_out = out_ch.read().decode(errors="replace").strip()
            out_ch.channel.recv_exit_status()
            if pip_out:
                logger.info(f"pip output: {pip_out[:400]}")

            # 6. Detect Python binary
            _, py_out, _ = c.exec_command("which python3 || which python")
            py_bin = py_out.read().decode().strip() or "python3"
            logger.info(f"Using Python binary: {py_bin}")

            # 7. Launch API with keepalive wrapper (runs as root for tcpdump)
            # Create a wrapper script that restarts the API if it dies
            wrapper_script = f"""#!/bin/bash
while true; do
    echo "[$(date)] Starting metrics API..." >> /tmp/metrics_api.log
    sudo {py_bin} {remote_path} >> /tmp/metrics_api.log 2>&1
    echo "[$(date)] API crashed! Restarting in 3s..." >> /tmp/metrics_api.log
    sleep 3
done
"""
            # Upload wrapper
            sftp = c.open_sftp()
            wrapper_path = f"/home/{self.ubuntu_user}/api_keepalive.sh"
            with sftp.open(wrapper_path, 'w') as f:
                f.write(wrapper_script)
            sftp.close()
            c.exec_command(f"chmod +x {wrapper_path}")
            
            # Launch with wrapper
            launch_cmd = f"nohup bash {wrapper_path} > /dev/null 2>&1 &"
            c.exec_command(launch_cmd)

            # 7. Wait for Flask + tshark to initialize
            logger.info("Waiting 15s for metrics API + tshark to initialize ...")
            time.sleep(15)

            # 8. Check process is alive; dump log if not
            _, ps_out, _ = c.exec_command("pgrep -f ubuntu_metrics_api | head -3")
            pids = ps_out.read().decode().strip()
            if pids:
                logger.info(f"Metrics API process(es) running: PID={pids.replace(chr(10), ' ')}")
            else:
                _, log_out, _ = c.exec_command("tail -30 /tmp/metrics_api.log 2>/dev/null")
                log_txt = log_out.read().decode(errors="replace").strip()
                logger.error(f"Metrics API process NOT found!\nLog:\n{log_txt}")

            c.close()

            # 9. HTTP health checks
            import requests
            api_url = f"http://{self.ubuntu_ip}:8080/health"
            for attempt in range(20):
                try:
                    r  = requests.get(api_url, timeout=12)
                    h  = r.json()
                    nz = h.get("nonzero", 0)
                    pk = h.get("packets", 0)
                    cpu= h.get("cpu",     0)
                    iface = h.get("iface", "?")
                    lip = h.get("local_ip", "?")
                    logger.info(
                        f"API health #{attempt+1}: "
                        f"iface={iface} ip={lip} packets={pk} nonzero={nz} cpu={cpu:.1f}"
                    )
                    if cpu > 0 or nz >= 3:
                        logger.info("Ubuntu metrics API verified [OK]")
                        return
                except Exception as e:
                    logger.warning(f"API health #{attempt+1}: {e}")

                    # On 3rd failure, SSH in and dump the remote log
                    if attempt == 2:
                        try:
                            c2 = paramiko.SSHClient()
                            c2.set_missing_host_key_policy(paramiko.AutoAddPolicy())
                            c2.connect(self.ubuntu_ip, port=self.ubuntu_ssh_port, username=self.ubuntu_user, password=self.ubuntu_pass, timeout=10)
                            _, log_out, _ = c2.exec_command("tail -40 /tmp/metrics_api.log 2>/dev/null")
                            log_txt = log_out.read().decode(errors="replace").strip()
                            _, port_out, _ = c2.exec_command("ss -tlnp | grep 8080 || echo 'PORT_8080_NOT_LISTENING'")
                            port_txt = port_out.read().decode(errors="replace").strip()
                            _, tshark_out, _ = c2.exec_command("which tshark && tshark --version 2>&1 | head -2 || echo 'TSHARK_NOT_FOUND'")
                            tshark_txt = tshark_out.read().decode(errors="replace").strip()
                            c2.close()
                            logger.error(
                                f"=== REMOTE DEBUG (attempt {attempt+1}) ===\n"
                                f"--- Port 8080 ---\n{port_txt}\n"
                                f"--- tshark ---\n{tshark_txt}\n"
                                f"--- /tmp/metrics_api.log (last 40 lines) ---\n{log_txt}"
                            )
                        except Exception as diag_err:
                            logger.error(f"Could not fetch remote diagnostics: {diag_err}")

                time.sleep(5)
            # Final: dump remote log one more time
            try:
                c3 = paramiko.SSHClient()
                c3.set_missing_host_key_policy(paramiko.AutoAddPolicy())
                c3.connect(self.ubuntu_ip, port=self.ubuntu_ssh_port, username=self.ubuntu_user, password=self.ubuntu_pass, timeout=10)
                _, log_out, _ = c3.exec_command("tail -50 /tmp/metrics_api.log 2>/dev/null")
                log_txt = log_out.read().decode(errors="replace").strip()
                c3.close()
                logger.warning(
                    f"API health check timed out after 20 attempts.\n"
                    f"--- /tmp/metrics_api.log ---\n{log_txt}"
                )
            except Exception:
                logger.warning(
                    "API health check timed out. Could not fetch remote logs.\n"
                    "  Manual check: ssh kishore@192.168.100.10 'tail -50 /tmp/metrics_api.log'"
                )
        except Exception as e:
            logger.error(f"deploy_metrics_api failed: {e}")


    def _setup_sudo_iptables(self):
        """Configure passwordless sudo for iptables and tshark on Ubuntu."""
        sudoers_line = (
            f"{self.ubuntu_user} ALL=(ALL) NOPASSWD: ALL"
        )
        sudoers_file = f"/etc/sudoers.d/rl_defense"
        cmd = (
            f"echo '{self.ubuntu_pass}' | sudo -S bash -c "
            f"\"echo '{sudoers_line}' | sudo tee {sudoers_file} > /dev/null "
            f"&& sudo chmod 440 {sudoers_file}\""
        )
        try:
            c = paramiko.SSHClient()
            c.set_missing_host_key_policy(paramiko.AutoAddPolicy())
            c.connect(
                self.ubuntu_ip,
                port=self.ubuntu_ssh_port,
                username=self.ubuntu_user,
                password=self.ubuntu_pass,
                timeout=self.ssh_connect_timeout,
                banner_timeout=self.ssh_banner_timeout,
                auth_timeout=self.ssh_connect_timeout,
                allow_agent=False,
                look_for_keys=False,
            )
            full_cmd = (
                f"echo {self.ubuntu_pass} | sudo -S bash -c "
                f"\"printf '{sudoers_line}\\n' > {sudoers_file} && "
                f"chmod 440 {sudoers_file}\""
            )
            _, stdout, stderr = c.exec_command(full_cmd, timeout=15)
            stdout.channel.recv_exit_status()

            # Ensure API and SSH survive SYN floods by bypassing nf_conntrack limits
            logger.info("Applying NOTRACK rules to immunize ports 22 and 8080 from DDoS exhaustion...")
            notrack_cmds = [
                "sudo iptables -t raw -F PREROUTING 2>/dev/null || true",
                "sudo iptables -t raw -F OUTPUT 2>/dev/null || true",
                "sudo iptables -t raw -A PREROUTING -p tcp -m multiport --dports 22,8080 -j NOTRACK",
                "sudo iptables -t raw -A OUTPUT -p tcp -m multiport --sports 22,8080 -j NOTRACK"
            ]
            for n_cmd in notrack_cmds:
                c.exec_command(f"echo {self.ubuntu_pass} | sudo -S {n_cmd}")

            c.close()
            logger.info("Passwordless sudo for iptables + tshark configured on Ubuntu.")
        except Exception as e:
            logger.warning(f"setup_sudo_iptables: {e} — "
                           "iptables/tshark will use password-piped sudo instead.")

    # ─────────────────────────────────────────────────────────────────────
    #  Public API
    # ─────────────────────────────────────────────────────────────────────
    def startup(self) -> bool:
        """
        Full boot sequence:
        1. Start Ubuntu VM via vmrun (if configured)
        2. Start Kali VM via vmrun (if configured)
        3. Wait for both SSH connections
        4. Deploy metrics API to Ubuntu
        5. Return True if both VMs are SSH-ready
        """
        logger.info("=" * 50)
        logger.info("  VM STARTUP SEQUENCE")
        logger.info("=" * 50)

        # Boot VMs
        self._start_vm(self.ubuntu_vmx, "Ubuntu")
        self._start_vm(self.kali_vmx,   "Kali")

        # Wait for both SSH endpoints in parallel (sequential wait doubled typical boot time)
        with ThreadPoolExecutor(max_workers=2) as pool:
            fut_u = pool.submit(
                self._wait_for_ssh,
                self.ubuntu_ip,
                self.ubuntu_user,
                self.ubuntu_pass,
                "Ubuntu",
                self.boot_timeout,
                self.ubuntu_ssh_port,
            )
            fut_k = pool.submit(
                self._wait_for_ssh,
                self.kali_ip,
                self.kali_user,
                self.kali_pass,
                "Kali",
                self.boot_timeout,
                self.kali_ssh_port,
            )
            ubuntu_ok = fut_u.result()
            kali_ok = fut_k.result()

        if not ubuntu_ok:
            if kali_ok:
                logger.error(
                    "Kali SSH works but Ubuntu does not (TCP timeout/refused). "
                    "Typical causes: Ubuntu VM is still powered off (fix vmware.vmrun_path "
                    "or start the VM manually), network.ubuntu_ip does not match "
                    "`ip -br a` on Ubuntu, or SSH is blocked (ufw / sshd). "
                    "From PowerShell: Test-NetConnection -ComputerName %s -Port %s",
                    self.ubuntu_ip,
                    self.ubuntu_ssh_port,
                )
            logger.error("Ubuntu VM not reachable — cannot run live mode.")
            return False
        if not kali_ok:
            logger.error("Kali VM not reachable — cannot run live mode.")
            return False

        # Deploy Ubuntu metrics API
        self._setup_sudo_iptables()
        self._deploy_metrics_api()

        logger.info("Both VMs ready ✓")
        return True

    def shutdown(self):
        """Optionally power off VMs after training."""
        if not self.shutdown_after:
            logger.info("shutdown_after=false — leaving VMs running.")
            return
        if not self._vmrun_available():
            return
        for vmx, label in [(self.ubuntu_vmx, "Ubuntu"), (self.kali_vmx, "Kali")]:
            if self._vmx_valid(vmx):
                try:
                    subprocess.run(
                        [self.vmrun, "-T", "ws", "stop", vmx, "soft"],
                        timeout=30, capture_output=True
                    )
                    logger.info(f"{label} VM powered off.")
                except Exception as e:
                    logger.warning(f"Could not stop {label} VM: {e}")
