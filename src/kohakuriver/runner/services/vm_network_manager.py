"""
VM Network Manager for KohakuRiver.

Provides network setup for QEMU VMs by creating TAP devices
and attaching them to the appropriate bridge based on the networking mode.

Two modes:
- Overlay mode (OVERLAY_ENABLED=True): TAP attaches to kohaku-overlay bridge.
  VMs share the same bridge as Docker containers, get IPs from overlay pool.
- Standard mode (OVERLAY_ENABLED=False): TAP attaches to kohaku-br0 NAT bridge.
  VMs get IPs from a local 10.200.0.0/24 pool with NAT for internet access.
"""

from __future__ import annotations

import asyncio
import hashlib
import ipaddress
import socket
import subprocess
from dataclasses import dataclass

import httpx

from kohakuriver.models.overlay_subnet import OverlaySubnetConfig
from kohakuriver.runner.config import config
from kohakuriver.runner.services.overlay_manager import RunnerOverlayManager
from kohakuriver.utils.logger import get_logger

logger = get_logger(__name__)


def _tap_name(task_id: int, suffix: str = "") -> str:
    """Generate a short TAP device name (max 15 chars for Linux IFNAMSIZ).

    Multi-NIC: suffix differentiates interfaces (e.g., empty for primary,
    '-1', '-2' for additional). Total length stays <= 15 chars.
    """
    h = hashlib.sha3_224(str(task_id).encode()).hexdigest()[:7]
    return f"tap-{h}{suffix}"


def _generate_mac(task_id: int, network_index: int = 0) -> str:
    """Generate a deterministic MAC address from task_id and NIC index.

    Uses QEMU's locally-administered range (52:54:00:xx:xx:xx).
    Different NICs on the same VM get different MACs by hashing in
    the network_index.
    """
    h = hashlib.sha3_224(f"{task_id}:{network_index}".encode()).digest()
    return f"52:54:00:{h[0]:02x}:{h[1]:02x}:{h[2]:02x}"


@dataclass
class VMNetworkInterface:
    """A single network interface attached to a VM."""

    network_name: str  # Overlay network name (e.g., "private", "public") or "standard"
    tap_device: str  # e.g., "tap-a1b2c3d4"
    mac_address: str  # e.g., "52:54:00:ab:cd:ef"
    vm_ip: str  # e.g., "10.128.64.5"
    gateway: str  # e.g., "10.128.64.1"
    bridge_name: str  # e.g., "kohaku-private" or "kohaku-br0"
    netmask: str
    prefix_len: int
    dns_servers: list[str]
    mode: str  # "overlay" or "standard"
    reservation_token: str | None = None  # Overlay only; for release


@dataclass
class VMNetworkInfo:
    """Network configuration for a VM (one or more interfaces).

    The first interface is the primary (default route, runner_url anchor).
    Additional interfaces are attached but don't override the default route.

    Backward-compatible accessors (tap_device, mac_address, vm_ip, etc.)
    return the primary interface's fields so older code keeps working.
    """

    interfaces: list[VMNetworkInterface]
    runner_url: str  # URL for VM agent to reach runner (uses primary gateway)

    @property
    def primary(self) -> VMNetworkInterface:
        return self.interfaces[0]

    # --- Backward-compat single-NIC accessors ---
    @property
    def tap_device(self) -> str:
        return self.primary.tap_device

    @property
    def mac_address(self) -> str:
        return self.primary.mac_address

    @property
    def vm_ip(self) -> str:
        return self.primary.vm_ip

    @property
    def gateway(self) -> str:
        return self.primary.gateway

    @property
    def bridge_name(self) -> str:
        return self.primary.bridge_name

    @property
    def netmask(self) -> str:
        return self.primary.netmask

    @property
    def prefix_len(self) -> int:
        return self.primary.prefix_len

    @property
    def dns_servers(self) -> list[str]:
        return self.primary.dns_servers

    @property
    def mode(self) -> str:
        return self.primary.mode

    @property
    def reservation_token(self) -> str | None:
        return self.primary.reservation_token


class VMNetworkManager:
    """
    Manages VM networking across overlay and standard modes.

    In overlay mode:
    - Uses the existing kohaku-overlay bridge (created by RunnerOverlayManager)
    - Reserves IPs from host's IPReservationManager via HTTP API
    - No bridge creation needed -- same bridge Docker containers use

    In standard mode:
    - Creates kohaku-br0 NAT bridge with 10.200.0.0/24 subnet
    - Manages a local IP pool (10.200.0.10 - 10.200.0.254)
    - Sets up iptables MASQUERADE for internet access
    - Docker doesn't need this -- Docker has its own bridge built-in
    """

    # Standard mode constants
    NAT_BRIDGE_NAME = "kohaku-br0"
    NAT_SUBNET = "10.200.0.0/24"
    NAT_GATEWAY = "10.200.0.1"
    NAT_PREFIX = 24
    NAT_POOL_START = 10  # 10.200.0.10
    NAT_POOL_END = 254  # 10.200.0.254
    DNS_SERVERS = ["8.8.8.8", "8.8.4.4"]

    def __init__(self):
        self._is_overlay: bool = False
        self._nat_bridge_ready: bool = False
        self._allocations: dict[int, VMNetworkInfo] = {}  # task_id -> info
        self._used_local_ips: set[str] = set()  # Standard mode pool tracking
        self._ipr = None

    # =========================================================================
    # Setup
    # =========================================================================

    async def setup(self) -> None:
        """
        Initialize VM network manager. Called AFTER overlay setup in runner/app.py.

        Overlay mode: verify kohaku-overlay bridge exists (no creation needed).
        Standard mode: create kohaku-br0 NAT bridge with MASQUERADE.
        """
        self._is_overlay = (
            config.OVERLAY_ENABLED
            and hasattr(config, "_overlay_configured")
            and config._overlay_configured
        )

        if self._is_overlay:
            logger.info("VM network: overlay mode -- using kohaku-overlay bridge")
            # Verify bridge exists
            exists = await asyncio.to_thread(
                self._check_bridge_exists_sync,
                RunnerOverlayManager.BRIDGE_NAME,
            )
            if not exists:
                raise RuntimeError("Overlay mode but kohaku-overlay bridge not found")
        else:
            logger.info("VM network: standard mode -- creating NAT bridge kohaku-br0")
            await asyncio.to_thread(self._setup_nat_bridge_sync)
            self._nat_bridge_ready = True

    def _check_bridge_exists_sync(self, bridge_name: str) -> bool:
        """Check if a bridge interface exists."""
        from pyroute2 import IPRoute

        ipr = IPRoute()
        try:
            for link in ipr.get_links():
                if link.get_attr("IFLA_IFNAME") == bridge_name:
                    return True
            return False
        finally:
            ipr.close()

    def _setup_nat_bridge_sync(self) -> None:
        """
        Create kohaku-br0 NAT bridge for standard mode (synchronous).

        1. Create bridge kohaku-br0
        2. Assign 10.200.0.1/24
        3. Bring bridge up
        4. Enable IP forwarding
        5. iptables MASQUERADE for 10.200.0.0/24
        6. iptables FORWARD rules for 10.200.0.0/24
        """
        from pyroute2 import IPRoute

        ipr = IPRoute()
        try:
            bridge_name = config.VM_BRIDGE_NAME

            # Check if bridge already exists
            bridge_idx = None
            for link in ipr.get_links():
                if link.get_attr("IFLA_IFNAME") == bridge_name:
                    bridge_idx = link["index"]
                    logger.info(f"NAT bridge {bridge_name} already exists")
                    break

            if bridge_idx is None:
                logger.info(f"Creating NAT bridge: {bridge_name}")
                ipr.link("add", ifname=bridge_name, kind="bridge")
                for link in ipr.get_links():
                    if link.get_attr("IFLA_IFNAME") == bridge_name:
                        bridge_idx = link["index"]
                        break

            if bridge_idx is None:
                raise RuntimeError(f"Failed to create bridge {bridge_name}")

            # Bring bridge up
            ipr.link("set", index=bridge_idx, state="up")

            # Add gateway IP if not present
            gateway = config.VM_BRIDGE_GATEWAY
            existing_addrs = list(ipr.get_addr(index=bridge_idx))
            has_ip = any(
                addr.get_attr("IFA_ADDRESS") == gateway for addr in existing_addrs
            )

            if not has_ip:
                logger.info(f"Adding IP {gateway}/{self.NAT_PREFIX} to {bridge_name}")
                ipr.addr(
                    "add",
                    index=bridge_idx,
                    address=gateway,
                    prefixlen=self.NAT_PREFIX,
                )

            # Enable IP forwarding
            try:
                with open("/proc/sys/net/ipv4/ip_forward", "w") as f:
                    f.write("1")
            except OSError as e:
                logger.warning(f"Failed to enable IP forwarding: {e}")

            # Set up iptables rules
            self._setup_nat_firewall_rules()

            logger.info(f"NAT bridge {bridge_name} ready ({gateway}/{self.NAT_PREFIX})")

        finally:
            ipr.close()

    def _setup_nat_firewall_rules(self) -> None:
        """Set up iptables MASQUERADE and FORWARD rules for NAT bridge."""
        subnet = config.VM_BRIDGE_SUBNET

        # MASQUERADE for internet access
        nat_check = [
            "iptables",
            "-t",
            "nat",
            "-C",
            "POSTROUTING",
            "-s",
            subnet,
            "!",
            "-d",
            subnet,
            "-j",
            "MASQUERADE",
        ]
        nat_add = [
            "iptables",
            "-t",
            "nat",
            "-A",
            "POSTROUTING",
            "-s",
            subnet,
            "!",
            "-d",
            subnet,
            "-j",
            "MASQUERADE",
        ]
        try:
            subprocess.run(nat_check, check=True, capture_output=True)
            logger.debug("NAT masquerade rule already exists")
        except subprocess.CalledProcessError:
            try:
                subprocess.run(nat_add, check=True, capture_output=True)
                logger.info("Added NAT masquerade rule for VM bridge")
            except subprocess.CalledProcessError as e:
                logger.warning(f"Failed to add NAT rule: {e}")

        # FORWARD rules
        for direction in ["-s", "-d"]:
            check = ["iptables", "-C", "FORWARD", direction, subnet, "-j", "ACCEPT"]
            add = ["iptables", "-I", "FORWARD", "1", direction, subnet, "-j", "ACCEPT"]
            try:
                subprocess.run(check, check=True, capture_output=True)
            except subprocess.CalledProcessError:
                try:
                    subprocess.run(add, check=True, capture_output=True)
                    logger.info(f"Added FORWARD rule for {direction} {subnet}")
                except subprocess.CalledProcessError as e:
                    logger.warning(f"Failed to add FORWARD rule: {e}")

    # =========================================================================
    # VM Network Lifecycle
    # =========================================================================

    async def create_vm_network(
        self,
        task_id: int,
        network_names: list[str] | None = None,
        reserved_ips: dict[str, str] | None = None,
    ) -> VMNetworkInfo:
        """
        Create network for a VM. Returns VMNetworkInfo.

        Args:
            task_id: VM task ID.
            network_names: Overlay networks to attach. First is primary
                (default gateway). If None and overlay enabled, uses the
                first configured overlay (legacy behavior). If None and
                overlay disabled, uses the standard NAT bridge.
            reserved_ips: {network_name: ip} pre-allocated by host. The VM
                gets these specific IPs in cloud-init.

        Behavior:
        - Overlay mode (per-network): create TAP per network -> attach to that
          network's bridge. If reserved_ips has the IP, use it directly;
          otherwise reserve from host IPAM.
        - Standard mode (single NIC fallback): create TAP -> attach to
          kohaku-br0 -> allocate from local pool.
        """
        # Standard mode: legacy single-NIC path
        if not self._is_overlay:
            info = await asyncio.to_thread(self._create_standard_vm_network, task_id)
            self._allocations[task_id] = info
            return info

        # Overlay mode: resolve network list
        # No explicit list → fall back to all configured overlays' first one
        if not network_names:
            net_list = config.get_overlay_network_names() or [None]
        else:
            net_list = list(network_names)

        interfaces: list[VMNetworkInterface] = []
        for idx, net_name in enumerate(net_list):
            iface = await self._create_overlay_interface(
                task_id,
                network_name=net_name,
                network_index=idx,
                reserved_ip=(reserved_ips or {}).get(net_name) if net_name else None,
            )
            interfaces.append(iface)

        primary = interfaces[0]
        info = VMNetworkInfo(
            interfaces=interfaces,
            runner_url=f"http://{primary.gateway}:{config.RUNNER_PORT}",
        )
        self._allocations[task_id] = info
        return info

    async def cleanup_vm_network(self, task_id: int) -> None:
        """Remove all TAP devices and release IPs for the VM."""
        info = self._allocations.pop(task_id, None)
        if info is None:
            return
        for iface in info.interfaces:
            await asyncio.to_thread(self._delete_tap_sync, iface.tap_device)
            if iface.mode == "overlay":
                if iface.reservation_token:
                    await self._release_overlay_ip_token(iface.reservation_token)
            else:
                self._release_local_ip(iface.vm_ip)

    # =========================================================================
    # Overlay mode: TAP -> kohaku-overlay, IP from host IPReservationManager
    # =========================================================================

    async def _create_overlay_interface(
        self,
        task_id: int,
        network_name: str | None,
        network_index: int,
        reserved_ip: str | None = None,
    ) -> VMNetworkInterface:
        """
        Create one overlay-network interface for a VM.

        If reserved_ip is provided (from host pre-allocation), use it directly
        and skip the host reservation API call. Otherwise, reserve from host
        IPAM via HTTP.
        """
        token: str | None = None
        if reserved_ip:
            vm_ip = reserved_ip
        else:
            vm_ip, token = await self._reserve_overlay_ip(task_id, network_name)

        tap_suffix = f"-{network_index}" if network_index > 0 else ""
        tap_name = _tap_name(task_id, suffix=tap_suffix)
        mac = _generate_mac(task_id, network_index=network_index)

        # Resolve bridge for this network. None → legacy default overlay bridge.
        if network_name:
            bridge = f"kohaku-{network_name}" if network_name != "default" else RunnerOverlayManager.BRIDGE_NAME
            gateway = config.get_container_gateway(network_name)
        else:
            bridge = RunnerOverlayManager.BRIDGE_NAME
            gateway = config._overlay_gateway

        await asyncio.to_thread(self._create_tap_sync, tap_name, bridge)

        # Derive prefix from the bridge's IP / subnet
        prefix_len = await asyncio.to_thread(
            self._get_bridge_prefix_sync, bridge, gateway
        )
        network = ipaddress.IPv4Network(f"{gateway}/{prefix_len}", strict=False)

        return VMNetworkInterface(
            network_name=network_name or "default",
            tap_device=tap_name,
            mac_address=mac,
            vm_ip=vm_ip,
            gateway=gateway,
            bridge_name=bridge,
            netmask=str(network.netmask),
            prefix_len=prefix_len,
            dns_servers=self.DNS_SERVERS,
            mode="overlay",
            reservation_token=token,
        )

    def _get_bridge_prefix_sync(self, bridge_name: str, gateway: str) -> int:
        """Read the prefix length from the bridge's assigned IP."""
        from pyroute2 import IPRoute

        ipr = IPRoute()
        try:
            for link in ipr.get_links():
                if link.get_attr("IFLA_IFNAME") != bridge_name:
                    continue
                idx = link["index"]
                for addr in ipr.get_addr(index=idx):
                    if addr.get_attr("IFA_ADDRESS") == gateway:
                        return addr["prefixlen"]
            # Fallback: parse from overlay subnet config
            subnet_cfg = OverlaySubnetConfig.parse(config.OVERLAY_SUBNET)
            return subnet_cfg.runner_prefix
        finally:
            ipr.close()

    async def _reserve_overlay_ip(
        self, task_id: int, network_name: str | None = None
    ) -> tuple[str, str]:
        """Reserve IP from host's IPReservationManager via HTTP API."""
        hostname = await asyncio.to_thread(socket.gethostname)
        host_url = config.get_host_url()
        params = {"runner": hostname, "ttl": 1800}
        if network_name:
            params["network"] = network_name

        try:
            async with httpx.AsyncClient() as client:
                response = await client.post(
                    f"{host_url}/api/overlay/ip/reserve",
                    params=params,
                    timeout=15.0,
                )
                response.raise_for_status()
                data = response.json()
                return data["ip"], data["token"]
        except Exception as e:
            raise RuntimeError(f"Failed to reserve overlay IP: {e}")

    async def _release_overlay_ip_token(self, token: str) -> None:
        """Release overlay IP via host API using reservation token."""
        host_url = config.get_host_url()
        try:
            async with httpx.AsyncClient() as client:
                await client.post(
                    f"{host_url}/api/overlay/ip/release",
                    params={"token": token},
                    timeout=10.0,
                )
        except Exception as e:
            logger.warning(f"Failed to release overlay IP token: {e}")

    # =========================================================================
    # Standard mode: TAP -> kohaku-br0, IP from local 10.200.0.0/24 pool
    # =========================================================================

    def _create_standard_vm_network(self, task_id: int) -> VMNetworkInfo:
        """Allocate local IP, create TAP on kohaku-br0 (single-NIC fallback)."""
        vm_ip = self._allocate_local_ip()
        tap_name = _tap_name(task_id)
        mac = _generate_mac(task_id)
        self._create_tap_sync(tap_name, config.VM_BRIDGE_NAME)
        network = ipaddress.IPv4Network(config.VM_BRIDGE_SUBNET)

        iface = VMNetworkInterface(
            network_name="standard",
            tap_device=tap_name,
            mac_address=mac,
            vm_ip=vm_ip,
            gateway=config.VM_BRIDGE_GATEWAY,
            bridge_name=config.VM_BRIDGE_NAME,
            netmask=str(network.netmask),
            prefix_len=self.NAT_PREFIX,
            dns_servers=self.DNS_SERVERS,
            mode="standard",
        )
        return VMNetworkInfo(
            interfaces=[iface],
            runner_url=f"http://{config.VM_BRIDGE_GATEWAY}:{config.RUNNER_PORT}",
        )

    def _allocate_local_ip(self) -> str:
        """Allocate next available IP from 10.200.0.10-254."""
        base = config.VM_BRIDGE_SUBNET.split("/")[0].rsplit(".", 1)[0]  # "10.200.0"
        for i in range(self.NAT_POOL_START, self.NAT_POOL_END + 1):
            ip = f"{base}.{i}"
            if ip not in self._used_local_ips:
                self._used_local_ips.add(ip)
                return ip
        raise RuntimeError("No available IPs in VM NAT pool")

    def _release_local_ip(self, ip: str) -> None:
        """Release local IP back to pool."""
        self._used_local_ips.discard(ip)

    # =========================================================================
    # TAP device operations (shared, uses pyroute2)
    # =========================================================================

    def _create_tap_sync(self, tap_name: str, bridge_name: str) -> None:
        """Create TAP device and attach to bridge."""
        from pyroute2 import IPRoute

        # Create TAP via ip tuntap (pyroute2's TUN/TAP API is unreliable)
        tap_exists = False
        ipr = IPRoute()
        try:
            for link in ipr.get_links():
                if link.get_attr("IFLA_IFNAME") == tap_name:
                    tap_exists = True
                    logger.info(f"TAP {tap_name} already exists")
                    break
        finally:
            ipr.close()

        if not tap_exists:
            logger.info(f"Creating TAP device: {tap_name}")
            result = subprocess.run(
                ["ip", "tuntap", "add", "dev", tap_name, "mode", "tap"],
                capture_output=True,
                text=True,
            )
            if result.returncode != 0:
                raise RuntimeError(f"Failed to create TAP {tap_name}: {result.stderr}")

        # Attach to bridge and bring up via pyroute2
        ipr = IPRoute()
        try:
            tap_idx = None
            bridge_idx = None
            for link in ipr.get_links():
                name = link.get_attr("IFLA_IFNAME")
                if name == tap_name:
                    tap_idx = link["index"]
                elif name == bridge_name:
                    bridge_idx = link["index"]

            if tap_idx is None:
                raise RuntimeError(f"TAP {tap_name} not found after creation")
            if bridge_idx is None:
                raise RuntimeError(f"Bridge {bridge_name} not found")

            ipr.link("set", index=tap_idx, master=bridge_idx)
            ipr.link("set", index=tap_idx, state="up")
            logger.info(f"TAP {tap_name} attached to bridge {bridge_name}")
        finally:
            ipr.close()

    def _delete_tap_sync(self, tap_name: str) -> None:
        """Delete TAP device via pyroute2."""
        from pyroute2 import IPRoute

        ipr = IPRoute()
        try:
            for link in ipr.get_links():
                if link.get_attr("IFLA_IFNAME") == tap_name:
                    ipr.link("del", index=link["index"])
                    logger.info(f"Deleted TAP {tap_name}")
                    return
            logger.debug(f"TAP {tap_name} not found for deletion")
        except Exception as e:
            logger.warning(f"Failed to delete TAP {tap_name}: {e}")
        finally:
            ipr.close()

    # =========================================================================
    # Cloud-init helpers
    # =========================================================================

    def get_cloud_init_network_config(self, info: VMNetworkInfo) -> dict:
        """Generate cloud-init network-config v2 for this VM.

        Uses MAC address matching instead of a hardcoded device name
        (e.g. 'ens3') because the actual name depends on PCI topology
        and firmware (could be enp0s2, ens3, eth0, etc.).
        """
        return {
            "version": 2,
            "ethernets": {
                "vmnic0": {
                    "match": {"macaddress": info.mac_address},
                    "addresses": [f"{info.vm_ip}/{info.prefix_len}"],
                    "routes": [{"to": "default", "via": info.gateway}],
                    "nameservers": {"addresses": info.dns_servers},
                }
            },
        }

    def get_vm_runner_url(self, info: VMNetworkInfo) -> str:
        """Get RUNNER_URL for VM agent to reach runner."""
        return info.runner_url


# Global instance
_vm_network_manager: VMNetworkManager | None = None


def get_vm_network_manager() -> VMNetworkManager:
    """Get global VMNetworkManager instance."""
    global _vm_network_manager
    if _vm_network_manager is None:
        _vm_network_manager = VMNetworkManager()
    return _vm_network_manager
