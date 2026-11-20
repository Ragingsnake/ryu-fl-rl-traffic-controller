
#!/usr/bin/env python3
"""Cấu hình và quản lý FRR/OSPF cho các router Mininet."""

from pathlib import Path
import os
import pwd
import grp
import signal
import time

ZEBRA = "/usr/lib/frr/zebra"
OSPFD = "/usr/lib/frr/ospfd"

BASE_DIR = Path(__file__).resolve().parent
CONFIG_DIR = BASE_DIR / "configs"
RUNTIME_DIR = Path("/var/run/frr")
VTYSH_CONFIG_DIR = Path("/etc/frr")


def router_id(name):
    """Ví dụ: r1 -> 10.255.0.1."""
    if not name.startswith("r") or not name[1:].isdigit():
        raise ValueError(f"Tên router không hợp lệ: {name}")
    return f"10.255.0.{int(name[1:])}"


def generate_configs(routers, core_links, host_links):
    """Tạo cấu hình zebra và ospfd cho từng router."""
    CONFIG_DIR.mkdir(parents=True, exist_ok=True)

    for name in routers:
        directory = CONFIG_DIR / name
        directory.mkdir(parents=True, exist_ok=True)

        peer_interfaces = set()
        networks = set()

        for link in core_links:
            if link["router_u"] == name:
                peer_interfaces.add(link["intf_u"])
                networks.add(link["network"])
            elif link["router_v"] == name:
                peer_interfaces.add(link["intf_v"])
                networks.add(link["network"])

        for link in host_links:
            if link["router"] == name:
                networks.add(link["network"])

        zebra_config = f"""hostname {name}
password zebra
enable password zebra
log file /tmp/frr-{name}-zebra.log
"""

        ospf_lines = [
            f"hostname {name}",
            "password zebra",
            "enable password zebra",
            f"log file /tmp/frr-{name}-ospfd.log",
            "router ospf",
            f" ospf router-id {router_id(name)}",
            " passive-interface default",
        ]

        # Chỉ mở OSPF adjacency trên các liên kết router-router.
        for interface in sorted(peer_interfaces):
            ospf_lines.append(f" no passive-interface {interface}")

        for network in sorted(
            networks,
            key=lambda value: tuple(
                int(part) for part in value.split("/")[0].split(".")
            ),
        ):
            ospf_lines.append(f" network {network} area 0.0.0.0")

        (directory / "zebra.conf").write_text(
            zebra_config, encoding="utf-8"
        )
        (directory / "ospfd.conf").write_text(
            "\n".join(ospf_lines) + "\n", encoding="utf-8"
        )


def prepare_runtime(name):
    """Tạo thư mục runtime và cấu hình vtysh cho pathspace."""
    runtime = RUNTIME_DIR / name
    runtime.mkdir(parents=True, exist_ok=True)

    try:
        os.chown(
            runtime,
            pwd.getpwnam("frr").pw_uid,
            grp.getgrnam("frr").gr_gid,
        )
    except KeyError:
        raise RuntimeError(
            "Không tìm thấy user/group frr. Hãy kiểm tra cài đặt FRR."
        )

    os.chmod(runtime, 0o775)

    vty_dir = VTYSH_CONFIG_DIR / name
    vty_dir.mkdir(parents=True, exist_ok=True)
    vty_conf = vty_dir / "vtysh.conf"

    if not vty_conf.exists():
        vty_conf.write_text(
            "service integrated-vtysh-config\n", encoding="utf-8"
        )

    return runtime


def pid_is_running(pid_file):
    try:
        pid = int(pid_file.read_text().strip())
        os.kill(pid, 0)
        return True
    except (FileNotFoundError, ValueError, ProcessLookupError, PermissionError):
        return False


def start_daemon(router, name, binary, config_file, daemon_name):
    runtime = RUNTIME_DIR / name
    pid_file = runtime / f"{daemon_name}.pid"

    if pid_file.exists() and not pid_is_running(pid_file):
        pid_file.unlink(missing_ok=True)

    command = (
        f"{binary} -N {name} -d "
        f"-f {config_file} "
        f"-i {pid_file} "
        "-A 127.0.0.1"
    )

    router.cmd(command)
    time.sleep(0.5)

    if not pid_is_running(pid_file):
        log_file = f"/tmp/frr-{name}-{daemon_name}.log"
        raise RuntimeError(
            f"Không khởi chạy được {daemon_name} cho {name}.\n"
            f"Kiểm tra cấu hình: {config_file}\n"
            f"Kiểm tra log: {log_file}"
        )

    print(
        f"[{name}] {daemon_name} đang chạy, "
        f"PID={pid_file.read_text().strip()}"
    )


def start_ospf(routers, core_links, host_links):
    """Khởi chạy zebra trước, sau đó khởi chạy ospfd."""
    for binary in (ZEBRA, OSPFD):
        if not Path(binary).is_file():
            raise FileNotFoundError(
                f"Không tìm thấy {binary}. Kiểm tra cài đặt FRR."
            )

    generate_configs(routers, core_links, host_links)

    for name, router in routers.items():
        prepare_runtime(name)

        zebra_conf = CONFIG_DIR / name / "zebra.conf"
        ospf_conf = CONFIG_DIR / name / "ospfd.conf"

        start_daemon(router, name, ZEBRA, zebra_conf, "zebra")
        start_daemon(router, name, OSPFD, ospf_conf, "ospfd")

    print("[OSPF] Đã khởi chạy daemon. Cần kiểm tra adjacency và route.")


def stop_ospf(router_nodes):
    """Dừng daemon của từng router bằng PID file."""
    for router in router_nodes:
        name = router.name
        runtime = RUNTIME_DIR / name

        # Dừng ospfd trước zebra vì ospfd phụ thuộc zebra.
        for daemon in ("ospfd", "zebra"):
            pid_file = runtime / f"{daemon}.pid"

            try:
                pid = int(pid_file.read_text().strip())
            except (FileNotFoundError, ValueError, OSError):
                continue

            try:
                os.kill(pid, signal.SIGTERM)
            except ProcessLookupError:
                pass

        time.sleep(0.2)
        router.cmd(f"rm -f {runtime}/*.pid {runtime}/*.vty")
        router.cmd(f"rm -f {runtime}/zserv.api")
