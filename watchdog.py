# -*- coding: utf-8 -*-
"""watchdog.py — 数据生成进程监控器

监控 generate_data_rhofold.py 的运行状态，如果停止或卡住，自动重启。

用法:
  python watchdog.py
  python watchdog.py --check-interval 60 --stall-threshold 300
"""
import subprocess
import time
import os
import sys
import argparse
from pathlib import Path

ROOT = Path(__file__).resolve().parent
LOG_FILE = Path("C:/tmp/test_isrna/generate.log")
PYTHON = "C:/ana/envs/comfyui/python.exe"
SCRIPT = "generate_data_rhofold.py"
DATA_DIR = Path("C:/tmp/test_isrna/rhofold_data/rhofold_data")
DISK_LOW_GB = 10  # 磁盘低于此值 (GB) 时清理


def count_products():
    """统计已完成的样本数."""
    if not DATA_DIR.exists():
        return 0
    return len([d for d in DATA_DIR.iterdir() if d.is_dir() and d.name.startswith("s") and
                (d / f"{d.name}_p.npy").exists()])


def get_disk_free_gb(drive="C"):
    """获取磁盘剩余空间 (GB)."""
    try:
        import shutil
        total, used, free = shutil.disk_usage(f"{drive}:/")
        return free / (1024**3)
    except Exception:
        return 999


def cleanup_temp():
    """清理临时文件释放空间."""
    cleaned = 0
    # 清理 l1_rhofold 缓存 (每个 ~24KB, 上万个)
    try:
        for d in DATA_DIR.iterdir():
            l1 = d / "l1_rhofold"
            if l1.exists():
                import shutil
                shutil.rmtree(l1, ignore_errors=True)
                cleaned += 1
    except Exception:
        pass
    # 清理旧的 isrna_verify/test 目录
    tmp_root = Path("C:/tmp")
    for d in tmp_root.iterdir():
        if d.is_dir() and d.name.startswith("isrna_") and d.name not in (
            "rhofold_data", "rhofold_split", "rhofold_1w", "rhofold_4w",
            "rhofold_8w", "rhofold_long", "isrna_bin"):
            try:
                import shutil
                shutil.rmtree(d, ignore_errors=True)
                cleaned += 1
            except Exception:
                pass
    return cleaned


def check_and_cleanup_disk():
    """检查磁盘空间, 低于阈值则清理."""
    free_gb = get_disk_free_gb()
    if free_gb < DISK_LOW_GB:
        print(f"  [磁盘] {free_gb:.1f}GB < {DISK_LOW_GB}GB, 清理中...")
        n = cleanup_temp()
        free_after = get_disk_free_gb()
        print(f"  [磁盘] 清理了 {n} 个目录, 剩余 {free_after:.1f}GB")
        return free_after
    return free_gb


def get_process_list():
    """获取当前运行的 Python 进程."""
    try:
        r = subprocess.run(
            ["tasklist", "/FI", "IMAGENAME eq python.exe"],
            capture_output=True, text=True, timeout=10
        )
        lines = [l for l in r.stdout.split("\n") if "python.exe" in l.lower()]
        return len(lines)
    except Exception:
        return 0


BAT_FILE = ROOT / "start_generate.bat"


def start_process():
    """启动数据生成进程 (用 .bat 文件, 跟双击运行一样)."""
    with open(LOG_FILE, "a", buffering=1) as f:
        proc = subprocess.Popen(
            str(BAT_FILE), shell=True, stdout=f, stderr=subprocess.STDOUT,
        )
    return proc.pid


def kill_process():
    """只杀数据生成进程 (generate_data_rhofold.py), 不杀 watchdog 自己."""
    try:
        # 找到 generate_data_rhofold.py 的进程 PID
        r = subprocess.run(
            ["wmic", "process", "where",
             "CommandLine like '%generate_data_rhofold%' and Name='python.exe'",
             "get", "ProcessId"],
            capture_output=True, text=True, timeout=10
        )
        pids = []
        for line in r.stdout.split("\n"):
            line = line.strip()
            if line.isdigit():
                pids.append(int(line))
        for pid in pids:
            subprocess.run(["taskkill", "/F", "/PID", str(pid)],
                           capture_output=True, timeout=10)
        if pids:
            print(f"  杀掉进程: {pids}")
        else:
            print("  没找到数据生成进程")
    except Exception as e:
        print(f"  杀进程失败: {e}")


def main():
    pa = argparse.ArgumentParser()
    pa.add_argument("--check-interval", type=int, default=60,
                    help="检查间隔 (秒)")
    pa.add_argument("--stall-threshold", type=int, default=300,
                    help="产物数不增长超过此秒数视为卡住")
    pa.add_argument("--max-restarts", type=int, default=10,
                    help="最大重启次数 (0=无限)")
    args = pa.parse_args()

    restart_count = 0
    last_count = count_products()
    last_change_time = time.time()
    process_start_time = time.time()
    grace_period = 600  # 启动后 10 分钟不检查卡住 (RhoFold 加载需要时间)

    print(f"[watchdog] 启动监控")
    print(f"  检查间隔: {args.check_interval}s")
    print(f"  卡住阈值: {args.stall_threshold}s")
    print(f"  启动宽限: {grace_period}s")
    print(f"  最大重启: {args.max_restarts or '无限'}")
    print(f"  初始产物数: {last_count}")

    # 如果没有 Python 进程在跑，先启动一个
    if get_process_list() == 0:
        pid = start_process()
        print(f"  启动进程: PID {pid}")
        process_start_time = time.time()
        time.sleep(30)  # 等进程初始化

    while True:
        time.sleep(args.check_interval)

        current_count = count_products()
        n_procs = get_process_list()
        now = time.time()

        # 检查磁盘空间 (每轮都检查)
        free_gb = check_and_cleanup_disk()

        # 检查产物是否增长
        if current_count > last_count:
            last_count = current_count
            last_change_time = now
            stall_secs = 0
        else:
            stall_secs = now - last_change_time

        # 启动宽限期内不判定为卡住
        elapsed = now - process_start_time
        in_grace = elapsed < grace_period

        status = "RUNNING" if n_procs > 0 else "STOPPED"
        if not in_grace and stall_secs > args.stall_threshold:
            status = "STALLED"

        print(f"[{time.strftime('%H:%M:%S')}] {status} | "
              f"products: {current_count} | processes: {n_procs} | "
              f"disk: {free_gb:.0f}GB | stall: {stall_secs:.0f}s | restarts: {restart_count}")

        # 如果停止或卡住，重启
        if status in ("STOPPED", "STALLED"):
            if args.max_restarts > 0 and restart_count >= args.max_restarts:
                print(f"[watchdog] 达到最大重启次数 {args.max_restarts}，退出")
                break

            print(f"[watchdog] {status}，重启... (#{restart_count + 1})")
            kill_process()
            time.sleep(5)

            pid = start_process()
            restart_count += 1
            last_change_time = time.time()
            process_start_time = time.time()  # 重置宽限期
            print(f"[watchdog] 新进程 PID: {pid}")
            time.sleep(30)  # 等进程初始化


if __name__ == "__main__":
    main()
