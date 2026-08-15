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

# 环境变量
ENV = os.environ.copy()
ENV["CIRCBASE_FASTA"] = "C:/Users/颜子壹/Documents/circbase_seqs.fa.gz"
ENV["DATA_OUT"] = "C:/tmp/test_isrna/rhofold_data"


def count_products():
    """统计已完成的样本数."""
    if not DATA_DIR.exists():
        return 0
    return len([d for d in DATA_DIR.iterdir() if d.is_dir() and d.name.startswith("s") and
                (d / f"{d.name}_p.npy").exists()])


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


def start_process():
    """启动数据生成进程."""
    cmd = [PYTHON, "-u", SCRIPT,
           "--n-workers", "4",
           "--n-samples", "113539",
           "--max-len", "2000",
           "--min-len", "50",
           "--n-anneal", "300",
           "--resume"]
    with open(LOG_FILE, "a") as f:
        proc = subprocess.Popen(
            cmd, cwd=str(ROOT), env=ENV,
            stdout=f, stderr=subprocess.STDOUT,
            creationflags=subprocess.CREATE_NEW_PROCESS_GROUP
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

    print(f"[watchdog] 启动监控")
    print(f"  检查间隔: {args.check_interval}s")
    print(f"  卡住阈值: {args.stall_threshold}s")
    print(f"  最大重启: {args.max_restarts or '无限'}")
    print(f"  初始产物数: {last_count}")

    # 如果没有 Python 进程在跑，先启动一个
    if get_process_list() == 0:
        pid = start_process()
        print(f"  启动进程: PID {pid}")
        time.sleep(30)  # 等进程初始化

    while True:
        time.sleep(args.check_interval)

        current_count = count_products()
        n_procs = get_process_list()
        now = time.time()

        # 检查产物是否增长
        if current_count > last_count:
            last_count = current_count
            last_change_time = now
            stall_secs = 0
        else:
            stall_secs = now - last_change_time

        status = "RUNNING" if n_procs > 0 else "STOPPED"
        if stall_secs > args.stall_threshold:
            status = "STALLED"

        print(f"[{time.strftime('%H:%M:%S')}] {status} | "
              f"products: {current_count} | processes: {n_procs} | "
              f"stall: {stall_secs:.0f}s | restarts: {restart_count}")

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
            print(f"[watchdog] 新进程 PID: {pid}")
            time.sleep(30)  # 等进程初始化


if __name__ == "__main__":
    main()
