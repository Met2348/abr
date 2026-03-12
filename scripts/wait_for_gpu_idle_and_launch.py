#!/usr/bin/env python
"""Wait for one GPU to become idle, then launch a command.

English:
Use this as a safe overnight wrapper on crowded servers. It polls one GPU and
only starts the real experiment when both memory use and utilization drop below
configured thresholds.

中文：
在拥挤服务器上安全等待某张卡空闲后再启动真正实验，避免和已有任务直接抢显存/算力。
"""
from __future__ import annotations

import argparse
import datetime as dt
import os
import shlex
import subprocess
import sys
import time
from pathlib import Path


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument('--gpu-id', type=int, required=True)
    p.add_argument('--max-used-mib', type=int, default=6000)
    p.add_argument('--max-util', type=int, default=20)
    p.add_argument('--poll-seconds', type=int, default=300)
    p.add_argument('--log-file', required=True)
    p.add_argument('--workdir', default='.')
    p.add_argument('--command', required=True)
    return p.parse_args()


def log(path: Path, msg: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    ts = dt.datetime.now().astimezone().strftime('%Y-%m-%d %H:%M:%S %z')
    with path.open('a', encoding='utf-8') as f:
        f.write(f'[{ts}] {msg}\n')


def gpu_state(gpu_id: int) -> tuple[int, int, int]:
    out = subprocess.check_output(
        [
            'nvidia-smi',
            '--query-gpu=memory.used,memory.total,utilization.gpu',
            '--format=csv,noheader,nounits',
            '-i',
            str(gpu_id),
        ],
        text=True,
    ).strip()
    used, total, util = [int(x.strip()) for x in out.split(',')]
    return used, total, util


def main() -> int:
    args = parse_args()
    log_path = Path(args.log_file)
    workdir = Path(args.workdir)
    log(log_path, f'gpu-idle watcher started: gpu={args.gpu_id} max_used_mib={args.max_used_mib} max_util={args.max_util}')
    log(log_path, f'launch command: {args.command}')
    while True:
        try:
            used, total, util = gpu_state(args.gpu_id)
            log(log_path, f'gpu{args.gpu_id} used={used}MiB total={total}MiB util={util}%')
            if used <= args.max_used_mib and util <= args.max_util:
                log(log_path, f'gpu{args.gpu_id} considered idle; launching command')
                proc = subprocess.run(
                    ['bash', '-lc', args.command],
                    cwd=str(workdir),
                    stdout=log_path.open('a', encoding='utf-8'),
                    stderr=subprocess.STDOUT,
                    text=True,
                    check=False,
                    env=os.environ.copy(),
                )
                log(log_path, f'command exited code={proc.returncode}')
                return proc.returncode
        except Exception as exc:  # noqa: BLE001
            log(log_path, f'watcher_exception={type(exc).__name__}: {exc}')
        time.sleep(args.poll_seconds)


if __name__ == '__main__':
    raise SystemExit(main())
