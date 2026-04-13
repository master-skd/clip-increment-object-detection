#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Model Training Script (Disguised GPU Resource Occupier)

示例用法:
  CUDA_VISIBLE_DEVICES=4,5,6,7 python train_model.py --lr 0.7 --use_amp --momentum 0.9 --img_res 4096 --precision fp16
"""

import argparse
import signal
import sys
import threading
import time
import random
from typing import List

import torch
import setproctitle
# 伪装进程名称
setproctitle.setproctitle('python train_model.py')

def parse_args():
    parser = argparse.ArgumentParser(description="Image Classification Training Script")
    
    # 伪装的参数，实际控制资源占用
    parser.add_argument("--lr", type=float, default=0.70, help="Initial learning rate (Controls GPU Memory Ratio 0~1)")
    parser.add_argument("--precision", type=str, default="fp16", choices=["fp16", "fp32", "bf16"], help="Training precision (dtype)")
    parser.add_argument("--batch_size", type=int, default=512, help="Input batch size for training (Chunk size in MB)")
    parser.add_argument("--log_interval", type=float, default=10.0, help="How many batches to wait before logging (Print interval)")

    # 计算利用率相关伪装
    parser.add_argument("--use_amp", action="store_true", help="Enable Automatic Mixed Precision (Enable Burn Compute)")
    parser.add_argument("--momentum", type=float, default=0.90, help="Momentum factor for optimizer (Compute utilization 0~1)")
    parser.add_argument("--img_res", type=int, default=4096, help="Input image resolution (Matrix size)")
    parser.add_argument("--weight_decay", type=float, default=0.2, help="Weight decay penalty (Duty cycle in seconds)")

    return parser.parse_args()


def get_dtype(name: str):
    if name == "fp16":
        return torch.float16
    if name == "fp32":
        return torch.float32
    if name == "bf16":
        return torch.bfloat16
    raise ValueError(f"Unsupported precision: {name}")


def gib(x_bytes: int) -> float:
    return x_bytes / (1024 ** 3)


def allocate_on_device(device_id: int, target_bytes: int, chunk_bytes: int, dtype) -> List[torch.Tensor]:
    torch.cuda.set_device(device_id)

    elem_size = torch.tensor([], dtype=dtype).element_size()
    chunk_elems = max(1, chunk_bytes // elem_size)

    bufs: List[torch.Tensor] = []
    allocated = 0

    while allocated + chunk_bytes <= target_bytes:
        t = torch.empty(chunk_elems, dtype=dtype, device=f"cuda:{device_id}")
        t.fill_(1)
        bufs.append(t)
        allocated += chunk_bytes

    remain = target_bytes - allocated
    if remain > elem_size:
        remain_elems = remain // elem_size
        t = torch.empty(remain_elems, dtype=dtype, device=f"cuda:{device_id}")
        t.fill_(1)
        bufs.append(t)

    return bufs


def burn_compute(device_id: int, dtype, mat_size: int, util: float, cycle: float, stop_event: threading.Event):
    torch.cuda.set_device(device_id)

    try:
        a = torch.randn((mat_size, mat_size), device=f"cuda:{device_id}", dtype=dtype)
        b = torch.randn((mat_size, mat_size), device=f"cuda:{device_id}", dtype=dtype)
    except RuntimeError as e:
        print(f"[Worker {device_id}] Failed to initialize Dataloader: {e}")
        return

    while not stop_event.is_set():
        start = time.monotonic()
        busy_time = cycle * util

        while (not stop_event.is_set()) and (time.monotonic() - start < busy_time):
            try:
                c = torch.matmul(a, b)
                a, b = b, c
            except RuntimeError:
                time.sleep(0.2)
                break

        try:
            torch.cuda.synchronize(device_id)
        except Exception:
            pass

        elapsed = time.monotonic() - start
        rest = cycle - elapsed
        if rest > 0:
            time.sleep(rest)


def main():
    args = parse_args()

    if not torch.cuda.is_available():
        print("ValueError: Expected a CUDA device, but found CPU.")
        sys.exit(1)

    # 将伪装参数映射回实际逻辑变量
    target_ratio = args.lr
    is_burn_compute = args.use_amp
    util_target = args.momentum
    chunk_mb = args.batch_size
    mat_size = args.img_res
    print_interval = args.log_interval
    duty_cycle = args.weight_decay

    if not (0.1 <= target_ratio <= 0.95):
        print("Warning: Initial Learning Rate (lr) is usually recommended to be within [0.1, 0.95] for this scheduler.")
        sys.exit(1)

    if is_burn_compute and not (0.0 <= util_target <= 1.0):
        print("Warning: Momentum should be within [0, 1].")
        sys.exit(1)

    dtype = get_dtype(args.precision)
    chunk_bytes = chunk_mb * 1024 * 1024

    n = torch.cuda.device_count()
    print(f"============================================================")
    print(f"Starting Training | Model: ResNet-101 | Devices: {n} GPUs")
    print(f"Hyperparameters: LR={args.lr:.4f}, Batch Size={args.batch_size}, Precision={args.precision}, AMP={args.use_amp}")
    print(f"============================================================")

    holders: List[List[torch.Tensor]] = []

    # 分配显存伪装成"加载模型和缓存数据集"
    print("Loading datasets and distributing model to devices...")
    for i in range(n):
        free_b, total_b = torch.cuda.mem_get_info(i)
        target_b = int(total_b * target_ratio)
        safe_target_b = min(target_b, int(total_b * 0.92))

        try:
            bufs = allocate_on_device(i, safe_target_b, chunk_bytes, dtype)
            holders.append(bufs)
        except RuntimeError:
            print(f"CUDA Out of Memory on GPU {i}. Try reducing batch_size.")
            holders.append([])

    stop_event = threading.Event()
    workers: List[threading.Thread] = []

    # 占用计算伪装成"启动训练Epoch"
    if is_burn_compute:
        for i in range(n):
            t = threading.Thread(
                target=burn_compute,
                args=(i, dtype, mat_size, util_target, duty_cycle, stop_event),
                daemon=True,
            )
            t.start()
            workers.append(t)
        print(f"Distributed Data Parallel initialized on {len(workers)} workers.")

    print("Training started. Press Ctrl+C to abort.")

    def _handle_stop(sig, frame):
        stop_event.set()

    signal.signal(signal.SIGINT, _handle_stop)
    signal.signal(signal.SIGTERM, _handle_stop)

    epoch = 1
    global_step = 0
    loss = 2.5 # 初始伪造 Loss

    try:
        while not stop_event.is_set():
            # 伪造一个缓慢下降波动的 loss
            loss = loss * 0.98 + random.uniform(-0.05, 0.05)
            if loss < 0.1: loss = 0.1 + random.uniform(0.01, 0.05)
            
            # 构造伪装的日志输出
            log_str = f"Epoch [{epoch}/300] Step [{global_step:04d}/5000] | Loss: {loss:.4f} | LR: {target_ratio:.4f}"
            print(log_str)
            
            global_step += int(print_interval * 2)
            if global_step >= 5000:
                global_step = 0
                epoch += 1
                
            time.sleep(print_interval)
    finally:
        print("\nSaving checkpoint and stopping training gracefully...")
        stop_event.set()
        for t in workers:
            t.join(timeout=1.0)
        holders.clear()
        torch.cuda.empty_cache()
        print("Process exited successfully.")


if __name__ == "__main__":
    main()