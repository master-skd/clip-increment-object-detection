#!/usr/bin/env python3
import argparse
import re
import signal
import sys
import time
from typing import List

import torch


def parse_size_to_bytes(size_text: str) -> int:
    """Parse strings like 1024, 1024MB, 8GB, 1.5GiB into bytes."""
    text = size_text.strip()
    m = re.fullmatch(r"(?i)\s*([0-9]+(?:\.[0-9]+)?)\s*([kmgt]?i?b?)?\s*", text)
    if not m:
        raise ValueError(f"Invalid memory size: {size_text}")

    value = float(m.group(1))
    unit = (m.group(2) or "mb").lower()

    if unit in ("b",):
        mul = 1
    elif unit in ("k", "kb"):
        mul = 1000
    elif unit in ("ki", "kib"):
        mul = 1024
    elif unit in ("m", "mb"):
        mul = 1000 ** 2
    elif unit in ("mi", "mib"):
        mul = 1024 ** 2
    elif unit in ("g", "gb"):
        mul = 1000 ** 3
    elif unit in ("gi", "gib"):
        mul = 1024 ** 3
    elif unit in ("t", "tb"):
        mul = 1000 ** 4
    elif unit in ("ti", "tib"):
        mul = 1024 ** 4
    else:
        raise ValueError(f"Unsupported unit: {unit}")

    size_bytes = int(value * mul)
    if size_bytes <= 0:
        raise ValueError("Memory size must be > 0")
    return size_bytes


def format_mib(num_bytes: int) -> str:
    return f"{num_bytes / (1024 ** 2):.2f} MiB"


def parse_gpu_list(text: str) -> List[int]:
    gpus = []
    for part in text.split(","):
        p = part.strip()
        if not p:
            continue
        if not p.isdigit():
            raise ValueError(f"Invalid GPU id: {p}")
        gpus.append(int(p))
    if not gpus:
        raise ValueError("GPU list is empty")
    return sorted(set(gpus))


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Reserve GPU VRAM on one or multiple devices with minimal utilization."
    )
    parser.add_argument(
        "--gpu",
        type=int,
        default=None,
        help="Single GPU index, e.g. 0 (legacy option, mutually exclusive with --gpus)",
    )
    parser.add_argument(
        "--gpus",
        type=str,
        default=None,
        help="GPU list, e.g. 0,1,3",
    )
    parser.add_argument(
        "--memory",
        type=str,
        required=True,
        help="If --memory-per-gpu is not set: total memory to reserve (auto split across GPUs)",
    )
    parser.add_argument(
        "--memory-per-gpu",
        action="store_true",
        help="Treat --memory as per-GPU reservation instead of total reservation",
    )
    parser.add_argument(
        "--chunk-mb",
        type=int,
        default=256,
        help="Allocate in chunks (MiB) to reduce big-allocation failures (default: 256)",
    )
    parser.add_argument(
        "--hold-seconds",
        type=int,
        default=0,
        help="How long to hold memory; <=0 means hold until Ctrl+C",
    )
    args = parser.parse_args()

    if not torch.cuda.is_available():
        print("CUDA is not available.", file=sys.stderr)
        return 1

    if args.gpu is not None and args.gpus is not None:
        print("Use either --gpu or --gpus, not both.", file=sys.stderr)
        return 1

    if args.gpus is not None:
        try:
            gpu_ids = parse_gpu_list(args.gpus)
        except ValueError as e:
            print(str(e), file=sys.stderr)
            return 1
    elif args.gpu is not None:
        gpu_ids = [args.gpu]
    else:
        print("You must provide --gpu or --gpus.", file=sys.stderr)
        return 1

    device_count = torch.cuda.device_count()
    for gid in gpu_ids:
        if gid < 0 or gid >= device_count:
            print(f"Invalid GPU {gid}. Available range: 0..{device_count - 1}", file=sys.stderr)
            return 1

    if args.chunk_mb <= 0:
        print("--chunk-mb must be > 0", file=sys.stderr)
        return 1

    try:
        memory_arg_bytes = parse_size_to_bytes(args.memory)
    except ValueError as e:
        print(str(e), file=sys.stderr)
        return 1

    num_gpus = len(gpu_ids)
    if args.memory_per_gpu:
        per_gpu_targets = [memory_arg_bytes] * num_gpus
    else:
        base = memory_arg_bytes // num_gpus
        rem = memory_arg_bytes % num_gpus
        per_gpu_targets = [base + (1 if i < rem else 0) for i in range(num_gpus)]

    chunk_bytes = args.chunk_mb * 1024 * 1024

    for gid, target_bytes in zip(gpu_ids, per_gpu_targets):
        device = torch.device(f"cuda:{gid}")
        free_bytes, total_bytes = torch.cuda.mem_get_info(device)
        print(
            f"GPU {gid} ({torch.cuda.get_device_name(device)}): "
            f"free={format_mib(free_bytes)}, total={format_mib(total_bytes)}, "
            f"target={format_mib(target_bytes)}"
        )
        if target_bytes > free_bytes:
            print(
                f"GPU {gid} does not have enough free memory: "
                f"{format_mib(target_bytes)} > {format_mib(free_bytes)}",
                file=sys.stderr,
            )
            return 1

    buffers = {gid: [] for gid in gpu_ids}
    allocated = {gid: 0 for gid in gpu_ids}

    stop = {"value": False}

    def handle_signal(signum, _frame):
        stop["value"] = True
        print(f"\nReceived signal {signum}, releasing memory...")

    signal.signal(signal.SIGINT, handle_signal)
    signal.signal(signal.SIGTERM, handle_signal)

    try:
        for gid, target_bytes in zip(gpu_ids, per_gpu_targets):
            device = torch.device(f"cuda:{gid}")
            torch.cuda.set_device(gid)
            while allocated[gid] < target_bytes:
                n = min(chunk_bytes, target_bytes - allocated[gid])
                buffers[gid].append(torch.empty((n,), dtype=torch.uint8, device=device))
                allocated[gid] += n
            torch.cuda.synchronize(device)
            print(f"Reserved {format_mib(allocated[gid])} on GPU {gid}.")

        print("Process is now idle; memory stays reserved until exit.")

        if args.hold_seconds > 0:
            end_time = time.time() + args.hold_seconds
            while not stop["value"] and time.time() < end_time:
                time.sleep(1)
        else:
            while not stop["value"]:
                time.sleep(3600)

    except RuntimeError as e:
        print(f"CUDA allocation failed: {e}", file=sys.stderr)
        return 1
    finally:
        for gid in gpu_ids:
            buffers[gid].clear()
        torch.cuda.empty_cache()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
