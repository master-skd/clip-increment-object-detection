import subprocess
import time
import os

# ================= Config =================
FREE_MEM_REQUIRED_MB = 50 * 1024   # 每张卡至少空闲 60GB
TARGET_GPUS = [0,1,2,3]         # 只检测这八张卡
MIN_GPUS = 4                       # 至少需要几张满足条件的卡
MAX_GPUS = 4                       # 最终使用几张卡
CHECK_INTERVAL = 70                # 检查间隔（秒）
LAUNCH_CMD = "bash incremental_train_val.sh"
# ==========================================


def get_free_gpus(required_free_mb, target_gpus):
    """检测目标 GPU 中空闲显存满足阈值的卡"""
    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=index,memory.total,memory.used",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            check=True,
        )

        free_ids = []
        for line in result.stdout.strip().split("\n"):
            if not line.strip():
                continue
            parts = [p.strip() for p in line.split(",")]
            if len(parts) != 3:
                continue

            gpu_id = int(parts[0])
            mem_total = int(parts[1])
            mem_used = int(parts[2])

            if gpu_id not in target_gpus:
                continue

            mem_free = mem_total - mem_used
            if mem_free >= required_free_mb:
                free_ids.append(gpu_id)
            else:
                print(
                    f"  GPU {gpu_id}: free {mem_free}MB "
                    f"(< {required_free_mb}MB), used {mem_used}MB / total {mem_total}MB"
                )
        return free_ids

    except Exception as e:
        print(f"Check failed: {e}")
        return []


def main():
    if MAX_GPUS < MIN_GPUS:
        raise ValueError("MAX_GPUS must be >= MIN_GPUS")
    if MIN_GPUS <= 0:
        raise ValueError("MIN_GPUS must be > 0")
    if CHECK_INTERVAL <= 0:
        raise ValueError("CHECK_INTERVAL must be > 0")

    start_msg = (
        f"监控已启动：等待 >= {MIN_GPUS} 张空闲显存 >= {FREE_MEM_REQUIRED_MB}MB 的 GPU，"
        f"检测范围 {TARGET_GPUS}。"
    )
    print(start_msg)

    while True:
        free_gpus = get_free_gpus(FREE_MEM_REQUIRED_MB, TARGET_GPUS)

        if len(free_gpus) >= MIN_GPUS:
            selected_gpus = free_gpus[:MAX_GPUS]
            gpu_str = ",".join(str(g) for g in selected_gpus)

            print(f"目标达成！已锁定 GPU: {gpu_str}，启动训练...")
            cmd = f"CUDA_VISIBLE_DEVICES={gpu_str} {LAUNCH_CMD}"

            ret = subprocess.run(cmd, shell=True)
            if ret.returncode == 0:
                print("训练程序已正常退出。")
            else:
                print(f"训练程序异常退出，返回码: {ret.returncode}")
            break

        current_time = time.strftime("%H:%M:%S", time.localtime())
        print(f"[{current_time}] 找到 {len(free_gpus)} 张满足条件的显卡，继续等待...")
        time.sleep(CHECK_INTERVAL)


if __name__ == "__main__":
    main()
