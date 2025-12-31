#!/usr/bin/env python3
import os
import time
import urllib.request
from pathlib import Path

import torch
import torch.distributed as dist
from torch.utils.data import Dataset, DataLoader, DistributedSampler

import pandas as pd
from ultralytics import YOLO
import ultralytics.nn.tasks  # 注册 YOLOv10DetectionModel

# 允许 YOLOv10DetectionModel 在 torch.load 时安全反序列化
torch.serialization.add_safe_globals([ultralytics.nn.tasks.RTDETRDetectionModel])


# ==============================
# 1. 分布式初始化
# ==============================
def init_distributed():
    rank = int(os.environ.get("RANK", 0))
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))

    is_distributed = world_size > 1
    if is_distributed:
        dist.init_process_group(backend="nccl", init_method="env://")
    torch.cuda.set_device(local_rank)

    return is_distributed, rank, local_rank, world_size


# ==============================
# 2. 随机数据集
# ==============================
class RandomYoloDataset(Dataset):
    def __init__(self, num_samples=5000, img_size=640):
        self.num_samples = num_samples
        self.img_size = img_size

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        img = torch.randn(3, self.img_size, self.img_size)
        return img


def build_dataloader(batch_per_gpu, world_size, rank, num_samples=5000, img_size=640):
    dataset = RandomYoloDataset(num_samples=num_samples, img_size=img_size)
    if world_size > 1:
        sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank, shuffle=True)
    else:
        sampler = None

    dataloader = DataLoader(
        dataset,
        batch_size=batch_per_gpu,
        sampler=sampler,
        shuffle=(sampler is None),
        num_workers=8,
        pin_memory=True,
        drop_last=True,
    )
    return dataloader


# ==============================
# 3. Benchmark 循环 (包含追加写入逻辑)
# ==============================
def benchmark_forward(
    model,
    dataloader,
    epochs,
    rank,
    local_rank,
    world_size,
    excel_output,
):
    device = torch.device(f"cuda:{local_rank}")
    torch.backends.cudnn.benchmark = True

    records = []

    # 获取当前的 Global Batch Size 记录到表格中，方便区分不同轮次
    current_global_bs = int(os.environ.get("BATCH_SIZE", 0))

    for epoch in range(epochs):
        if isinstance(dataloader.sampler, DistributedSampler):
            dataloader.sampler.set_epoch(epoch)

        model.eval()
        torch.cuda.reset_peak_memory_stats(device)

        epoch_start = time.time()
        total_samples_local = 0

        for step, imgs in enumerate(dataloader):
            imgs = imgs.to(device, non_blocking=True)
            bs = imgs.size(0)
            total_samples_local += bs

            with torch.cuda.amp.autocast(enabled=True):
                _ = model(imgs)

            torch.cuda.synchronize(device)

        # 聚合所有卡处理的样本数
        total_samples_tensor = torch.tensor([total_samples_local], dtype=torch.float64, device=device)
        if world_size > 1:
            dist.all_reduce(total_samples_tensor, op=dist.ReduceOp.SUM)
        total_samples = int(total_samples_tensor.item())

        epoch_time = time.time() - epoch_start
        throughput = total_samples / epoch_time if epoch_time > 0 else 0.0
        max_mem_gb = torch.cuda.max_memory_allocated(device=device) / 1e9

        if rank == 0:
            print(
                f"[Epoch {epoch + 1}] BS={current_global_bs} | "
                f"time={epoch_time:.3f}s | thr={throughput:.2f} samples/s | "
                f"GPU_mem={max_mem_gb:.2f}GB"
            )

            records.append(
                {
                    "global_batch_size": current_global_bs,
                    "epoch": epoch + 1,
                    "time(s)": epoch_time,
                    "throughput(samples/s)": throughput,
                    "GPU_mem(GB)": max_mem_gb,
                }
            )

        if world_size > 1:
            dist.barrier()

    # ==========================================
    # 【修改点】追加写入逻辑 (Append Mode)
    # ==========================================
    if rank == 0:
        new_df = pd.DataFrame(records)
        
        # 整理列顺序
        cols = ["global_batch_size", "epoch", "time(s)", "throughput(samples/s)", "GPU_mem(GB)"]
        new_df = new_df[cols]

        Path(excel_output).parent.mkdir(parents=True, exist_ok=True)

        if os.path.exists(excel_output):
            try:
                # 1. 读取旧数据
                print(f"[rank0] 发现现有文件 {excel_output}，正在追加...")
                old_df = pd.read_excel(excel_output)
                
                # 2. 拼接新旧数据
                final_df = pd.concat([old_df, new_df], ignore_index=True)
            except Exception as e:
                print(f"[rank0] ⚠️ 读取旧文件失败 ({e})，将覆盖写入。")
                final_df = new_df
        else:
            print(f"[rank0] 创建新文件 {excel_output}...")
            final_df = new_df

        # 3. 保存
        final_df.to_excel(excel_output, index=False)
        print(f"[rank0] 写入完成！总行数: {len(final_df)}")


# ==============================
# 4. 主入口
# ==============================
def main():
    weights_path = "/root/computeEvaltool/train/B1/yolov10s.pt"
    weights_url = "https://github.com/THU-MIG/yolov10/releases/download/v1.1/yolov10s.pt"
    results_dir = "/root/computeEvaltool/train/B1/results"
    
    # 结果文件名固定，以便脚本循环运行时都写入同一个文件
    excel_output = f"{results_dir}/training_yolov10-s_2.xlsx"

    is_distributed, rank, local_rank, world_size = init_distributed()
    device = torch.device(f"cuda:{local_rank}")

    # 下载权重
    if not os.path.exists(weights_path):
        if rank == 0:
            urllib.request.urlretrieve(weights_url, weights_path)
        if is_distributed:
            dist.barrier()

    # 加载模型
    if rank == 0:
        print(f"--- Process Start | Rank {rank} ---")
    
    yolo = YOLO(weights_path)
    model = yolo.model.to(device)
    model.eval()

    # 从环境变量读取 BS
    global_batch_size = int(os.environ.get("BATCH_SIZE", 128))
    batch_per_gpu = max(global_batch_size // world_size, 1)
    
    # 配置
    num_epochs = 5      # 跑 5 个 epoch 足够测出稳定速度
    num_samples = 20000 # 样本数可以少一点，只要能跑完流程即可

    dataloader = build_dataloader(
        batch_per_gpu=batch_per_gpu,
        world_size=world_size,
        rank=rank,
        num_samples=num_samples, # 简化的固定样本数
        img_size=640,
    )

    benchmark_forward(
        model=model,
        dataloader=dataloader,
        epochs=num_epochs,
        rank=rank,
        local_rank=local_rank,
        world_size=world_size,
        excel_output=excel_output,
    )

    if is_distributed and dist.is_initialized():
        dist.destroy_process_group()


if __name__ == "__main__":
    main()