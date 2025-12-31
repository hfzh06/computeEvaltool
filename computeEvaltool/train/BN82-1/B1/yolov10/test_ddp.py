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
# 1. 分布式初始化（仅用于多进程 & all_reduce，不用 DDP）
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
# 2. 随机数据集（仅用于产生稳定负载）
# ==============================
class RandomYoloDataset(Dataset):
    def __init__(self, num_samples=5000, img_size=640):
        self.num_samples = num_samples
        self.img_size = img_size

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        img = torch.randn(3, self.img_size, self.img_size)  # 随机图像
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
# 3. 前向-only Benchmark 训练循环
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

    for epoch in range(epochs):
        if isinstance(dataloader.sampler, DistributedSampler):
            dataloader.sampler.set_epoch(epoch)

        model.eval()
        torch.cuda.reset_peak_memory_stats(device)

        # rank0 统计“全局”的样本数和时间，所以用 all_reduce 聚合
        epoch_start = time.time()
        total_samples_local = 0

        for step, imgs in enumerate(dataloader):
            imgs = imgs.to(device, non_blocking=True)

            bs = imgs.size(0)
            total_samples_local += bs

            # 纯前向推理
            with torch.cuda.amp.autocast(enabled=True):
                _ = model(imgs)

            # 为了计时准确，确保本 step 完成
            torch.cuda.synchronize(device)

        # 本 rank 的样本数 → 所有 rank 求和
        total_samples_tensor = torch.tensor([total_samples_local], dtype=torch.float64, device=device)
        if world_size > 1:
            dist.all_reduce(total_samples_tensor, op=dist.ReduceOp.SUM)
        total_samples = int(total_samples_tensor.item())

        epoch_time = time.time() - epoch_start
        throughput = total_samples / epoch_time if epoch_time > 0 else 0.0
        max_mem_gb = torch.cuda.max_memory_allocated(device=device) / 1e9

        if rank == 0:
            print(
                f"[Epoch {epoch + 1}] "
                f"time={epoch_time:.3f}s | throught={throughput:.2f} samples/s | "
                f"GPU_mem={max_mem_gb:.2f}GB "
            )

            records.append(
                {
                    "epoch": epoch + 1,
                    "time(s)": epoch_time,
                    "throughput(samples/s)": throughput,
                    "GPU_mem(GB)": max_mem_gb,
                }
            )

        if world_size > 1:
            dist.barrier()

    if rank == 0:
        df = pd.DataFrame(records)
        cols = [
            "epoch",
            "time(s)",
            "throughput(samples/s)",
            "GPU_mem(GB)",
        ]
        df = df[cols]
        Path(excel_output).parent.mkdir(parents=True, exist_ok=True)
        df.to_excel(excel_output, index=False)
        print(f"[rank0] forward-only metrics saved → {excel_output}")


# ==============================
# 4. 主入口
# ==============================
def main():
    # ------------------
    # 路径配置
    # ------------------
    weights_path = "/root/computeEvaltool/train/B1/yolov10s.pt"
    weights_url = "https://github.com/THU-MIG/yolov10/releases/download/v1.1/yolov10s.pt"

    results_dir = "/root/computeEvaltool/train/B1/results"
    excel_output = f"{results_dir}/training_yolov10-s_1.xlsx"
    os.makedirs(results_dir, exist_ok=True)

    # ------------------
    # 初始化分布式
    # ------------------
    is_distributed, rank, local_rank, world_size = init_distributed()
    if rank == 0:
        print(f"[INIT] world_size={world_size}")
    print(f"[rank {rank}] local_rank={local_rank}")

    # ------------------
    # 下载权重（仅 rank0），然后 barrier
    # ------------------
    if not os.path.exists(weights_path):
        if rank == 0:
            print(f"[rank0] downloading weights from {weights_url} -> {weights_path}")
            urllib.request.urlretrieve(weights_url, weights_path)
        if is_distributed:
            dist.barrier()

    device = torch.device(f"cuda:{local_rank}")

    # ------------------
    # 加载 YOLOv10-S 模型（推理模型即可）
    # ------------------
    if rank == 0:
        print(f"[rank0] loading YOLOv10-S from {weights_path}")
    yolo = YOLO(weights_path)
    yolo.overrides["pretrained"] = False
    yolo.overrides["val"] = False

    model = yolo.model.to(device)
    model.eval()  # 用推理模式即可

    # 注意：这里不再使用 DDP，也不做 backward，只做多进程前向。
    # torchrun 只是帮我们起 16 个进程，每个进程负责一个 GPU。

    # ------------------
    # Dataloader & 配置
    # ------------------
    global_batch_size = 128           # 全局 batch，总共 16 卡
    batch_per_gpu = max(global_batch_size // world_size, 1)

    img_size = 640
    num_samples = 500000                # 单 epoch 的样本总数（全局）
    num_epochs = 5

    if rank == 0:
        print(
            f"[CONFIG] global_batch_size={global_batch_size}, "
            f"batch_per_gpu={batch_per_gpu}, "
            f"epochs={num_epochs}, "
            f"num_samples={num_samples}"
        )

    dataloader = build_dataloader(
        batch_per_gpu=batch_per_gpu,
        world_size=world_size,
        rank=rank,
        num_samples=num_samples // world_size + 1,  # 各 rank 加起来约 num_samples
        img_size=img_size,
    )

    # ------------------
    # 前向-only Benchmark & 记录指标
    # ------------------
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