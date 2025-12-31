#!/usr/bin/env python3
import os
import sys
import time
import urllib.request
from pathlib import Path

import torch
import torch.distributed as dist
from ultralytics import YOLO
import pandas as pd


def init_distributed():
    """Initialize DDP environment from torchrun env variables."""
    rank = int(os.environ.get("RANK", 0))
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))

    is_distributed = world_size > 1
    if is_distributed:
        dist.init_process_group(backend="nccl", init_method="env://")
        torch.cuda.set_device(local_rank)
    else:
        local_rank = 0

    return is_distributed, rank, local_rank, world_size


class EpochMetricsCallback:
    """
    收集每个 epoch 的指标，并在 rank0 打印 & 训练结束后写入 Excel.

    记录字段：
    - epoch
    - time(s)
    - throughput(samples/s)
    - loss = box_loss + cls_loss + dfl_loss
    - acc(%) = NaN（占位）
    - GPU_mem(GB)
    - box_loss, cls_loss, dfl_loss
    - instances（近似为一个 epoch 的样本数）
    - img_size
    """

    def __init__(self, batch_size_per_gpu, world_size, rank):
        self.batch_size_per_gpu = batch_size_per_gpu
        self.world_size = world_size
        self.rank = rank

        self.epoch_start = None
        self.records = []

        # 用于累积样本数（approx），在 epoch_start 时重置
        self.seen_samples = 0

    def on_train_epoch_start(self, trainer):
        self.epoch_start = time.time()
        self.seen_samples = 0

    def on_train_batch_end(self, trainer):
        """每个 batch 结束时调用，用于估计 epoch 内总样本数。"""
        # 一个 batch 的样本数（单卡）
        bs = getattr(trainer, "batch_size", None)
        if bs is None:
            # 如果 trainer 没有 batch_size 属性，就退化为 train_loader 的 batch_size
            try:
                bs = trainer.train_loader.batch_size
            except Exception:
                bs = self.batch_size_per_gpu

        # world_size 张卡
        self.seen_samples += bs * self.world_size

    def on_train_epoch_end(self, trainer):
        epoch = trainer.epoch + 1
        epoch_time = time.time() - self.epoch_start

        # throughput: epoch 级别的吞吐
        total_samples = self.seen_samples if self.seen_samples > 0 else (
            len(trainer.train_loader) * self.batch_size_per_gpu * self.world_size
        )
        throughput = total_samples / epoch_time if epoch_time > 0 else 0.0

        # loss components（ultralytics 提供）
        # trainer.loss_items 通常是 [box_loss, cls_loss, dfl_loss]
        box_loss = cls_loss = dfl_loss = 0.0
        loss_items = getattr(trainer, "loss_items", None)
        if loss_items is not None and len(loss_items) >= 3:
            box_loss = float(loss_items[0])
            cls_loss = float(loss_items[1])
            dfl_loss = float(loss_items[2])

        total_loss = box_loss + cls_loss + dfl_loss

        # GPU 显存使用（GB）
        gpu_mem = torch.cuda.max_memory_allocated() / 1e9

        # Instances：没有直接提供，使用估计的 total_samples
        instances = total_samples

        # img size
        img_size = getattr(trainer.args, "imgsz", None)

        # 占位的 acc(%)，暂无真实计算
        acc_pct = float("nan")

        # rank0 打印一行
        if self.rank == 0:
            print(
                f"[Epoch {epoch}] "
                f"time={epoch_time:.3f}s | thr={throughput:.2f} samples/s | "
                f"loss={total_loss:.4f} | acc={acc_pct if acc_pct == acc_pct else 0.0:.2f}% | "
                f"GPU_mem={gpu_mem:.2f}GB | "
                f"box={box_loss:.4f} cls={cls_loss:.4f} dfl={dfl_loss:.4f} | "
                f"instances={instances} size={img_size}"
            )

        # 记录到列表（列顺序按你的需求）
        self.records.append({
            "epoch": epoch,
            "time(s)": epoch_time,
            "throughput(samples/s)": throughput,
            "loss": total_loss,
            "acc(%)": acc_pct,
            "GPU_mem(GB)": gpu_mem,
            "box_loss": box_loss,
            "cls_loss": cls_loss,
            "dfl_loss": dfl_loss,
            "instances": instances,
            "img_size": img_size,
        })


def main():
    # 路径配置（按你之前的设置）
    repo_path = "/root/computeEvaltool/train/B1/yolov10/yolov10-main"
    weights_path = "/root/computeEvaltool/train/B1/yolo11n.pt"
    weights_url = "https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11n.pt"

    project_dir = "/root/computeEvaltool/train/B1"
    results_dir = "/root/computeEvaltool/train/B1/results"
    excel_output = f"{results_dir}/training_yolov10-s_1.xlsx"

    os.makedirs(results_dir, exist_ok=True)

    # DDP 初始化
    is_distributed, rank, local_rank, world_size = init_distributed()
    print(f"[rank {rank}] DDP={is_distributed}, local_rank={local_rank}, world_size={world_size}")

    # 保证 yolov10 仓库在 sys.path 中
    if repo_path not in sys.path:
        sys.path.append(repo_path)

    # 下载权重（仅 rank0），然后所有 rank barrier
    if not os.path.exists(weights_path):
        if rank == 0:
            print(f"[rank0] downloading weights from {weights_url} -> {weights_path}")
            urllib.request.urlretrieve(weights_url, weights_path)
        if is_distributed:
            dist.barrier()

    torch.cuda.set_device(local_rank)

    # 加载 YOLO 模型
    model = YOLO(weights_path)
    # 不使用预训练配置（只用权重）
    model.overrides["pretrained"] = False
    model.overrides["final_validation"] = False   # ← 必加
    model.overrides["val"] = False 

    # 每个 epoch 的指标回调
    total_batch_size = 128  # 总 batch（所有 GPU 的和）
    batch_per_gpu = total_batch_size // max(world_size, 1)

    metrics_callback = EpochMetricsCallback(batch_per_gpu, world_size, rank)

    # 注册回调
    model.add_callback("on_train_epoch_start", metrics_callback.on_train_epoch_start)
    model.add_callback("on_train_epoch_end", metrics_callback.on_train_epoch_end)
    model.add_callback("on_train_batch_end", metrics_callback.on_train_batch_end)

    # 数据集配置（你已有的 yolov10 仓库中的 coco128 配置）
    data_cfg = os.path.join(repo_path, "ultralytics/cfg/datasets/coco128.yaml")

    # 开始训练
    model.train(
        data=data_cfg,
        epochs=100,
        batch=total_batch_size,
        imgsz=640,
        workers=8,
        project=project_dir,
        name="yolov10_16gpu",
        exist_ok=True,
        verbose=(rank == 0),
        pretrained=False,
        amp=True,
        save=False,   # 不保存 best.pt / last.pt
        val=False,    # 不做 validation
        plots=False,  # 不生成图
    )

    # rank0 写 Excel
    if rank == 0:
        df = pd.DataFrame(metrics_callback.records)
        # 确保列顺序
        cols = [
            "epoch",
            "time(s)",
            "throughput(samples/s)",
            "loss",
            "acc(%)",
            "GPU_mem(GB)",
            "box_loss",
            "cls_loss",
            "dfl_loss",
            "instances",
            "img_size",
        ]
        df = df[cols]

        Path(excel_output).parent.mkdir(parents=True, exist_ok=True)
        df.to_excel(excel_output, index=False)
        print(f"[rank0] training metrics saved → {excel_output}")

    # 结束 DDP
    if is_distributed and dist.is_initialized():
        dist.destroy_process_group()

    sys.exit(0)


if __name__ == "__main__":
    main()

