#!/usr/bin/env python3
import os
import sys
import time
import urllib.request
from pathlib import Path

import torch
import torch.distributed as dist
import pandas as pd
from ultralytics import YOLO


# ================================
# 初始化 DDP
# ================================
def init_distributed():

    is_distributed = "RANK" in os.environ and "WORLD_SIZE" in os.environ
    # 1. 直接从环境变量获取 DDP 信息 (由 torchrun 注入)
    
    if is_distributed:

        rank = int(os.environ.get("RANK", 0))
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        world_size = int(os.environ.get("WORLD_SIZE", 1))

        num_gpus = torch.cuda.device_count()

        if local_rank >= num_gpus:
            local_rank = rank % num_gpus
            raise ValueError(f"Local rank {local_rank} exceeds available GPUs {num_gpus}")

        torch.cuda.set_device(local_rank)
        dist.init_process_group(backend="nccl", init_method="env://")
        
        return True, rank, local_rank, world_size
    else:
        local_rank = 0
        torch.cuda.set_device(local_rank)
        return False, 0, local_rank, 1

# ================================
#  Epoch Metrics Callback
# ================================
class EpochMetricsCallback:
    def __init__(self, batch_size_per_gpu, world_size, excel_file, rank):
        self.batch_size_per_gpu = batch_size_per_gpu
        self.world_size = world_size
        self.excel_file = excel_file
        self.rank = rank

        # 只在主节点初始化 Excel
        if self.rank == 0 and not os.path.exists(self.excel_file):
            df = pd.DataFrame(columns=[
                "epoch", "time(s)", "throughput(samples/s)",
                "box_loss", "cls_loss", "dfl_loss"
            ])
            df.to_excel(self.excel_file, index=False)

    def on_train_epoch_start(self, trainer):
        self.epoch_start = time.time()

    def on_train_epoch_end(self, trainer):
        epoch = trainer.epoch + 1
        t = round(time.time() - self.epoch_start, 2)

        num_batches = len(trainer.train_loader)
        num_samples = num_batches * self.batch_size_per_gpu * self.world_size
        throughput = round(num_samples / t, 2)

        box_loss = round(float(trainer.loss_items[0]), 2)
        cls_loss = round(float(trainer.loss_items[1]), 2)
        dfl_loss = round(float(trainer.loss_items[2]), 2)

        if self.rank == 0:
            print(f"[Epoch {epoch}] "
                  f"time={t:.2f}s, throughput={throughput:.2f}, "
                  f"box={box_loss:.2f}, cls={cls_loss:.2f}, dfl={dfl_loss:.2f}")

            new_row = pd.DataFrame([{
                "epoch": epoch,
                "time(s)": t,
                "throughput(samples/s)": throughput,
                "box_loss": box_loss,
                "cls_loss": cls_loss,
                "dfl_loss": dfl_loss
            }])

            with pd.ExcelWriter(self.excel_file, engine="openpyxl",
                                mode="a", if_sheet_exists="overlay") as writer:
                start_row = writer.sheets["Sheet1"].max_row
                new_row.to_excel(writer, index=False, header=False, startrow=start_row)


# ================================
#  Main Training
# ================================
def main():
    # ⚠️ 必须移除任何手动设置 CUDA_VISIBLE_DEVICES 的代码
    # 让 torchrun 自动管理可见性，我们只负责选设备
    os.environ["YOLO_DISABLE_GPU_AUTO_ASSIGN"] = "1"

    base_dir = "/root/computeEvaltool/train/B1/yolov10"
    repo_path = os.path.join(base_dir, "yolov10-main")
    weights_path = os.path.join("/root/computeEvaltool/train/B1/yolov10/yolov10-main/ultralytics/cfg/models/v10/yolov10s.yaml")
    weights_url = "https://github.com/THU-MIG/yolov10/releases/download/v1.1/yolov10s.pt"

    results_dir = "/root/computeEvaltool/train/B1/results"
    os.makedirs(results_dir, exist_ok=True)
    excel_output = os.path.join(results_dir, "training_yolov10-s_1.xlsx")

    data_cfg = os.path.join(base_dir, "coco128.yaml")

    # 初始化
    is_distributed, rank, local_rank, world_size = init_distributed()

    if repo_path not in sys.path:
        sys.path.append(repo_path)

    # 权重下载同步
    if not os.path.exists(weights_path):
        if rank == 0:
            print(f"Downloading {weights_url} → {weights_path}")
            urllib.request.urlretrieve(weights_url, weights_path)
        if is_distributed:
            dist.barrier()

    # Load model
    model = YOLO(weights_path)
    model.overrides["pretrained"] = False

    batch_total = 128
    batch_per_gpu = batch_total // world_size

    metrics_cb = EpochMetricsCallback(
        batch_size_per_gpu=batch_per_gpu,
        world_size=world_size,
        excel_file=excel_output,
        rank=rank,
    )

    model.add_callback("on_train_epoch_start", metrics_cb.on_train_epoch_start)
    model.add_callback("on_train_epoch_end", metrics_cb.on_train_epoch_end)

    # 构造明确的设备字符串，例如 "cuda:0", "cuda:7"
    # 这比传 int 更安全，能防止 Ultralytics 内部混淆
    device_str = f"cuda:{local_rank}"

    if rank == 0:
        print(f"Master Node started. Using device: {device_str}")
    
    # =======================
    # 1）多机多卡训练
    # =======================
    try:
        model.train(
            data=data_cfg,
            epochs=50,
            batch=batch_total,
            imgsz=640,
            workers=8,
            project=base_dir,
            name="yolov10s_ddp",
            exist_ok=True,
            verbose=(rank == 0),
            pretrained=False,
            amp=True,
            save=False,
            val=False,
            plots=False,
            # ★★★ 关键修正 ★★★ 
            # 显式指定本地设备，格式为 "cuda:X"
            device=device_str
        )
    except Exception as e:
        print(f"[rank {rank}] Training exception: {e}")
        import traceback
        traceback.print_exc()

    # =======================
    # 2）训练结束后，只在 rank0 单机验证
    # =======================
    if is_distributed:
        dist.destroy_process_group()

    if rank == 0:
        print("\n======== 开始单机验证（仅 rank0） ========")
        # 清理环境变量
        for k in ["RANK", "WORLD_SIZE", "LOCAL_RANK"]:
            os.environ.pop(k, None)
        
        # 强制指定用第一张卡做验证
        torch.cuda.set_device(0)
        
        model.val(
            data=data_cfg,
            imgsz=640,
            batch=32,
            workers=8,
            plots=False,
            device="cuda:0" 
        )
        print("======== 验证完成 ========")


if __name__ == "__main__":
    main()