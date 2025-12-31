#!/usr/bin/env python3
import os
import sys
import time
import argparse
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
    rank = int(os.environ.get("RANK", 0))
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))

    is_distributed = world_size > 1
    if is_distributed:
        dist.init_process_group(backend="nccl", init_method="env://")
        torch.cuda.set_device(local_rank)

    return is_distributed, rank, local_rank, world_size


# ================================
#  Epoch Metrics Callback (store all, write only last 3)
# ================================
class EpochMetricsCallback:
    def __init__(self, batch_size_per_gpu, world_size, rank):
        self.batch_size_per_gpu = batch_size_per_gpu
        self.world_size = world_size
        self.rank = rank
        self.epoch_start = None
        self.buffer = []   # 存所有 epoch ，最后三行才写入 Excel

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

        record = {
            "epoch": epoch,
            "time(s)": t,
            "throughput(samples/s)": throughput,
            "box_loss": box_loss,
            "cls_loss": cls_loss,
            "dfl_loss": dfl_loss
        }

        self.buffer.append(record)

        if self.rank == 0:
            print(f"[Epoch {epoch}]  time={t:.2f}s  thr={throughput:.2f}  "
                  f"box={box_loss:.2f} cls={cls_loss:.2f} dfl={dfl_loss:.2f}")


# ================================
#  Main Training
# ================================
def main():
    import os
    os.environ["NCCL_SOCKET_IFNAME"] = "enp46s0np0"
    os.environ["NCCL_IB_DISABLE"] = "0"
    os.environ["MASTER_ADDR"] = "10.1.73.17"
    os.environ["MASTER_PORT"] = "29688"

    parser = argparse.ArgumentParser()
    parser.add_argument("--bs", type=int, required=True)  # 当前 batch size
    args = parser.parse_args()

    # yolov10 根目录
    yolo_dir = "/root/computeEvaltool/train/B1/yolov10"

    # Excel 保存到 B1/results
    excel_path = "/root/computeEvaltool/train/B1/results/training_yolov10-s_2.xlsx"

    repo_path = f"{yolo_dir}/yolov10-main"
    weights_path = f"{yolo_dir}/yolov10s.pt"

    data_cfg = f"{yolo_dir}/coco128.yaml"

    # 初始化 DDP
    is_distributed, rank, local_rank, world_size = init_distributed()

    if repo_path not in sys.path:
        sys.path.append(repo_path)

    torch.cuda.set_device(local_rank)

    model = YOLO(weights_path)
    model.overrides["pretrained"] = False

    # per-GPU batch
    batch_total = args.bs
    batch_per_gpu = batch_total // world_size

    # 回调
    cb = EpochMetricsCallback(batch_per_gpu, world_size, rank)
    model.add_callback("on_train_epoch_start", cb.on_train_epoch_start)
    model.add_callback("on_train_epoch_end", cb.on_train_epoch_end)

    # 训练
    model.train(
        data=data_cfg,
        epochs=10,          # 你可改成任意 epoch
        batch=batch_total,
        imgsz=640,
        workers=8,
        project=yolo_dir,
        name=f"bs_{args.bs}",
        exist_ok=True,
        verbose=(rank == 0),
        save=False,
        val=False,
        plots=False,
    )

    # ============================
    # 写入 Excel：只写最后三行
    # ============================
    if rank == 0:
        last3 = cb.buffer[-3:] if len(cb.buffer) >= 3 else cb.buffer

        df = pd.DataFrame(last3)
        df["batch_size"] = args.bs

        # 如果文件不存在，写入表头
        if not os.path.exists(excel_path):
            df.to_excel(excel_path, index=False)
        else:
            with pd.ExcelWriter(excel_path, engine="openpyxl", mode="a",
                                if_sheet_exists="overlay") as writer:
                start_row = writer.sheets['Sheet1'].max_row
                df.to_excel(writer, index=False, header=False, startrow=start_row)

        print(f"已写入 batch_size={args.bs} 的最后三行结果 → {excel_path}")

    os._exit(0)


if __name__ == "__main__":
    main()
