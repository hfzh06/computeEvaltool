import os
import time
import argparse
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms

import deepspeed
import pandas as pd
from pathlib import Path
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

from transformers import AutoImageProcessor, AutoModelForImageClassification


def print_rank0(msg: str):
    if torch.distributed.get_rank() == 0:
        print(msg)


def main():

    # ====== 命令行参数（必须包含 local_rank）======
    parser = argparse.ArgumentParser()
    parser.add_argument("--dynamic_bs", type=int, default=16)
    parser.add_argument("--local_rank", type=int, default=0)
    args = parser.parse_args()
    dynamic_bs = args.dynamic_bs


    deepspeed.init_distributed()

    world_size = torch.distributed.get_world_size()
    rank = torch.distributed.get_rank()
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)

    print_rank0(f"World Size={world_size}, Rank={rank}, Local Rank={local_rank}")
    print_rank0(f"Dynamic Batch Size per GPU: {dynamic_bs}")

    # -----------------------------
    # 预处理器与模型
    # -----------------------------
    model_dir = "/root/computeEvaltool/train/B2/vit-large/vit-large"
    print_rank0(f"Loading ViT-Large from: {model_dir}")

    processor = AutoImageProcessor.from_pretrained(model_dir)

    model = AutoModelForImageClassification.from_pretrained(
        model_dir,
        num_labels=10,
        ignore_mismatched_sizes=True
    )

    criterion = nn.CrossEntropyLoss()

    # -----------------------------
    # 数据集
    # -----------------------------
    size = 224
    if hasattr(processor, "size"):
        s = processor.size
        if isinstance(s, dict):
            size = s.get("shortest_edge", s.get("height", 224))
        elif isinstance(s, int):
            size = s

    mean = getattr(processor, "image_mean", [0.5, 0.5, 0.5])
    std = getattr(processor, "image_std", [0.5, 0.5, 0.5])

    transform_train = transforms.Compose([
        transforms.Resize((size, size)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std),
    ])

    train_set = torchvision.datasets.CIFAR10(
        root="/data/cifar10",
        train=True,
        download=False,
        transform=transform_train
    )

    train_sampler = DistributedSampler(train_set, num_replicas=world_size, rank=rank, shuffle=True)

    train_loader = DataLoader(
        train_set,
        batch_size=dynamic_bs,
        sampler=train_sampler,
        num_workers=4,
        pin_memory=True
    )

    # -----------------------------
    # DeepSpeed 初始化
    # -----------------------------
    ds_config = "/root/computeEvaltool/train/B2/vit-large/ds_config.json"
    if rank == 0:
        try:
            print(open(ds_config).read())
        except:
            print_rank0("Cannot read ds_config.json")

    model_engine, optimizer, _, _ = deepspeed.initialize(
        model=model,
        model_parameters=model.parameters(),
        config=ds_config,
        config_params={"train_micro_batch_size_per_gpu": dynamic_bs,
                       "train_batch_size": dynamic_bs * 8},

    )

    # -----------------------------
    # 训练循环
    # -----------------------------
    num_epochs = 4
    global_batch_size = dynamic_bs * world_size

    epoch_records = []
    sweep_records = []

    print_rank0(f"Training for {num_epochs} epochs...")

    for epoch in range(1, num_epochs + 1):
        train_sampler.set_epoch(epoch)
        epoch_start = time.time()

        model_engine.train()

        total_loss = 0.0
        total_correct = 0
        total_seen = 0

        for imgs, labels in train_loader:
            imgs = imgs.to(local_rank)
            labels = labels.to(local_rank)

            outputs = model_engine(imgs)
            logits = outputs.logits if hasattr(outputs, "logits") else outputs

            loss = criterion(logits, labels)
            model_engine.backward(loss)
            model_engine.step()

            total_loss += loss.item()
            preds = logits.argmax(dim=1)
            total_correct += (preds == labels).sum().item()
            total_seen += labels.size(0)

        # 计算指标
        epoch_time = time.time() - epoch_start
        num_batches = len(train_loader)
        num_samples = num_batches * global_batch_size
        throughput = num_samples / epoch_time

        avg_loss_tensor = torch.tensor(total_loss / num_batches, device=local_rank)
        acc_tensor = torch.tensor(total_correct / total_seen, device=local_rank)

        torch.distributed.all_reduce(avg_loss_tensor)
        torch.distributed.all_reduce(acc_tensor)

        avg_loss = (avg_loss_tensor / world_size).item()
        accuracy = (acc_tensor / world_size).item()

        print_rank0(f"[Epoch {epoch}] time={epoch_time:.2f}s throughput={throughput:.2f} "
                    f"loss={avg_loss:.2f} acc={accuracy:.2f}")

        # rank0 保留记录
        if rank == 0:
            epoch_records.append({
                "epoch": epoch,
                "epoch_time": round(epoch_time, 2),
                "throughput": round(throughput, 2),
                "avg_loss": round(avg_loss, 2),
                "accuracy": round(accuracy, 2),
            })

            # sweep 最后三个 epoch
            if epoch >= num_epochs - 2:
                sweep_records.append({
                    "batch_size": dynamic_bs,
                    "epoch": epoch,
                    "time(s)": round(epoch_time, 2),
                    "throughput(samples/s)": round(throughput, 2),
                    "loss": round(avg_loss, 2),
                    "acc(%)": round(accuracy * 100, 2),
                })

    # -----------------------------
    # 保存文件
    # -----------------------------
    if rank == 0:
        result_dir = "/root/computeEvaltool/train/B2/results"
        Path(result_dir).mkdir(parents=True, exist_ok=True)

        # 1. 原始日志
        df = pd.DataFrame(epoch_records)
        df.to_excel(f"{result_dir}/training_vitlarge_cifar10.xlsx", index=False)

        # 2. 格式化日志
        df2 = pd.DataFrame([{
            "Epoch": rec["epoch"],
            "time(s)": rec["epoch_time"],
            "throughput(samples/s)": rec["throughput"],
            "loss": rec["avg_loss"],
            "acc(%)": round(rec["accuracy"] * 100, 2)
        } for rec in epoch_records])
        df2.to_excel(f"{result_dir}/training_vit-large_temp.xlsx", index=False)

        # 3. sweep 追加写入
        sweep_file = f"{result_dir}/training_vit-large_2.xlsx"
        df3 = pd.DataFrame(sweep_records)

        if not os.path.exists(sweep_file):
            df3.to_excel(sweep_file, index=False)
            print_rank0(f"Created sweep log → {sweep_file}")
        else:
            old_df = pd.read_excel(sweep_file)
            new_df = pd.concat([old_df, df3], ignore_index=True)
            new_df.to_excel(sweep_file, index=False)
            print_rank0(f"Appended sweep logs → {sweep_file}")

        print_rank0("Saved all Excel logs.")


if __name__ == "__main__":
    main()
