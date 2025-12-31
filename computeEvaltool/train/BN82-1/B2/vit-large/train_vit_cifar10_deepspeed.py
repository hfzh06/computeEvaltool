import os
import time
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

    # 初始化分布式
    deepspeed.init_distributed()

    world_size = torch.distributed.get_world_size()
    rank = torch.distributed.get_rank()
    local_rank = int(os.environ["LOCAL_RANK"])

    torch.cuda.set_device(local_rank)
    print_rank0(f"World Size={world_size}, Rank={rank}, Local Rank={local_rank}")

    # ============================
    # 模型与预处理器
    # ============================
    model_dir = "/root/computeEvaltool/train/B2/vit-large/vit-large"
    print_rank0(f"Loading ViT-Large from: {model_dir}")

    processor = AutoImageProcessor.from_pretrained(model_dir)

    model = AutoModelForImageClassification.from_pretrained(
        model_dir,
        num_labels=10,
        ignore_mismatched_sizes=True
    )

    criterion = nn.CrossEntropyLoss()

    # ============================
    # CIFAR10 DATASET
    # ============================
    size = 224
    if hasattr(processor, "size"):
        s = processor.size
        if isinstance(s, dict):
            if "shortest_edge" in s:
                size = s["shortest_edge"]
            elif "height" in s and "width" in s:
                size = s["height"]
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

    train_sampler = DistributedSampler(
        train_set,
        num_replicas=world_size,
        rank=rank,
        shuffle=True
    )

    train_loader = DataLoader(
        train_set,
        batch_size=16,
        sampler=train_sampler,
        num_workers=2,
        pin_memory=True
    )

    # ============================
    # DeepSpeed Engine
    # ============================
    ds_config = "/root/computeEvaltool/train/B2/vit-large/ds_config.json"
    print_rank0(f"Using DeepSpeed config: {ds_config}")

    if rank == 0:
        try:
            print_rank0(open(ds_config).read())
        except Exception as e:
            print_rank0(f"Failed to read ds_config: {e}")

    model_engine, optimizer, _, _ = deepspeed.initialize(
        model=model,
        model_parameters=model.parameters(),
        config=ds_config,
    )

    # ============================
    # METRICS
    # ============================
    epoch_records = []
    num_epochs = 3
    global_batch_size = train_loader.batch_size * world_size

    print_rank0(f"Training ViT-Large for {num_epochs} epochs...")

    # ============================
    # TRAINING LOOP
    # ============================
    for epoch in range(1, num_epochs + 1):

        train_sampler.set_epoch(epoch)
        epoch_start = time.time()

        model_engine.train()

        total_loss = 0.0
        total_correct = 0
        total_seen = 0

        for batch_idx, (imgs, labels) in enumerate(train_loader):
            imgs = imgs.to(local_rank, non_blocking=True)
            labels = labels.to(local_rank, non_blocking=True)

            outputs = model_engine(imgs)
            logits = outputs.logits if hasattr(outputs, "logits") else outputs

            loss = criterion(logits, labels)

            model_engine.backward(loss)
            model_engine.step()

            total_loss += loss.item()
            preds = logits.argmax(dim=1)
            total_correct += (preds == labels).sum().item()
            total_seen += labels.size(0)

        epoch_time = time.time() - epoch_start
        num_batches = len(train_loader)
        num_samples = num_batches * global_batch_size
        throughput = num_samples / epoch_time

        avg_loss_tensor = torch.tensor(total_loss / num_batches, device=local_rank)
        acc_tensor = torch.tensor(total_correct / total_seen, device=local_rank)

        torch.distributed.all_reduce(avg_loss_tensor, op=torch.distributed.ReduceOp.SUM)
        torch.distributed.all_reduce(acc_tensor, op=torch.distributed.ReduceOp.SUM)

        avg_loss = (avg_loss_tensor / world_size).item()
        accuracy = (acc_tensor / world_size).item()

        print_rank0(
            f"[Epoch {epoch}] "
            f"time={epoch_time:.2f}s "
            f"throughput={throughput:.2f} samples/s "
            f"loss={avg_loss:.2f} acc={accuracy:.2f}"
        )

        if rank == 0:
            epoch_records.append({
                "epoch": epoch,
                "epoch_time": round(epoch_time, 2),
                "throughput": round(throughput, 2),
                "avg_loss": round(avg_loss, 2),
                "accuracy": round(accuracy, 2),
            })

    # ============================
    # SAVE XLSX (rank 0 only)
    # ============================
    if rank == 0:
        results_dir = "/root/computeEvaltool/train/B2/results"
        Path(results_dir).mkdir(parents=True, exist_ok=True)

        # 原始格式
        out_file = f"{results_dir}/training_vitlarge_cifar10.xlsx"
        df = pd.DataFrame(epoch_records)
        df.to_excel(out_file, index=False)
        print_rank0(f"Saved training logs → {out_file}")

        # 新增格式：所有字段两位小数
        formatted_records = []
        for rec in epoch_records:
            formatted_records.append({
                "Epoch": rec["epoch"],
                "time(s)": round(rec["epoch_time"], 2),
                "throughput(samples/s)": round(rec["throughput"], 2),
                "loss": round(rec["avg_loss"], 2),
                "acc(%)": round(rec["accuracy"] * 100, 2),
            })

        df2 = pd.DataFrame(formatted_records)
        out_file2 = f"{results_dir}/training_vit-large_1.xlsx"
        df2.to_excel(out_file2, index=False)
        print_rank0(f"Saved formatted training table → {out_file2}")

    print_rank0("Training finished.")


if __name__ == "__main__":
    main()
