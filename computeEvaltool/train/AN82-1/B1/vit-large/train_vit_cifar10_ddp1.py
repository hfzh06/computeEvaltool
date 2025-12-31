import os
import argparse
from pathlib import Path
import time
import importlib
import pandas as pd

import timm
import torch
import torch.nn as nn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
import torchvision
import torchvision.transforms as transforms


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-path", required=True)
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--img-size", type=int, default=224)
    parser.add_argument("--save-dir", default="./checkpoints")
    parser.add_argument("--resume", default="")
    parser.add_argument("--result-path", required=True)   # Excel 结果路径
    return parser.parse_args()


def get_vit_large(num_classes=10):
    return timm.create_model(
        "vit_large_patch16_224",
        pretrained=False,
        num_classes=num_classes
    )


def main():
    args = parse_args()

    # =======================
    # DDP Init
    # =======================
    rank = int(os.environ["RANK"])
    local_rank = int(os.environ["LOCAL_RANK"])
    world_size = int(os.environ["WORLD_SIZE"])

    torch.cuda.set_device(local_rank)
    dist.init_process_group(backend="nccl")

    # =======================
    # Dataset
    # =======================
    normalize = transforms.Normalize(
        mean=[0.4914, 0.4822, 0.4465],
        std=[0.2470, 0.2435, 0.2616]
    )

    train_tf = transforms.Compose([
        transforms.RandomResizedCrop(args.img_size),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalize,
    ])
    val_tf = transforms.Compose([
        transforms.Resize(int(args.img_size * 1.14)),
        transforms.CenterCrop(args.img_size),
        transforms.ToTensor(),
        normalize,
    ])

    Path(args.data_path).mkdir(parents=True, exist_ok=True)

    train_dataset = torchvision.datasets.CIFAR10(
        root=args.data_path, train=True, download=True, transform=train_tf
    )
    val_dataset = torchvision.datasets.CIFAR10(
        root=args.data_path, train=False, download=True, transform=val_tf
    )

    train_sampler = DistributedSampler(train_dataset, num_replicas=world_size, rank=rank)
    val_sampler = DistributedSampler(val_dataset, num_replicas=world_size, rank=rank, shuffle=False)

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        sampler=train_sampler,
        num_workers=args.num_workers,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        sampler=val_sampler,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    # =======================
    # Model
    # =======================
    model = get_vit_large(num_classes=10).cuda()
    model = DDP(model, device_ids=[local_rank])

    criterion = nn.CrossEntropyLoss().cuda()

    effective_bs = args.batch_size * world_size
    lr = args.lr * (effective_bs / 256)
    optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9)
    scaler = torch.cuda.amp.GradScaler()

    # =======================
    # Logs
    # =======================
    epoch_logs = []

    # =======================
    # Training Loop
    # =======================
    for epoch in range(args.epochs):

        train_sampler.set_epoch(epoch)
        model.train()

        t0 = time.time()
        total_samples = 0

        # ---------------------
        # Train One Epoch
        # ---------------------
        for images, labels in train_loader:
            images = images.cuda(non_blocking=True)
            labels = labels.cuda(non_blocking=True)

            optimizer.zero_grad()
            with torch.cuda.amp.autocast():
                out = model(images)
                loss = criterion(out, labels)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            total_samples += labels.size(0)

        # gather samples
        t_total = torch.tensor(total_samples, device="cuda")
        dist.all_reduce(t_total)

        # ---------------------
        # Validation
        # ---------------------
        model.eval()
        vloss = 0
        vcorrect = 0
        vtotal = 0

        with torch.no_grad():
            for images, labels in val_loader:
                images = images.cuda(non_blocking=True)
                labels = labels.cuda(non_blocking=True)

                with torch.cuda.amp.autocast():
                    out = model(images)
                    loss = criterion(out, labels)

                preds = out.argmax(1)
                vloss += loss.item() * labels.size(0)
                vcorrect += (preds == labels).sum().item()
                vtotal += labels.size(0)

        # sync validation metrics
        t_loss = torch.tensor(vloss, device="cuda")
        t_corr = torch.tensor(vcorrect, device="cuda")
        t_cnt = torch.tensor(vtotal, device="cuda")

        dist.all_reduce(t_loss)
        dist.all_reduce(t_corr)
        dist.all_reduce(t_cnt)

        val_loss = (t_loss / t_cnt).item()
        val_acc = (t_corr / t_cnt).item() * 100

        # ---------------------
        # Time & throughput
        # ---------------------
        epoch_time = time.time() - t0
        throughput = t_total.item() / epoch_time

        if rank == 0:
            print(
                f"[Epoch {epoch+1}] "
                f"time={epoch_time:.3f}s, thr={throughput:.1f}, "
                f"loss={val_loss:.3f}, acc={val_acc:.2f}%"
            )

            # -------- 保存格式 --------
            epoch_logs.append({
                "BatchSize": args.batch_size,
                "Epoch": epoch + 1,
                "Time(s)": round(epoch_time, 3),
                "Throughput(samples/s)": round(throughput, 2),
                "Loss": round(val_loss, 4),
                "Acc(%)": round(val_acc, 2)
            })

    # =======================
    # Save last three epochs (append mode)
    # =======================
    if rank == 0:
        Path(args.result_path).parent.mkdir(parents=True, exist_ok=True)

        new_df = pd.DataFrame(epoch_logs[-3:])  # 只要最后 3 行

        if Path(args.result_path).exists():
            old_df = pd.read_excel(args.result_path)
            final_df = pd.concat([old_df, new_df], ignore_index=True)
        else:
            final_df = new_df

        final_df.to_excel(args.result_path, index=False)
        print(f"\n[OK] 已写入文件: {args.result_path}\n")

    dist.destroy_process_group()


if __name__ == "__main__":
    main()
