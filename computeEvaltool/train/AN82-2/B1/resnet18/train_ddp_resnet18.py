#!/usr/bin/env python3
# train_ddp_resnet18.py
import os
import argparse
import time
from typing import List

import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models

from openpyxl import Workbook
import csv

# -------------------------
# Utilities
# -------------------------
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=90)
    parser.add_argument('--batch-size', type=int, default=128, help='per-GPU batch size')
    parser.add_argument('--lr', type=float, default=0.05, help='base learning rate (per-GPU)')
    parser.add_argument('--momentum', type=float, default=0.9)
    parser.add_argument('--weight-decay', type=float, default=5e-4)
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--save-dir', type=str, default='/root/computeEvaltool/train/B1/resnet18/checkpoints')
    parser.add_argument('--data-root', type=str, default='/root/computeEvaltool/train/B1/resnet18')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--warmup-epochs', type=int, default=5)
    parser.add_argument('--label-smoothing', type=float, default=0.0)
    parser.add_argument('--clip-grad', type=float, default=5.0)
    return parser.parse_args()


def init_distributed_mode():
    local_rank = int(os.environ.get('LOCAL_RANK', '0'))
    world_size = int(os.environ.get('WORLD_SIZE', '1'))
    rank = int(os.environ.get('RANK', '0'))

    dist.init_process_group(backend='nccl', init_method='env://')
    torch.cuda.set_device(local_rank)
    return local_rank, rank, world_size


def get_dataloaders(data_root, batch_size, workers):
    normalize = transforms.Normalize((0.4914, 0.4822, 0.4465),
                                     (0.2023, 0.1994, 0.2010))
    train_transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalize,
    ])
    test_transform = transforms.Compose([
        transforms.ToTensor(),
        normalize,
    ])

    train_dataset = datasets.CIFAR10(root=data_root, train=True, download=False, transform=train_transform)
    val_dataset = datasets.CIFAR10(root=data_root, train=False, download=False, transform=test_transform)
    train_sampler = DistributedSampler(train_dataset)
    val_sampler = DistributedSampler(val_dataset, shuffle=False)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, sampler=train_sampler,
                              num_workers=workers, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=256, sampler=val_sampler,
                            num_workers=workers, pin_memory=True)
    return train_loader, val_loader, train_sampler, val_sampler


# -------------------------
# Main
# -------------------------
def main():
    args = parse_args()
    local_rank, rank, world_size = init_distributed_mode()

    torch.manual_seed(args.seed + rank)
    torch.backends.cudnn.benchmark = True

    model = models.resnet18(num_classes=10).cuda()
    model = DDP(model, device_ids=[local_rank], output_device=local_rank, broadcast_buffers=False)

    train_loader, val_loader, train_sampler, val_sampler = get_dataloaders(args.data_root, args.batch_size, args.workers)

    optimizer = optim.SGD(model.parameters(), lr=args.lr,
                          momentum=args.momentum, weight_decay=args.weight_decay)

    if args.label_smoothing > 0.0:
        try:
            criterion = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing).cuda()
        except:
            criterion = nn.CrossEntropyLoss().cuda()
    else:
        criterion = nn.CrossEntropyLoss().cuda()

    scaler = torch.cuda.amp.GradScaler(enabled=True)

    effective_max_lr = args.lr * world_size
    warmup_epochs = args.warmup_epochs
    total_epochs = args.epochs

    cosine_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max(1, total_epochs - warmup_epochs), eta_min=1e-6
    )

    if rank == 0 and not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir, exist_ok=True)

    records = []

    for epoch in range(total_epochs):
        train_sampler.set_epoch(epoch)

        if epoch < warmup_epochs:
            lr = effective_max_lr * float(epoch + 1) / warmup_epochs
            for pg in optimizer.param_groups:
                pg["lr"] = lr
        else:
            if epoch == warmup_epochs:
                for pg in optimizer.param_groups:
                    pg["lr"] = effective_max_lr
            cosine_scheduler.step(epoch - warmup_epochs)
            lr = optimizer.param_groups[0]["lr"]

        model.train()
        epoch_start = time.time()

        train_loss_local = 0
        train_correct_local = 0
        train_total_local = 0

        for images, targets in train_loader:
            images = images.cuda(non_blocking=True)
            targets = targets.cuda(non_blocking=True)

            with torch.cuda.amp.autocast():
                outputs = model(images)
                loss = criterion(outputs, targets)

            optimizer.zero_grad()
            scaler.scale(loss).backward()

            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=args.clip_grad)

            scaler.step(optimizer)
            scaler.update()

            bsz = images.size(0)
            train_loss_local += loss.item() * bsz
            train_correct_local += (outputs.argmax(1) == targets).sum().item()
            train_total_local += bsz

        # reduce
        t_loss = torch.tensor(train_loss_local, device='cuda')
        t_corr = torch.tensor(train_correct_local, device='cuda')
        t_total = torch.tensor(train_total_local, device='cuda')

        dist.all_reduce(t_loss); dist.all_reduce(t_corr); dist.all_reduce(t_total)

        train_loss = t_loss.item() / t_total.item()
        train_acc = (t_corr.item() / t_total.item()) * 100

        # validation
        model.eval()
        val_loss_local = 0
        val_corr_local = 0
        val_total_local = 0

        with torch.no_grad():
            for images, targets in val_loader:
                images = images.cuda(non_blocking=True)
                targets = targets.cuda(non_blocking=True)

                with torch.cuda.amp.autocast():
                    outputs = model(images)
                    loss = criterion(outputs, targets)

                bsz = images.size(0)
                val_loss_local += loss.item() * bsz
                val_corr_local += (outputs.argmax(1) == targets).sum().item()
                val_total_local += bsz

        tv_loss = torch.tensor(val_loss_local, device='cuda')
        tv_corr = torch.tensor(val_corr_local, device='cuda')
        tv_total = torch.tensor(val_total_local, device='cuda')

        dist.all_reduce(tv_loss); dist.all_reduce(tv_corr); dist.all_reduce(tv_total)

        val_loss = tv_loss.item() / tv_total.item()
        val_acc = (tv_corr.item() / tv_total.item()) * 100

        epoch_time = time.time() - epoch_start
        throughput = t_total.item() / epoch_time

        if rank == 0:
            print(f"Epoch {epoch} | time={epoch_time:.2f}s | throughput={throughput:.2f} | "
                  f"loss={val_loss:.2f} | acc={val_acc:.2f}%")

            # --------------------------
            # ✔ 记录格式仅保存 5 个字段
            # --------------------------
            records.append([
                epoch,
                round(epoch_time, 2),
                round(throughput, 2),
                round(val_loss, 2),
                round(val_acc, 2),
            ])

            torch.save({
                "epoch": epoch,
                "model": model.module.state_dict(),
                "opt": optimizer.state_dict(),
                "lr": lr,
            }, os.path.join(args.save_dir, f"epoch_{epoch}.pth"))

    # -------- Save to Excel/CSV --------
    if rank == 0:
        results_dir = "/root/computeEvaltool/train/B1/results"
        os.makedirs(results_dir, exist_ok=True)

        xlsx_path = os.path.join(results_dir, "training_resnet18_1.xlsx")
        csv_path = os.path.join(results_dir, "training_resnet18_1.csv")

        # Excel
        wb = Workbook()
        ws = wb.active
        ws.title = "Metrics"

        header = ["epoch", "time(s)", "throughput(samples/s)", "loss", "acc(%)"]
        ws.append(header)

        for r in records:
            ws.append(r)

        wb.save(xlsx_path)

        # CSV
        with open(csv_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(header)
            writer.writerows(records)

        print(f"Saved {xlsx_path} and {csv_path}")

    dist.destroy_process_group()


if __name__ == '__main__':
    main()
