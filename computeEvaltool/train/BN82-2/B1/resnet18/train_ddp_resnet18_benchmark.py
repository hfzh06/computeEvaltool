import os
import time
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader, DistributedSampler
from openpyxl import Workbook


def train_one_epoch(model, criterion, optimizer, train_loader, device):
    model.train()
    loss_sum, correct, total = 0.0, 0, 0
    start = time.time()

    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        loss_sum += loss.item() * images.size(0)
        _, pred = outputs.max(1)
        correct += pred.eq(labels).sum().item()
        total += labels.size(0)

    epoch_time = time.time() - start
    avg_loss = loss_sum / total
    top1 = 100.0 * correct / total
    throughput = total / epoch_time

    return avg_loss, top1, epoch_time, throughput


def evaluate(model, criterion, val_loader, device):
    model.eval()
    loss_sum, correct, total = 0.0, 0, 0

    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)

            loss_sum += loss.item() * images.size(0)
            _, pred = outputs.max(1)
            correct += pred.eq(labels).sum().item()
            total += labels.size(0)

    avg_loss = loss_sum / total
    top1 = 100.0 * correct / total

    return avg_loss, top1


def save_metrics_to_excel(metrics, filename="training_metrics.xlsx"):
    wb = Workbook()
    ws = wb.active
    ws.title = "Metrics"

    ws.append(["Epoch", "train_loss", "train_top1(%)", "val_loss", "val_top1(%)", "time(s)", "throughput(img/s)"])

    for m in metrics:
        ws.append([
            m["epoch"],
            m["train_loss"],
            m["train_top1"],
            m["val_loss"],
            m["val_top1"],
            m["time"],
            m["throughput"]
        ])

    wb.save(filename)


def train_ddp(rank, world_size, args):

    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)
    device = torch.device("cuda", rank)

    transform = transforms.Compose([
        transforms.Resize(224),
        transforms.ToTensor(),
    ])

    # You can replace FakeData with CIFAR-10 or ImageNet for real training
    train_dataset = datasets.FakeData(transform=transform)
    val_dataset = datasets.FakeData(transform=transform)

    train_sampler = DistributedSampler(train_dataset, num_replicas=world_size, rank=rank, shuffle=True)
    val_sampler = DistributedSampler(val_dataset, num_replicas=world_size, rank=rank, shuffle=False)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, sampler=train_sampler, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, sampler=val_sampler, num_workers=2)

    model = models.resnet18().to(device)
    model = DDP(model, device_ids=[rank])

    criterion = nn.CrossEntropyLoss().to(device)
    optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    metrics = []

    for epoch in range(1, args.epochs + 1):
        train_loader.sampler.set_epoch(epoch)

        train_loss, train_top1, epoch_time, throughput = train_one_epoch(
            model, criterion, optimizer, train_loader, device
        )
        val_loss, val_top1 = evaluate(model, criterion, val_loader, device)

        scheduler.step()   # 不再传入 epoch ⇒ 不会触发警告

        if rank == 0:
            print(f"Epoch {epoch} | "
                  f"train_loss={train_loss:.4f}, train_top1={train_top1:.2f}% | "
                  f"val_loss={val_loss:.4f}, val_top1={val_top1:.2f}% | "
                  f"time={epoch_time:.2f}s | throughput={throughput:.2f} img/s")

            metrics.append({
                "epoch": epoch,
                "train_loss": train_loss,
                "train_top1": train_top1,
                "val_loss": val_loss,
                "val_top1": val_top1,
                "time": epoch_time,
                "throughput": throughput
            })

    if rank == 0:
        save_metrics_to_excel(metrics)
        print("Excel 文件已保存：training_metrics.xlsx")

    dist.destroy_process_group()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=128)
    args = parser.parse_args()

    world_size = torch.cuda.device_count() * 2  # 2 nodes, each with 8 GPUs
    mp.spawn(train_ddp, args=(world_size, args), nprocs=world_size)


if __name__ == "__main__":
    main()

