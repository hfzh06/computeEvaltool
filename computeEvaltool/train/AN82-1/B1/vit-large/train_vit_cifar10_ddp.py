import os
import argparse
from pathlib import Path
import time
import importlib
import pandas as pd

import torch
import torch.nn as nn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler

import torchvision
import torchvision.transforms as transforms


def get_model(num_classes=10):
    try:
        vit_module = importlib.import_module("vit-large".replace("-", "_"))
        if hasattr(vit_module, "create_model"):
            return vit_module.create_model(num_classes=num_classes)
    except Exception:
        pass

    try:
        import timm
        return timm.create_model(
            "vit_large_patch16_224",
            pretrained=False,
            num_classes=num_classes
        )
    except Exception:
        pass

    print("WARNING: Using fallback ResNet18!")
    return torchvision.models.resnet18(num_classes=num_classes)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-path", default="/mnt/ray_share/cifar10")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--img-size", type=int, default=224)
    parser.add_argument("--save-dir", default="/mnt/ray_share/checkpoints")
    parser.add_argument("--resume", default="")
    parser.add_argument("--result-path", default="/root/computeEvaltool/train/B1/results/training_vit-large_1.xlsx")
    return parser.parse_args()


def main():
    args = parse_args()

    rank = int(os.environ["RANK"])
    local_rank = int(os.environ["LOCAL_RANK"])
    world_size = int(os.environ["WORLD_SIZE"])

    torch.cuda.set_device(local_rank)
    dist.init_process_group(backend="nccl")

    if rank == 0:
        print(f"[DDP Init] rank={rank}, world_size={world_size}, local_rank={local_rank}")
        print("Args:", args)

    # ---------------- DATA ----------------
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

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size,
        sampler=train_sampler, num_workers=args.num_workers, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size,
        sampler=val_sampler, num_workers=args.num_workers, pin_memory=True)

    # ---------------- MODEL ----------------
    model = get_model(num_classes=10)
    model.cuda()
    model = DDP(model, device_ids=[local_rank])

    criterion = nn.CrossEntropyLoss().cuda()

    effective_bs = args.batch_size * world_size
    lr = args.lr * (effective_bs / 256)
    optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9)

    scaler = torch.cuda.amp.GradScaler()

    start_epoch = 0
    if args.resume:
        ckpt = torch.load(args.resume, map_location=f"cuda:{local_rank}")
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        start_epoch = ckpt["epoch"] + 1

    if rank == 0:
        Path(args.save_dir).mkdir(parents=True, exist_ok=True)

    # ---------------- LOG LIST (new format) ----------------
    log_rows = []

    # ---------------- TRAIN LOOP ----------------
    for epoch in range(start_epoch, args.epochs):
        train_sampler.set_epoch(epoch)
        model.train()

        t0 = time.time()
        total_samples = 0

        # ---------- Train ----------
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

        # ---------- Validation ----------
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

                vloss += loss.item() * labels.size(0)
                vcorrect += (out.argmax(1) == labels).sum().item()
                vtotal += labels.size(0)

        # sync
        vloss = torch.tensor(vloss, device="cuda")
        vcorrect = torch.tensor(vcorrect, device="cuda")
        vtotal = torch.tensor(vtotal, device="cuda")
        tsample = torch.tensor(total_samples, device="cuda")

        dist.all_reduce(vloss)
        dist.all_reduce(vcorrect)
        dist.all_reduce(vtotal)
        dist.all_reduce(tsample)

        val_loss = (vloss / vtotal).item()
        val_acc = (vcorrect / vtotal).item() * 100
        epoch_time = time.time() - t0
        throughput = tsample.item() / epoch_time

        # ---------- Logging ----------
        if rank == 0:
            print(f"[Epoch {epoch+1}] time={epoch_time:.2f}s | thr={throughput:.2f} | "
                  f"loss={val_loss:.2f} | acc={val_acc:.2f}%")

            log_rows.append({
                "epoch": epoch + 1,
                "time(s)": round(epoch_time, 2),
                "throughput(samples/s)": round(throughput, 2),
                "loss": round(val_loss, 2),
                "acc(%)": round(val_acc, 2),
            })

    # ---------------- SAVE XLSX ----------------
    if rank == 0:
        df = pd.DataFrame(log_rows)
        Path(args.result_path).parent.mkdir(parents=True, exist_ok=True)
        df.to_excel(args.result_path, index=False)
        print(f"[INFO] Log saved → {args.result_path}")

    dist.destroy_process_group()


if __name__ == "__main__":
    main()
