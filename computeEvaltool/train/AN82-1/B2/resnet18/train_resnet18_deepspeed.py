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


def print_rank0(msg: str):
    if torch.distributed.get_rank() == 0:
        print(msg)


def main():
    import os
    # ====== NCCL 通信配置 ======
    os.environ["NCCL_SOCKET_IFNAME"] = "enp168s0np0"
    os.environ["NCCL_DEBUG"] = "INFO"
    os.environ["NCCL_IB_DISABLE"] = "0"
    # os.environ["MASTER_ADDR"] = "10.1.73.25"
    # os.environ["MASTER_PORT"] = "29500"

    deepspeed.init_distributed()

    world_size = torch.distributed.get_world_size()
    rank = torch.distributed.get_rank()
    local_rank = int(os.environ["LOCAL_RANK"])

    torch.cuda.set_device(local_rank)
    print_rank0(f"World Size={world_size}, Rank={rank}, Local Rank={local_rank}")

    # ============================
    # CIFAR10 DATASET
    # ============================
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
    ])

    train_set = torchvision.datasets.CIFAR10(
        # root="/share/cifar10",
        root="./cifar10",
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
        batch_size=32,
        sampler=train_sampler,
        num_workers=4,
        pin_memory=True
    )

    # ============================
    # MODEL
    # ============================
    print_rank0("Loading ResNet18...")
    model = torchvision.models.resnet18(num_classes=10)
    criterion = nn.CrossEntropyLoss()

    # ============================
    # DeepSpeed Engine
    # ============================
    ds_config = "/root/computeEvaltool/train/B2/resnet18/ds_config.json"
    print_rank0(f"Using DeepSpeed config: {ds_config}")

    if rank == 0:
        print_rank0(open(ds_config).read())

    model_engine, optimizer, _, _ = deepspeed.initialize(
        model=model,
        model_parameters=model.parameters(),
        config=ds_config,
    )

    # ============================
    # METRICS RECORD
    # ============================
    epoch_records = []
    num_epochs = 20
    global_batch_size = 32 * world_size

    print_rank0(f"Training for {num_epochs} epochs...")

    # ============================
    # TRAINING LOOP
    # ============================
    for epoch in range(1, num_epochs + 1):

        train_sampler.set_epoch(epoch)
        epoch_start = time.time()

        model_engine.train()

        # ======= 新增统计项 ========
        total_loss = 0.0
        total_correct = 0
        total_seen = 0

        for batch_idx, (imgs, labels) in enumerate(train_loader):
            imgs = imgs.to(local_rank, non_blocking=True).half()
            labels = labels.to(local_rank, non_blocking=True)

            outputs = model_engine(imgs)
            loss = criterion(outputs, labels)

            model_engine.backward(loss)
            model_engine.step()

            # accumulate loss
            total_loss += loss.item()

            # accuracy
            preds = outputs.argmax(dim=1)
            total_correct += (preds == labels).sum().item()
            total_seen += labels.size(0)

        # ======= epoch time + throughput ========
        epoch_time = time.time() - epoch_start
        num_batches = len(train_loader)
        num_samples = num_batches * global_batch_size
        throughput = num_samples / epoch_time

        # aggregate metrics across GPUs
        avg_loss_tensor = torch.tensor(total_loss / num_batches, device=local_rank)
        acc_tensor = torch.tensor(total_correct / total_seen, device=local_rank)

        torch.distributed.all_reduce(avg_loss_tensor, op=torch.distributed.ReduceOp.SUM)
        torch.distributed.all_reduce(acc_tensor, op=torch.distributed.ReduceOp.SUM)

        avg_loss = (avg_loss_tensor / world_size).item()
        accuracy = (acc_tensor / world_size).item()

        print_rank0(
            f"[Epoch {epoch}] time={epoch_time:.2f}s "
            f"throughput={throughput:.2f} samples/s "
            f"loss={avg_loss:.4f} acc={accuracy:.4f}"
        )

        if rank == 0:
            epoch_records.append({
                "epoch": epoch,
                "epoch_time": epoch_time,
                "throughput": throughput,
                "num_batches": num_batches,
                "num_samples": num_samples,
                "avg_loss": avg_loss,
                "accuracy": accuracy,
            })

    # ============================
    # SAVE XLSX (rank 0 only)
    # ============================
    if rank == 0:
        results_dir = "/root/computeEvaltool/train/B2/results"
        Path(results_dir).mkdir(parents=True, exist_ok=True)

        formatted_records = []
        for rec in epoch_records:
            formatted_records.append({
                "Epoch": rec["epoch"],
                "time(s)": round(rec["epoch_time"], 2),
                "throughput(samples/s)": round(rec["throughput"], 2),
                "loss": round(rec["avg_loss"], 2),
                "acc(%)": round(rec["accuracy"] * 100, 2),
            })

        df = pd.DataFrame(formatted_records)
        out_file = f"{results_dir}/training_resnet18_1.xlsx"
        df.to_excel(out_file, index=False)

        print_rank0(f"Saved formatted training logs → {out_file}")


    print_rank0(f"Saved formatted training logs → {out_file}")



if __name__ == "__main__":
    main()

