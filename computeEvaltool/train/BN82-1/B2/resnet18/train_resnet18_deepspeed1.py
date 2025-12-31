import os
import time
import argparse
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import deepspeed
import pandas as pd
import json
from pathlib import Path
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler


def print_rank0(msg: str):
    if torch.distributed.get_rank() == 0:
        print(f"[Rank 0] {msg}")


def main():

    parser = argparse.ArgumentParser()
    parser.add_argument('--bs', type=int, default=32)
    parser.add_argument('--local_rank', type=int, default=-1)
    args = parser.parse_args()

    # ====== 1. 初始化 ======
    deepspeed.init_distributed()

    rank = torch.distributed.get_rank()
    world_size = torch.distributed.get_world_size()
    local_rank = int(os.environ.get("LOCAL_RANK", args.local_rank))
    torch.cuda.set_device(local_rank)

    # ====== 2. 读取 DeepSpeed 配置 ======
    ds_config_path = "/root/computeEvaltool/train/B2/resnet18/ds_config.json"
    with open(ds_config_path, 'r') as f:
        ds_config = json.load(f)

    ds_config['train_micro_batch_size_per_gpu'] = args.bs
    if 'train_batch_size' in ds_config and ds_config['train_batch_size'] != "auto":
        del ds_config['train_batch_size']

    # ====== 3. 数据集 ======
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
    ])
    train_set = torchvision.datasets.CIFAR10(root="/data/cifar10", train=True, download=False, transform=transform_train)

    model = torchvision.models.resnet18(num_classes=10)

    # ====== 4. DeepSpeed 初始化 ======
    try:
        model_engine, optimizer, _, _ = deepspeed.initialize(
            model=model,
            model_parameters=model.parameters(),
            config=ds_config
        )

        train_sampler = DistributedSampler(train_set, num_replicas=world_size, rank=rank, shuffle=True)
        train_loader = DataLoader(train_set, batch_size=args.bs, sampler=train_sampler, num_workers=4, pin_memory=True)

        global_bs = args.bs * world_size
        

        warmup_epochs = 3       # 不记录
        record_epochs = 3       # 只记录最后 3 个 epoch
        total_epochs = warmup_epochs + record_epochs

        # ====== 5. 训练 ======
        for epoch in range(1, total_epochs + 1):
            train_sampler.set_epoch(epoch)
            torch.cuda.synchronize()
            t_start = time.time()

            total_loss = 0.0
            correct = 0
            seen = 0

            model_engine.train()
            for imgs, labels in train_loader:
                imgs = imgs.to(local_rank).half()
                labels = labels.to(local_rank)

                outputs = model_engine(imgs)
                loss = nn.CrossEntropyLoss()(outputs, labels)

                model_engine.backward(loss)
                model_engine.step()

                total_loss += loss.item() * imgs.size(0)

                pred = outputs.argmax(dim=1)
                correct += (pred == labels).sum().item()
                seen += imgs.size(0)

            torch.cuda.synchronize()
            t_end = time.time()

            epoch_time = t_end - t_start
            throughput = seen / epoch_time
            avg_loss = total_loss / seen
            acc = correct / seen * 100

            print_rank0(f"Epoch {epoch} | Time={epoch_time:.2f}s | TP={throughput:.2f} | Loss={avg_loss:.4f} | Acc={acc:.2f}%")

            # ========= 仅保存最后 3 个 epoch =========
            if epoch > warmup_epochs and rank == 0:
                result_row = {
                    "batch_size": args.bs,
                    "epoch": epoch,
                    "time(s)": round(epoch_time, 2),
                    "throughput(samples/s)": round(throughput, 2),
                    "loss": round(avg_loss, 4),
                    "acc(%)": round(acc, 2),
                }
                save_result(result_row)

    except RuntimeError as e:
        if "out of memory" in str(e).lower():
            print_rank0(f"OOM at BS={args.bs}")
            if rank == 0:
                save_result({"batch_size": args.bs, "status": "OOM"})
            exit(0)
        else:
            raise e


def save_result(data):
    file_path = "/root/computeEvaltool/train/B2/results/training_resnet18_2.xlsx"

    if os.path.exists(file_path):
        df_old = pd.read_excel(file_path)
        df_new = pd.DataFrame([data])
        df = pd.concat([df_old, df_new], ignore_index=True)
    else:
        df = pd.DataFrame([data])

    df.to_excel(file_path, index=False)
    print(f"[Rank 0] Saved result to Excel → {file_path}")


if __name__ == "__main__":
    main()
