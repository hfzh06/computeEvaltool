#!/usr/bin/env python3
"""
train_ds_resnet18_benchmark.py

DeepSpeed 版本的 ResNet18 极限性能测试。
保持与 train_ddp_resnet18_benchmark.py 相同功能、相同参数。

新增：
  --deepspeed
  --deepspeed_config
"""

import os
import argparse
import time
import deepspeed
import torch
import torch.nn as nn
import torch.distributed as dist
from torch.utils.data import DataLoader, DistributedSampler
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=90)
    parser.add_argument('--batch-size', type=int, default=128)
    parser.add_argument('--lr', type=float, default=0.1)
    parser.add_argument('--momentum', type=float, default=0.9)
    parser.add_argument('--weight-decay', type=float, default=5e-4)
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--data-root', type=str, default='./')

    # speed test
    parser.add_argument('--test-speed-only', action='store_true')
    parser.add_argument('--synthetic', action='store_true')
    parser.add_argument('--warmup-iters', type=int, default=20)
    parser.add_argument('--meas-iters', type=int, default=100)
    parser.add_argument('--print-interval', type=int, default=10)

    # DeepSpeed
    parser = deepspeed.add_config_arguments(parser)
    args = parser.parse_args()
    return args


def init_dist():
    dist.init_process_group(backend='nccl')
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    return local_rank, rank, world_size


def get_dataloader(args, world_size, rank):
    normalize = transforms.Normalize((0.4914, 0.4822, 0.4465),
                                     (0.2023, 0.1994, 0.2010))
    train_transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalize,
    ])

    train_dataset = datasets.CIFAR10(args.data_root, train=True, download=False,
                                     transform=train_transform)
    sampler = DistributedSampler(train_dataset, world_size, rank, shuffle=True)
    loader = DataLoader(train_dataset, batch_size=args.batch_size,
                        sampler=sampler, num_workers=args.workers)
    return loader


def synthetic_loader(args):
    class Synthetic(torch.utils.data.IterableDataset):
        def __iter__(self):
            for _ in range(10**7):
                yield torch.randn(3, 32, 32), torch.randint(0, 10, (1,)).item()
    ds = Synthetic()
    return DataLoader(ds, batch_size=args.batch_size)


def run_speed(model_engine, args, device, rank, world_size):
    warmup = args.warmup_iters
    meas = args.meas_iters

    loader = synthetic_loader(args) if args.synthetic else get_dataloader(args, world_size, rank)
    it = iter(loader)

    # warmup
    for _ in range(warmup):
        img, tgt = next(it)
        img = img.to(device)
        tgt = tgt.to(device)
        loss = model_engine(img, tgt)
        model_engine.backward(loss)
        model_engine.zero_grad()

    torch.cuda.synchronize(device)
    t0 = time.time()

    for i in range(meas):
        img, tgt = next(it)
        img = img.to(device)
        tgt = tgt.to(device)
        loss = model_engine(img, tgt)
        model_engine.backward(loss)
        model_engine.zero_grad()

        if (i+1) % args.print_interval == 0 and rank == 0:
            print(f"[SpeedTest] rank0 iter {i+1}/{meas}")

    torch.cuda.synchronize(device)
    t1 = time.time()

    elapsed = t1 - t0
    elapsed_tensor = torch.tensor(elapsed, device=device)
    dist.all_reduce(elapsed_tensor, op=dist.ReduceOp.MAX)
    elapsed_max = elapsed_tensor.item()

    total = args.batch_size * meas * world_size
    throughput = total / elapsed_max

    if rank == 0:
        print("===============================================")
        print(f"[DS SpeedTest RESULT] batch_size={args.batch_size}")
        print(f"Images/sec = {throughput:.2f}")
        print("===============================================")

    return throughput


def main():
    args = parse_args()
    local_rank, rank, world_size = init_dist()

    model = models.resnet18(num_classes=10)

    parameters = filter(lambda p: p.requires_grad, model.parameters())

    model_engine, optimizer, _, _ = deepspeed.initialize(
        args=args,
        model=model,
        model_parameters=parameters
    )

    device = torch.device("cuda", local_rank)

    if args.test_speed_only:
        run_speed(model_engine, args, device, rank, world_size)
        dist.barrier()
        dist.destroy_process_group()
        return


if __name__ == '__main__':
    main()

