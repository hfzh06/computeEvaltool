#!/usr/bin/env python3
import os
import argparse
from pathlib import Path
import time
import importlib

import torch
import torch.nn as nn
import torch.distributed as dist
import deepspeed 
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.distributed import DistributedSampler
import torchvision
import torchvision.transforms as transforms

# ============== 🌟 新增：随机数据集类 �� ==============
class RandomDataset(Dataset):
    """
    生成随机图像和标签的 Dataset。
    用于在不依赖实际数据I/O的情况下测试训练速度和DeepSpeed性能。
    """
    def __init__(self, image_size=224, num_classes=10, length=50000):
        self.length = length
        self.image_size = image_size
        self.num_classes = num_classes

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        # 随机生成 (3, 224, 224) 的浮点图像数据 (FP32)
        image = torch.randn(3, self.image_size, self.image_size, dtype=torch.float32)
        # 随机生成标签 (Long类型)
        label = torch.randint(0, self.num_classes, (1,)).squeeze(0)
        return image, label

# ======================================================================
# ⚙️ 模型加载：使用 timm ViT-Large 或 fallback resnet18
# ======================================================================
def get_model(num_classes=10):
    try:
        import timm
        if dist.get_rank() == 0:
            print("[MODEL] Using timm ViT-Large.")
        return timm.create_model(
            "vit_large_patch16_224",
            pretrained=False,
            num_classes=num_classes
        )
    except Exception as e:
        if dist.get_rank() == 0:
            print(f"[ERROR] timm or ViT-Large loading failed: {e}")
            print("[WARNING] Using fallback ResNet18!")
        return torchvision.models.resnet18(num_classes=num_classes)


# ======================================================================
# ⚙️ 参数解析
# ======================================================================
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch-size", type=int, default=64, help="Micro batch size per GPU.") 
    parser.add_argument("--epochs", type=int, default=3, help="Number of training epochs.")
    parser.add_argument("--lr", type=float, default=0.0001, help="Base learning rate.") # 与 ds_config 保持一致
    parser.add_argument('--ds_config', type=str, default='ds_config_vit.json', help='DeepSpeed config file path.')
    parser.add_argument("--data-path", default="/mnt/ray_share/cifar10", help="Path (Placeholder) to store CIFAR10 data.")
    parser.add_argument("--img-size", type=int, default=224, help="Input image size for ViT.")
    parser.add_argument("--num-workers", type=int, default=8, help="Number of data loader workers.") # 🌟 增加 num_workers
    parser.add_argument("--save-dir", default="/mnt/ray_share/checkpoints", help="Directory to save checkpoints.")
    parser.add_argument("--resume", default="", help="Path/tag to resume DeepSpeed checkpoint from.")

    return parser.parse_args()


# ======================================================================
# 🚀 静态初始化 distributed
# ======================================================================
def init_distributed():
    rank = int(os.environ["RANK"])
    local_rank = int(os.environ["LOCAL_RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    
    if not dist.is_initialized():
        dist.init_process_group(
            backend="nccl",
            init_method="env://", 
            rank=rank,
            world_size=world_size
        )
    
    torch.cuda.set_device(local_rank)
    dist.barrier() 
    
    if dist.get_rank() == 0:
        print(f"[DS Init] Rank={dist.get_rank()}, WorldSize={dist.get_world_size()}, LocalRank={local_rank} (Env Static)")


# ======================================================================
# 🎯 main
# ======================================================================
def main():
    args = parse_args()
    init_distributed()
    
    rank = dist.get_rank()
    local_rank = int(os.environ["LOCAL_RANK"])
    world_size = dist.get_world_size()
    device = torch.device(f"cuda:{local_rank}")

    if rank == 0:
        print("Args:", args)
    
    # --- 1. 🌟 数据集加载 (使用随机数据集) ---
    TRAIN_SAMPLES = 50000 
    VAL_SAMPLES = 10000
    NUM_CLASSES = 10 
    
    if rank == 0:
        print(f"[DATA] Using RandomDataset: {TRAIN_SAMPLES} train samples, {VAL_SAMPLES} val samples. (Fast performance testing)")
    
    train_dataset = RandomDataset(
        image_size=args.img_size, 
        num_classes=NUM_CLASSES, 
        length=TRAIN_SAMPLES
    )
    val_dataset = RandomDataset(
        image_size=args.img_size, 
        num_classes=NUM_CLASSES, 
        length=VAL_SAMPLES
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
    
    # --- 2. DeepSpeed 初始化 ---
    model = get_model(num_classes=NUM_CLASSES)
    criterion = nn.CrossEntropyLoss().to(device) 

    model_engine, optimizer, _, lr_scheduler = deepspeed.initialize(
        model=model,
        model_parameters=model.parameters(), 
        config=args.ds_config,
        lr_scheduler=None,
    )

    start_epoch = 0
    # ... (Resume 逻辑保持不变，为简洁省略)
    if rank == 0:
        Path(args.save_dir).mkdir(parents=True, exist_ok=True)


    # --- 3. 训练循环 ---
    for epoch in range(start_epoch, args.epochs):
        train_sampler.set_epoch(epoch)
        model_engine.train()
        t0 = time.time()

        total_loss = 0
        total_correct = 0
        total_samples = 0

        for step, (images, labels) in enumerate(train_loader):
            # 🌟 关键：数据传输和 FP16 转换，避免类型错误和I/O阻塞
            images = images.to(device, non_blocking=True).half()
            labels = labels.to(device, non_blocking=True)

            out = model_engine(images) 
            loss = criterion(out, labels)

            model_engine.backward(loss)
            model_engine.step()

            preds = out.argmax(1)
            total_correct += (preds == labels).sum().item()
            total_samples += labels.size(0)
            total_loss += loss.item() * labels.size(0)

        # 归约指标 (All-Reduce)
        metrics = torch.tensor([total_loss, total_correct, total_samples], device=device, dtype=torch.float64)
        dist.all_reduce(metrics, op=dist.ReduceOp.SUM)
        
        global_loss, global_correct, global_total = metrics.tolist()

        train_loss = global_loss / global_total
        train_acc = global_correct / global_total
        
        epoch_total_duration = time.time() - t0
        throughput = global_total / epoch_total_duration if epoch_total_duration > 0 else 0


        # 验证 (使用随机数据，指标无意义，但需运行以完成整个 Epoch 周期)
        model_engine.eval()
        
        if rank == 0:
            vloss = 0
            vcorrect = 0
            vtotal = 0
            
            with torch.no_grad():
                for images, labels in val_loader:
                    # 🌟 关键：数据传输和 FP16 转换
                    images = images.to(device, non_blocking=True).half()
                    labels = labels.to(device, non_blocking=True)
                    
                    out = model_engine.module(images) 
                    loss = criterion(out, labels)

                    preds = out.argmax(1)
                    vloss += loss.item() * labels.size(0)
                    vcorrect += (preds == labels).sum().item()
                    vtotal += labels.size(0)
            
            val_loss = vloss / vtotal
            val_acc = vcorrect / vtotal
            
            # 🌟 输出吞吐量，评估训练速度
            print(f"[Epoch {epoch+1}/{args.epochs}] "
                  f"Train Loss {train_loss:.4f}, Acc {train_acc*100:.2f}% | "
                  f"Val Loss {val_loss:.4f}, Acc {val_acc*100:.2f}% | "
                  f"Throughput {throughput:.2f} samples/s " 
                  f"({epoch_total_duration:.1f}s)")

            # 保存 DeepSpeed Checkpoint
            client_state = {'epoch': epoch}
            save_tag = f"epoch_{epoch+1:03d}"
            model_engine.save_checkpoint(args.save_dir, save_tag, client_state=client_state)
            print(f"Saved DeepSpeed checkpoint to {args.save_dir}/{save_tag}")

        dist.barrier()
    
    dist.destroy_process_group()


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        if dist.is_initialized():
             print(f"[PID:{os.getpid()} Rank {dist.get_rank()}] An error occurred: {e}", flush=True)
             dist.destroy_process_group()
        else:
             print(f"An error occurred: {e}", flush=True)
