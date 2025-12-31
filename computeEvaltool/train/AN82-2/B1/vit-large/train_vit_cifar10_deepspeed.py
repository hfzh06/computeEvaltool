import torch
import time
from torch.utils.data import DataLoader
from transformers import ViTForImageClassification, AutoImageProcessor, get_scheduler
from torch.optim import AdamW
from datasets import load_dataset
import deepspeed
from tqdm import tqdm

# ========== 加载模型与处理器 ==========
model_path = "/home/wtc/vit-large/vit-large"
model = ViTForImageClassification.from_pretrained(model_path, num_labels=10)
processor = AutoImageProcessor.from_pretrained(model_path)

# ========== 加载 CIFAR10 ==========
dataset = load_dataset("cifar10")

# 正确：map + batched=True，HuggingFace 官方方法
def transform(batch):
    inputs = processor(images=batch["img"], return_tensors="pt")
    inputs["labels"] = batch["label"]
    return inputs

prepared_ds = dataset.map(transform, batched=True)

train_dataloader = DataLoader(prepared_ds["train"], batch_size=8, shuffle=True)
test_dataloader = DataLoader(prepared_ds["test"], batch_size=8)

# ========== 优化器 & 学习率 ==========
optimizer = AdamW(model.parameters(), lr=5e-5)
num_training_steps = len(train_dataloader) * 3  # 3 epochs

lr_scheduler = get_scheduler(
    "linear",
    optimizer=optimizer,
    num_warmup_steps=0,
    num_training_steps=num_training_steps
)

# ========== DeepSpeed 初始化 ==========
model, optimizer, _, lr_scheduler = deepspeed.initialize(
    model=model,
    optimizer=optimizer,
    args=None,
    lr_scheduler=lr_scheduler,
    config="/home/wtc/vit-large/ds_config.json"
)

# ========== 训练循环 ==========
for epoch in range(3):
    model.train()
    epoch_start = time.time()

    for step, batch in enumerate(tqdm(train_dataloader, desc=f"Epoch {epoch}")):
        step_start = time.time()

        # 把数据放到当前 GPU / 分布式设备
        batch = {k: v.to(model.device) for k, v in batch.items()}

        outputs = model(**batch)
        loss = outputs.loss

        model.backward(loss)
        model.step()

        # 打印步时间
        step_end = time.time()
        step_time = step_end - step_start
        sample_time = step_time / batch["labels"].size(0)

        if step % 10 == 0 and model.local_rank == 0:
            print(f"Epoch {epoch} | Step {step}/{len(train_dataloader)} | "
                  f"Loss: {loss.item():.4f} | Step time: {step_time:.3f}s | "
                  f"Per sample: {sample_time:.4f}s")

    epoch_end = time.time()
    if model.local_rank == 0:
        print(f"Epoch {epoch} finished in {(epoch_end - epoch_start):.2f}s")

# ========== 验证 ==========
if model.local_rank == 0:
    print("Evaluating on test set...")

model.eval()
correct = 0
total = 0

with torch.no_grad():
    for batch in tqdm(test_dataloader, desc="Evaluating"):
        batch = {k: v.to(model.device) for k, v in batch.items()}
        outputs = model(**batch)
        preds = outputs.logits.argmax(dim=-1)
        correct += (preds == batch["labels"]).sum().item()
        total += preds.size(0)

if model.local_rank == 0:
    print(f"Test Accuracy: {correct / total * 100:.2f}%")

