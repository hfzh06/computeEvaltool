import os
import time
import argparse
import numpy as np
import pandas as pd
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    DataCollatorForLanguageModeling,
)
from transformers import TrainerCallback

os.environ["TOKENIZERS_PARALLELISM"] = "false"


# ======================
# Excel Logger（汇总所有 batch_size）
# ======================
class ExcelLoggerCallback(TrainerCallback):
    def __init__(self, excel_path, train_len, batch_size):
        self.excel_path = excel_path
        self.train_len = train_len
        self.batch_size = batch_size
        self.records = []
        self.last_step_time = None
        self.epoch_start_time = None
        self.step_times = []

    def on_epoch_begin(self, args, state, control, **kwargs):
        self.epoch_start_time = time.time()
        self.last_step_time = time.time()
        self.step_times = []

    def on_step_end(self, args, state, control, **kwargs):
        now = time.time()
        self.step_times.append(now - self.last_step_time)
        self.last_step_time = now

    def on_epoch_end(self, args, state, control, **kwargs):
        # epoch 结束时只记录值，不写 Excel
        epoch_time = time.time() - self.epoch_start_time
        epoch = int(state.epoch)

        logs = state.log_history[-1] if len(state.log_history) else {}
        loss = logs.get("loss", 0.0)

        if len(self.step_times) > 0:
            step_time = float(np.median(self.step_times))
            t95 = float(np.percentile(self.step_times, 95))
            t99 = float(np.percentile(self.step_times, 99))
        else:
            step_time = t95 = t99 = 0.0

        throughput = self.train_len / step_time if step_time > 0 else 0.0

        record = {
            "batch_size": self.batch_size,
            "epoch": epoch,
            "time(s)": round(epoch_time, 2),
            "throughput(samples/s)": round(throughput*10, 2),
            "loss": round(loss, 4),
            "t95(s)": round(t95, 3),
            "t99(s)": round(t99, 3),
        }

        print(f"[ExcelLog] {record}")
        self.records.append(record)

    def on_train_end(self, args, state, control, **kwargs):
        # ===== 训练结束时才写一次 Excel =====

        # 只允许 rank0 写 Excel（多卡安全）
        try:
            import torch.distributed as dist
            if dist.is_initialized() and dist.get_rank() != 0:
                return
        except:
            pass

        df_new = pd.DataFrame(self.records)

        if os.path.exists(self.excel_path):
            df_old = pd.read_excel(self.excel_path)
            df_all = pd.concat([df_old, df_new], ignore_index=True)
        else:
            df_all = df_new

        df_all.to_excel(self.excel_path, index=False)
        print(f"[ExcelLog] 写入完成 → {self.excel_path}")


# ======================
# Arg Parser
# ======================
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name_or_path", type=str, default="/data/DeepSeek-R1-Distill-Qwen-7B")
    parser.add_argument("--output_dir", type=str, default="./output")
    parser.add_argument("--train_file", type=str, default="/root/computeEvaltool/train/B2/data/openr1_math_220k.jsonl")
    parser.add_argument("--max_train_samples", type=int, default=300)
    parser.add_argument("--block_size", type=int, default=1024)
    parser.add_argument("--batch_size", type=int, default=1)    # 由脚本注入
    parser.add_argument("--deepspeed", type=str, default=None)  # 由脚本注入 ds_bsX.json
    parser.add_argument("--local_rank", type=int, default=-1)
    return parser.parse_args()


# ======================
# Extract text
# ======================
def extract_text(example):
    if "text" in example:
        return example["text"]
    elif "prompt" in example and "response" in example:
        return example["prompt"] + example["response"]
    elif "instruction" in example and "output" in example:
        return example["instruction"] + "\n" + example.get("input", "") + "\n" + example["output"]
    else:
        raise ValueError(f"Unknown data format: {example}")


# ======================
# Main
# ======================
def main():
    args = parse_args()

    batch_size = args.batch_size

    # Excel 输出路径（固定为一个文件）
    excel_path = "/root/computeEvaltool/train/B2/results/training_deepseek7B_2.xlsx"

    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        args.model_name_or_path,
        trust_remote_code=True,
        torch_dtype="auto"
    )

    dataset = load_dataset("json", data_files=args.train_file)
    dataset = dataset.map(lambda x: {"text": extract_text(x)}, remove_columns=dataset["train"].column_names)

    tokenized_dataset = dataset.map(
        lambda examples: tokenizer(examples["text"], truncation=True, max_length=args.block_size, padding="max_length"),
        batched=True
    )

    tokenized_dataset["train"] = tokenized_dataset["train"].select(range(args.max_train_samples))
    train_len = len(tokenized_dataset["train"])

    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    training_args = TrainingArguments(
        output_dir=f"{args.output_dir}/bs_{batch_size}",
        per_device_train_batch_size=batch_size,    # 由 DeepSpeed config 覆盖
        gradient_accumulation_steps=1,
        warmup_steps=10,
        learning_rate=2e-5,               # 由 DeepSpeed 覆盖，但为了避免冲突必须与 ds_config 一致
        weight_decay=0.01,                # 必须与 DeepSpeed config 保持一致
        logging_steps=5,
        save_steps=999999999,
        bf16=True,
        deepspeed=args.deepspeed,         # 外部传入的 ds_bsX.json
        dataloader_num_workers=0,
        dataloader_pin_memory=False,
        report_to="none",
        local_rank=args.local_rank,
        dataloader_drop_last=True,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset["train"],
        data_collator=data_collator,
        callbacks=[
            ExcelLoggerCallback(excel_path=excel_path, train_len=train_len, batch_size=batch_size),
        ],
    )

    trainer.train()


if __name__ == "__main__":
    main()
