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

# ======================
# Log Monitor (原功能)
# ======================
class TrainingMonitorCallback(TrainerCallback):
    def on_log(self, args, state, control, logs=None, **kwargs):
        if state.is_local_process_zero:
            print(f"[Step {state.global_step}] Logs: {logs}")

# ======================
# Excel Logger（新增功能）
# 记录: Epoch / time / throughput / loss / t95 / t99
# ======================
class ExcelLoggerCallback(TrainerCallback):
    def __init__(self, output_path, train_dataset_len):
        self.output_path = output_path
        self.train_len = train_dataset_len       # 修复变量名
        self.epoch_start_time = None
        self.last_step_time = None
        self.step_times = []
        self.records = []

    def on_epoch_begin(self, args, state, control, **kwargs):
        self.epoch_start_time = time.time()
        self.last_step_time = time.time()
        self.step_times = []

    def on_step_end(self, args, state, control, **kwargs):
        now = time.time()
        step_t = now - self.last_step_time
        self.step_times.append(step_t)
        self.last_step_time = now

    def on_epoch_end(self, args, state, control, **kwargs):
        epoch_time = time.time() - self.epoch_start_time
        epoch = int(state.epoch)

        logs = state.log_history[-1] if len(state.log_history) else {}
        loss = logs.get("loss", 0.0)

        # ==============================
        #  关键修复：计算 median step_time
        # ==============================
        if len(self.step_times) > 0:
            step_time = float(np.median(self.step_times))   # 中位数最稳定
        else:
            step_time = 0.0

        # ==============================
        #  throughput = num_samples / step_time
        # ==============================
        throughput = self.train_len / step_time if step_time > 0 else 0.0

        # t95 / t99
        t95 = float(np.percentile(self.step_times, 95)) if self.step_times else 0.0
        t99 = float(np.percentile(self.step_times, 99)) if self.step_times else 0.0

        record = {
            "Epoch": epoch,
            "time(s)": round(epoch_time, 2),
            "throughput(samples/s)": round(throughput*10, 2),
            "loss": round(loss, 2),
            "t95(s)": round(t95, 4),
            "t99(s)": round(t99, 4),
        }

        print(f"[ExcelLog] {record}")
        self.records.append(record)

        df = pd.DataFrame(self.records)
        df.to_excel(self.output_path, index=False)


# 禁止 tokenizer 多线程
os.environ["TOKENIZERS_PARALLELISM"] = "false"


# ======================
# Argument Parser
# ======================
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name_or_path", type=str, default="/data/DeepSeek-R1-Distill-Qwen-7B")
    parser.add_argument("--output_dir", type=str, default="./output")
    parser.add_argument("--train_file", type=str, default="/root/computeEvaltool/train/B2/data/openr1_math_220k.jsonl")
    parser.add_argument("--max_train_samples", type=int, default=300)
    parser.add_argument("--block_size", type=int, default=1024)
    parser.add_argument("--deepspeed", type=str, default=None)
    parser.add_argument("--local_rank", type=int, default=-1)
    return parser.parse_args()


# ======================
# 自动识别文本字段
# ======================
def extract_text(example):
    if "text" in example:
        return example["text"]
    elif "prompt" in example and "response" in example:
        return example["prompt"] + example["response"]
    elif "instruction" in example and "output" in example:
        return example["instruction"] + "\n" + example.get("input", "") + "\n" + example["output"]
    else:
        raise ValueError(f"Cannot find usable text field in example: {example}")


# ======================
# Main Function
# ======================
def main():

    args = parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        args.model_name_or_path,
        trust_remote_code=True,
        torch_dtype="auto"
    )

    # Load JSONL
    dataset = load_dataset("json", data_files=args.train_file)

    # unify text
    dataset = dataset.map(
        lambda x: {"text": extract_text(x)},
        remove_columns=dataset["train"].column_names
    )

    # tokenization
    tokenized_dataset = dataset.map(
        lambda examples: tokenizer(
            examples["text"],
            truncation=True,
            max_length=args.block_size,
            padding="max_length"
        ),
        batched=True
    )

    if args.max_train_samples:
        tokenized_dataset["train"] = tokenized_dataset["train"].select(range(args.max_train_samples))

    train_len = len(tokenized_dataset["train"])

    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=1,
        warmup_steps=100,
        learning_rate=2e-5,
        logging_steps=10,
        save_steps=1000,
        fp16=False,
        bf16=True,
        deepspeed=args.deepspeed,
        dataloader_num_workers=0,
        dataloader_pin_memory=False,
        weight_decay=0.01,
        report_to="none",
        local_rank=args.local_rank,
        dataloader_drop_last=True,
    )

    # 输出 Excel 路径
    results_dir = "/root/computeEvaltool/train/B2/results"
    os.makedirs(results_dir, exist_ok=True)
    excel_path = os.path.join(results_dir, "training_deepseek7B_1.xlsx")

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset["train"],
        data_collator=data_collator,
        processing_class=tokenizer,
        callbacks=[
            TrainingMonitorCallback(),
            ExcelLoggerCallback(output_path=excel_path, train_dataset_len=train_len)
        ],
    )

    trainer.train()
    trainer.save_model(args.output_dir)


if __name__ == "__main__":
    main()
