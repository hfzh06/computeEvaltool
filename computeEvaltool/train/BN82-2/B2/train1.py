import sys
import logging
import argparse
import time
import numpy as np
import pandas as pd  # 用于导出 Excel/CSV
import datasets
import transformers
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    DataCollatorForLanguageModeling,
    TrainerCallback,
)

logger = logging.getLogger(__name__)

# =================================================================
# 自定义 Callback 用于统计每步耗时 (Step Latency)
# =================================================================
class StepTimeCallback(TrainerCallback):
    def __init__(self):
        self.step_times = []
        self.current_step_start = None

    def on_step_begin(self, args, state, control, **kwargs):
        # 记录每一步开始的时间
        self.current_step_start = time.time()

    def on_step_end(self, args, state, control, **kwargs):
        # 记录每一步结束的时间，并计算差值
        if self.current_step_start is not None:
            duration = time.time() - self.current_step_start
            self.step_times.append(duration)
            self.current_step_start = None

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name_or_path", type=str, default="/root/computeEvaltool/train/deepseek-70b")
    parser.add_argument("--output_dir", type=str, default="./output")
    parser.add_argument("--train_file", type=str, default="/root/computeEvaltool/train/B2/data/openr1_math_220k.jsonl")
    parser.add_argument("--max_train_samples", type=int, default=1)
    parser.add_argument("--block_size", type=int, default=1024)
    parser.add_argument("--deepspeed", type=str, default=None)
    parser.add_argument("--local_rank", type=int, default=-1, help="local rank passed by deepspeed/torchrun")
    return parser.parse_args()


def extract_text(example):
    if "text" in example:
        return example["text"]
    elif "prompt" in example and "response" in example:
        return example["prompt"] + example["response"]
    elif "instruction" in example and "output" in example:
        return example["instruction"] + "\n" + example.get("input", "") + "\n" + example["output"]
    else:
        raise ValueError(f"Cannot find usable text field in example: {example}")


def main():
    import os
    

    # 2. 打印详细日志，如果卡住能看到卡在哪一步
    os.environ["NCCL_DEBUG"] = "INFO"
    args = parse_args()

    # =================================================================
    # 1. TrainingArguments 初始化
    # =================================================================
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=8,
        warmup_steps=100,
        learning_rate=2e-5,
        logging_steps=1,
        save_steps=1000,
        fp16=False,
        bf16=True,
        deepspeed=args.deepspeed,
        dataloader_num_workers=4,
        weight_decay=0.01,
        report_to="none",
        local_rank=args.local_rank,
        logging_dir=f"{args.output_dir}/logs",
        logging_first_step=True,
        disable_tqdm=False,
    )

    # 2. 配置日志
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )

    if training_args.local_rank in [-1, 0]:
        logger.setLevel(logging.INFO)
        datasets.utils.logging.set_verbosity_warning()
        transformers.utils.logging.set_verbosity_info()
    else:
        logger.setLevel(logging.WARN)
        datasets.utils.logging.set_verbosity_error()
        transformers.utils.logging.set_verbosity_error()

    if training_args.local_rank == 0:
        logger.info(f"Training arguments initialized: {training_args}")
        # 打印一下环境变量以确认生效
        logger.info(f"NCCL_SOCKET_IFNAME set to: {os.environ.get('NCCL_SOCKET_IFNAME')}")
        logger.info(f"NCCL_IB_DISABLE set to: {os.environ.get('NCCL_IB_DISABLE')}")

    # 3. 加载模型和 Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name_or_path,
        trust_remote_code=True
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        args.model_name_or_path,
        trust_remote_code=True,
        torch_dtype="auto"
    )

    # 4. 加载数据
    dataset = datasets.load_dataset("json", data_files=args.train_file)

    # 5. 数据预处理
    processing_num_workers = 16

    with training_args.main_process_first(desc="dataset map processing"):
        dataset = dataset.map(
            lambda x: {"text": extract_text(x)},
            remove_columns=dataset["train"].column_names,
            desc="Converting to unified text format",
            num_proc=processing_num_workers
        )

        tokenized_dataset = dataset.map(
            lambda examples: tokenizer(
                examples["text"],
                truncation=True,
                max_length=args.block_size,
                padding="max_length"
            ),
            batched=True,
            num_proc=processing_num_workers,
            remove_columns=["text"]
        )

    if args.max_train_samples:
        tokenized_dataset["train"] = tokenized_dataset["train"].select(range(args.max_train_samples))

    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    # =================================================================
    # 初始化自定义 Callback
    # =================================================================
    step_time_callback = StepTimeCallback()

    # 6. 初始化 Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset["train"],
        data_collator=data_collator,
        processing_class=tokenizer,
        callbacks=[step_time_callback]  # 注册 Callback
    )

    if training_args.local_rank in [-1, 0]:
        transformers.utils.logging.set_verbosity_info()

    # 开始训练
    logger.info("*** Starting training ***")
    trainer.train()
    trainer.save_model(args.output_dir)

    # =================================================================
    # 7. 计算统计数据并保存 Excel/CSV
    # =================================================================
    if training_args.local_rank in [-1, 0]:
        times = step_time_callback.step_times

        # 过滤掉前几步 Warmup 的时间
        warmup_steps_skip = 2
        if len(times) > warmup_steps_skip:
            valid_times = times[warmup_steps_skip:]
            skipped_msg = f"(skipped first {warmup_steps_skip} steps)"
        else:
            valid_times = times
            skipped_msg = "(no steps skipped)"

        if len(valid_times) > 0:
            arr = np.array(valid_times)

            # 计算各项指标
            avg_time = np.mean(arr)
            min_time = np.min(arr)
            max_time = np.max(arr)
            t95 = np.percentile(arr, 95)
            t99 = np.percentile(arr, 99)

            # 计算吞吐量 (Samples per second)
            # Total Batch Size = per_device * accum_steps * world_size
            total_batch_size = training_args.per_device_train_batch_size * \
                               training_args.gradient_accumulation_steps * \
                               training_args.world_size

            avg_throughput = total_batch_size / avg_time

            # -------------------------------------------------------
            # A. 打印到控制台
            # -------------------------------------------------------
            print("\n" + "="*60)
            print(f" TRAINING PERFORMANCE REPORT {skipped_msg}")
            print("="*60)
            table_header = f"| {'Metric':<20} | {'Value':<15} | {'Unit':<10} |"
            print(table_header)
            print(f"|{'-'*22}|{'-'*17}|{'-'*12}|")
            print(f"| {'Avg Step Time':<20} | {avg_time:<15.4f} | {'sec/step':<10} |")
            print(f"| {'Min Step Time':<20} | {min_time:<15.4f} | {'sec/step':<10} |")
            print(f"| {'Max Step Time':<20} | {max_time:<15.4f} | {'sec/step':<10} |")
            print(f"| {'T95 Step Time':<20} | {t95:<15.4f} | {'sec/step':<10} |")
            print(f"| {'T99 Step Time':<20} | {t99:<15.4f} | {'sec/step':<10} |")
            print(f"| {'Est. Throughput':<20} | {avg_throughput:<15.2f} | {'samples/s':<10} |")
            print("="*60 + "\n")

            # -------------------------------------------------------
            # B. 保存到 Excel 和 CSV
            # -------------------------------------------------------
            data = [
                {"Metric": "Avg Step Time", "Value": round(avg_time, 4), "Unit": "sec/step"},
                {"Metric": "Min Step Time", "Value": round(min_time, 4), "Unit": "sec/step"},
                {"Metric": "Max Step Time", "Value": round(max_time, 4), "Unit": "sec/step"},
                {"Metric": "T95 Step Time", "Value": round(t95, 4), "Unit": "sec/step"},
                {"Metric": "T99 Step Time", "Value": round(t99, 4), "Unit": "sec/step"},
                {"Metric": "Est. Throughput", "Value": round(avg_throughput, 2), "Unit": "samples/s"},
            ]

            df = pd.DataFrame(data)

            # 定义保存路径
            result_dir = "/root/computeEvaltool/train/B2/results"
            base_filename = "training_deepseek7b_1"

            # 确保目录存在
            if not os.path.exists(result_dir):
                os.makedirs(result_dir)
                logger.info(f"Created result directory: {result_dir}")

            # 保存 CSV
            csv_path = os.path.join(result_dir, f"{base_filename}.csv")
            df.to_csv(csv_path, index=False)
            logger.info(f"Performance metrics saved to: {csv_path}")

            # 保存 Excel
            xlsx_path = os.path.join(result_dir, f"{base_filename}.xlsx")
            try:
                df.to_excel(xlsx_path, index=False)
                logger.info(f"Performance metrics saved to: {xlsx_path}")
            except Exception as e:
                logger.error(f"Error saving Excel file (ensure openpyxl is installed): {e}")

        else:
            logger.warning("Not enough steps collected to calculate T95/T99 statistics.")

if __name__ == "__main__":
    main()
