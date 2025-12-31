import os
import argparse
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    DataCollatorForLanguageModeling,
)

from transformers import TrainerCallback

import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

class TrainingMonitorCallback(TrainerCallback):
    def on_log(self, args, state, control, logs=None, **kwargs):
        if state.is_local_process_zero:
            print(f"[Step {state.global_step}] Logs: {logs}")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name_or_path", type=str, default="/root/computeEvaltool/train/deepseek-llm-7b-chat")
    parser.add_argument("--output_dir", type=str, default="./output")
    parser.add_argument("--train_file", type=str, default="/root/computeEvaltool/train/B2/data/openr1_math_220k.jsonl")
    parser.add_argument("--max_train_samples", type=int, default=2048)
    parser.add_argument("--block_size", type=int, default=1024)
    parser.add_argument("--deepspeed", type=str, default=None)
    parser.add_argument("--local_rank", type=int, default=-1, help="local rank passed by deepspeed/torchrun")
    return parser.parse_args()


def extract_text(example):
    # 自动识别 JSONL 格式的字段
    if "text" in example:
        return example["text"]
    elif "prompt" in example and "response" in example:
        return example["prompt"] + example["response"]
    elif "instruction" in example and "output" in example:
        return example["instruction"] + "\n" + example.get("input", "") + "\n" + example["output"]
    else:
        raise ValueError(f"Cannot find usable text field in example: {example}")


def main():

    args = parse_args()

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

    # Load JSONL
    dataset = load_dataset("json", data_files=args.train_file)

    # Add unified text field
    dataset = dataset.map(
        lambda x: {"text": extract_text(x)},
        remove_columns=dataset["train"].column_names
    )

    # Tokenization
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
        dataloader_num_workers=128,
        weight_decay=0.01,
        report_to="none",
        local_rank=args.local_rank,
        dataloader_drop_last=True,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset["train"],
        data_collator=data_collator,
        processing_class=tokenizer,
        callbacks=[TrainingMonitorCallback()],
    )

    trainer.train()

    trainer.save_model(args.output_dir)


if __name__ == "__main__":
    main()