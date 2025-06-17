#!/usr/bin/env python
"""
One-file LoRA or QLoRA trainer using 🤗 PEFT + Accelerate.

For QLoRA add `--qlora` to enable 4-bit base-model loading.
Example:
    accelerate launch scripts/train.py \
        --model mistralai/Mistral-7B-v0.3 \
        --dataset_path data/tokenised \
        --output_dir checkpoints/h3-lora \
        --lora_r 8 --lora_alpha 16 --epochs 3 --batch_size 4 \
        --qlora
"""
import argparse, os, torch
from datasets import load_from_disk
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
)
from peft import LoraConfig, get_peft_model
from peft.tuners.lora import LoraLayer
from accelerate import Accelerator
from bitsandbytes import optim


def main(args):
    acc = Accelerator()
    tokenizer = AutoTokenizer.from_pretrained(args.model, use_fast=True)
    print("Loading base model …")
    model_kwargs = dict(
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
    )
    if args.qlora:
        model_kwargs["load_in_4bit"] = True
    model = AutoModelForCausalLM.from_pretrained(args.model, **model_kwargs)
    lora_cfg = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_cfg)
    model.print_trainable_parameters()

    # Load tokenised splits
    train_ds = load_from_disk(os.path.join(args.dataset_path, "train"))
    val_ds   = load_from_disk(os.path.join(args.dataset_path, "validation"))
    collator = DataCollatorForLanguageModeling(tokenizer, mlm=False)

    targs = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        learning_rate=2e-4,
        fp16=not args.qlora,  # 4-bit already quantised
        evaluation_strategy="epoch",
        logging_steps=20,
        save_strategy="epoch",
        gradient_accumulation_steps=args.grad_accum,
    )
    trainer = Trainer(
        model=model,
        args=targs,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        tokenizer=tokenizer,
        data_collator=collator,
    )
    trainer.train()
    acc.wait_for_everyone()
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    print(f"Finished; LoRA checkpoint written to {args.output_dir}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True)
    p.add_argument("--dataset_path", required=True)
    p.add_argument("--output_dir", required=True)
    p.add_argument("--lora_r", type=int, default=8)
    p.add_argument("--lora_alpha", type=int, default=16)
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--batch_size", type=int, default=4)
    p.add_argument("--grad_accum", type=int, default=1)
    p.add_argument("--qlora", action="store_true")
    main(p.parse_args())
