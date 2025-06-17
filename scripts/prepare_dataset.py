#!/usr/bin/env python
"""
Tokenise h3_api_examples.json and create train/valid splits
ready for supervised fine-tuning (SFT) or QLoRA.

Invoked as:
    python scripts/prepare_dataset.py \
           --input data/h3_api_examples.json \
           --out_dir data/tokenised \
           --model mistralai/Mistral-7B-v0.3 \
           --valid_frac 0.05
"""
import argparse, json, pathlib, random
from datasets import Dataset, load_from_disk
from transformers import AutoTokenizer
random.seed(42)


def load_json(input_path):
    with open(input_path) as f:
        rows = json.load(f)
    return rows


def build_prompt(example):
    """Merge (system, user, assistant) into Alpaca-style prompt/answer."""
    system = example["system"]
    user = example["user"]
    assistant = example["assistant"]
    prompt = f"<s>[SYSTEM] {system}\n[USER] {user}\n[ASSISTANT]"
    return {"prompt": prompt, "answer": assistant}


def tokenize(ds, tokenizer):
    def _tok(batch):
        concat = [f"{p} {a}</s>" for p, a in zip(batch["prompt"], batch["answer"])]
        out = tokenizer(
            concat,
            truncation=True,
            padding=False,
            max_length=2048,
        )
        out["labels"] = out["input_ids"].copy()
        return out
    return ds.map(_tok, batched=True, remove_columns=ds.column_names)


def main(args):
    rows = load_json(args.input)
    ds = Dataset.from_list(rows).map(build_prompt)
    ds = ds.train_test_split(test_size=args.valid_frac, seed=42)
    tokenizer = AutoTokenizer.from_pretrained(args.model, use_fast=True)
    tokenised = {
        "train": tokenize(ds["train"], tokenizer),
        "validation": tokenize(ds["test"], tokenizer),
    }
    out_dir = pathlib.Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    for split, d in tokenised.items():
        d.save_to_disk(out_dir / split)
    print(f"Saved tokenised dataset to {out_dir}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--input", required=True)
    p.add_argument("--out_dir", required=True)
    p.add_argument("--model", default="mistralai/Mistral-7B-v0.3")
    p.add_argument("--valid_frac", type=float, default=0.05)
    main(p.parse_args())
