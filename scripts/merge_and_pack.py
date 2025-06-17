#!/usr/bin/env python
"""
Merge LoRA weights back into the base model and (optionally) export
GGUF for llama.cpp or a safetensors file for vLLM.

Example:
    python scripts/merge_and_pack.py \
        --base_model mistralai/Mistral-7B-v0.3 \
        --lora_dir checkpoints/h3-lora \
        --out_dir merged/h3-v4
"""
import argparse, shutil, os
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
from pathlib import Path


def main(args):
    print("Loading base …")
    base = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        torch_dtype="auto",
    )
    print("Attaching LoRA …")
    merged = PeftModel.from_pretrained(base, args.lora_dir)
    merged = merged.merge_and_unload()  # adds deltas into base weights

    Path(args.out_dir).mkdir(parents=True, exist_ok=True)
    print("Saving merged safetensors …")
    merged.save_pretrained(args.out_dir, safe_serialization=True)
    AutoTokenizer.from_pretrained(args.base_model).save_pretrained(args.out_dir)

    # Optional: GGUF export (needs llama-cpp-python)
    if args.gguf:
        from llama_cpp import convert
        convert(
            f"{args.out_dir}/model.safetensors",
            f"{args.out_dir}/model.gguf",
            dtype="Q4_K_M",
        )
        print("Wrote GGUF file for llama.cpp")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--base_model", required=True)
    p.add_argument("--lora_dir", required=True)
    p.add_argument("--out_dir", required=True)
    p.add_argument("--gguf", action="store_true")
    main(p.parse_args())
