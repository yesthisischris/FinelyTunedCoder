#!/usr/bin/env python
"""Merge LoRA weights with the base model and export."""
import argparse


def main():
    parser = argparse.ArgumentParser(description="Merge LoRA and export")
    parser.add_argument("--config", required=True, help="Training config path")
    args = parser.parse_args()

    # Placeholder: implement merging logic
    print(f"Merging with config {args.config}")


if __name__ == "__main__":
    main()
