#!/usr/bin/env python
"""Tokenize and split dataset for training."""
import argparse


def main():
    parser = argparse.ArgumentParser(description="Prepare dataset")
    parser.add_argument("--input", required=True, help="Path to JSONL dataset")
    parser.add_argument("--output", default="data/processed.jsonl", help="Output file")
    args = parser.parse_args()

    # Placeholder: implement tokenization and splitting
    print(f"Processing {args.input} -> {args.output}")


if __name__ == "__main__":
    main()
