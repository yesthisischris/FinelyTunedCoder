# FinelyTunedCoder

This repository contains resources for fine-tuning large language models (LLMs) with up-to-date syntax for modern Python packages (e.g., `h3` v4 and `polars`). It is designed around [Axolotl](https://github.com/OpenAccess-AI-Collective/axolotl) for LoRA/QLoRA training but can be adapted for PEFT + Accelerate workflows.

## Repository Structure

```
├── data/                     # Training and evaluation datasets
│   └── h3_api_examples.json # Curated examples of modern H3 usage
├── configs/                  # Axolotl configuration files
│   └── h3_lora_qlora.yml
├── scripts/                  # Helper scripts for dataset prep and model training
│   ├── prepare_dataset.py
│   ├── train.py              # Optional reference training script
│   └── merge_and_pack.py     # Merge LoRA weights and export
├── eval/                     # Evaluation utilities
│   └── run_lm_eval.sh        # Uses lm-eval-harness
├── requirements.txt          # Python dependencies
└── README.md                 # Project documentation
```

### Getting Started
1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

2. Prepare your dataset and create train/validation splits:
   ```bash
   python scripts/prepare_dataset.py \
       --input data/h3_api_examples.json \
       --out_dir data/tokenised
   ```

3. Fine‑tune the base model with LoRA or QLoRA:
   ```bash
   accelerate launch scripts/train.py \
       --model mistralai/Mistral-7B-v0.3 \
       --dataset_path data/tokenised \
       --output_dir checkpoints/h3-lora \
       --qlora
   ```

4. Merge the LoRA weights back into the base model and save in `merged/`:
   ```bash
   python scripts/merge_and_pack.py \
       --base_model mistralai/Mistral-7B-v0.3 \
       --lora_dir checkpoints/h3-lora \
       --out_dir merged/h3-v4
   ```

5. Evaluate the merged model on a code benchmark:
   ```bash
   bash eval/run_lm_eval.sh merged/h3-v4
   ```

## Contributing
PRs that improve dataset quality, training configs, or documentation are welcome. Please open an issue first to discuss major changes.

