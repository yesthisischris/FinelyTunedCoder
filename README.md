# FinelyTunedCoder

This repository contains resources for fine-tuning large language models (LLMs) with up-to-date syntax for modern Python packages (e.g., `h3` v4 and `polars`). It is designed around [Axolotl](https://github.com/OpenAccess-AI-Collective/axolotl) for LoRA/QLoRA training but can be adapted for PEFT + Accelerate workflows.

## Repository Structure

```
├── data/                     # Training and evaluation datasets
│   └── h3_api_examples.jsonl # Curated examples of modern H3 usage
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

2. Prepare your dataset by placing JSON Lines files in `data/` and running:
   ```bash
   python scripts/prepare_dataset.py --input data/h3_api_examples.jsonl
   ```

3. Train with Axolotl using the configuration in `configs/h3_lora_qlora.yml`:
   ```bash
   axolotl train configs/h3_lora_qlora.yml
   ```

4. After training, merge LoRA weights and export formats (GGUF or Safetensors):
   ```bash
   python scripts/merge_and_pack.py --config configs/h3_lora_qlora.yml
   ```

5. Evaluate the model on the Code-Eval subset:
   ```bash
   bash eval/run_lm_eval.sh
   ```

## Contributing
PRs that improve dataset quality, training configs, or documentation are welcome. Please open an issue first to discuss major changes.

