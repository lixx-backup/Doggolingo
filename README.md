# Doggolingo QLoRA

Fine-tunes `unsloth/Qwen3-1.7B` to produce Doggolingo-style responses: normal user input goes in, playful dog-style assistant text comes out.

The current project is a Kaggle notebook workflow using Unsloth, QLoRA, and TRL `SFTTrainer`.

## Repository Contents

- `doggolingo_qlora_training.ipynb` - Kaggle training notebook.
- `data/doggolingo_sft.jsonl` - combined 500-example synthetic SFT dataset.
- `data/doggolingo_sft_agent*.jsonl` - four 125-example source shards used to build the combined dataset.

## Dataset Format

The dataset uses JSONL with a `messages` column in chat format:

```jsonl
{"messages":[{"role":"user","content":"Hi there!"},{"role":"assistant","content":"Boop, hello, fren. Tail wag engaged."}]}
```

This trains the model as a Doggolingo-style chat responder, not only as a strict English-to-Doggolingo translator.

## Training Setup

Default notebook settings:

- Base model: `unsloth/Qwen3-1.7B`
- Quantization: 4-bit QLoRA
- LoRA rank: `16`
- LoRA alpha: `16`
- LoRA dropout: `0.0`
- Target modules: `q_proj`, `k_proj`, `v_proj`, `o_proj`, `gate_proj`, `up_proj`, `down_proj`
- Sequence length: `1024`
- Epochs: `3`
- Batch size: `1`
- Gradient accumulation: `8`
- Learning rate: `2e-4`
- Validation split: `5%`

The notebook saves the LoRA adapter to:

```text
/kaggle/working/doggolingo-qlora-adapter
```

## Running On Kaggle

1. Create or open a Kaggle notebook.
2. Enable GPU acceleration.
3. Attach the dataset or upload the JSONL file.
4. Install dependencies:

```python
!pip install -U unsloth trl datasets
```

5. Run the cells in `doggolingo_qlora_training.ipynb`.

The notebook imports Unsloth before TRL:

```python
from unsloth import FastLanguageModel, is_bfloat16_supported
from trl import SFTConfig, SFTTrainer
```

Keep this import order. Importing TRL first can cause tokenizer/config issues with this stack.

## Testing

After training, the notebook defines `ask_doggo()` for quick generation:

```python
ask_doggo("What do you think is happening when I leave the house?")
```

Example output from the notebook:

```text
It's a whole world of sniff-sounds, doggo tail wag possibilities, and one brave little foot step away from the familiar floor.
```

## Current Status

This repo currently contains the training notebook, the synthetic dataset, and project notes. It does not yet include a standalone inference app, Hugging Face Hub upload script, model card, or deployment configuration.
