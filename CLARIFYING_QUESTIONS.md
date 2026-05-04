# Clarifying Questions And Recommendations

This document records the current project decisions, explains the areas marked as unfamiliar, and lists the remaining confirmations needed before implementation starts.

## What This Project Does

This project will be a single fine-tuning script for Google Colab or Kaggle free GPU environments. It will use supervised fine-tuning (SFT) to adapt an open-source Hugging Face causal language model to produce Doggolingo-style responses.

The intended training stack is:

- Hugging Face model as the base model.
- Unsloth for efficient model loading, 4-bit quantization, and LoRA setup.
- QLoRA so the base model can be loaded in 4-bit and only small adapter weights are trained.
- TRL `SFTTrainer` for supervised fine-tuning.
- A Doggolingo corpus or generated Doggolingo-style instruction dataset.

The project will not include a README, package structure, inference script, evaluation script, Hugging Face Hub publishing, or model card unless you later request them.

## Current Decisions From User

| Area               | Decision                            |
| ------------------ | ----------------------------------- |
| Project shape      | Single training script              |
| Runtime            | Google Colab or Kaggle free version |
| Training only      | Yes                                 |
| README/setup docs  | No                                  |
| Training framework | Unsloth                             |
| Hub publishing     | Not at this stage                   |
| Existing corpus    | None                                |
| Validation         | Not decided yet                     |

## Base Model Recommendation

You asked for model suggestions.

Recommended default:

- `unsloth/Qwen3-1.7B`

Why:

- Small enough for Colab/Kaggle free GPUs.
- Stronger than many older sub-2B models for instruction following.
- Available in an Unsloth-compatible Hugging Face repo.
- Good fit for style fine-tuning, where the goal is tone and wording rather than deep domain knowledge.

Good alternatives:

- `Qwen/Qwen3-1.7B`: use this if you prefer the original Qwen repo instead of the Unsloth mirror.
- `Qwen/Qwen3-4B`: better quality, but higher memory pressure on free GPUs.
- `google/gemma-3-1b-it`: very small and lightweight, but its license/terms should be checked carefully before use.

Suggested decision:

- Start with `unsloth/Qwen3-1.7B`.
- Keep `model_name` configurable at the top of the script so we can swap to a larger model later.

Answer: go with unsloth/Qwen3-1.7B

## Doggolingo Corpus

Your answer: no existing corpus; find one online.

Finding:

- I did not find a clear, ready-to-use public Hugging Face dataset specifically named Doggolingo or DoggoLingo.
- I found reference material explaining Doggolingo terms and style, but reference articles are not automatically safe or sufficient as a training corpus.

Recommendation:

- Build a small project-local JSONL dataset from generated instruction-response examples using a Doggolingo vocabulary and style guide.
- Keep the examples original, short, and clearly synthetic.
- Use online references only to define vocabulary and style patterns, not to copy large blocks of text.

Why:

- This avoids copyright and licensing problems from scraping articles or social media posts.
- For a style fine-tune, clean paired examples are more useful than messy raw internet text.
- A JSONL file is easy to inspect and easy for `datasets.load_dataset("json", ...)` to load.

Suggested dataset format:

```jsonl
{"messages":[{"role":"user","content":"Rewrite this in Doggolingo: The puppy is excited for dinner."},{"role":"assistant","content":"Henlo, tiny pupper is doin a big excite for din-dins."}]}
{"messages":[{"role":"user","content":"Write a short happy message from a dog waiting at the door."},{"role":"assistant","content":"Am good boi at door. Much waits. Very tail wags. Pls hooman return soon."}]}
```

Suggested decision:

- Use JSONL with a `messages` column.
- Start with a small synthetic corpus in `data/doggolingo_sft.jsonl`.
- Include a script option to load a Hugging Face dataset later if a suitable licensed dataset is found.

Answer: will provided later through using a large LLM to generate

## SFTTrainer And Unsloth

Your answer: "I'm not familiar with both."

Explanation:

- SFT means supervised fine-tuning. The model is trained on examples of desired behavior, such as user prompt plus ideal assistant response.
- TRL's `SFTTrainer` is Hugging Face's trainer for this exact post-training workflow.
- Unsloth is an efficiency layer that makes model loading, 4-bit training, and LoRA setup easier and faster on limited GPUs.

Recommendation:

- Use Unsloth for model loading and LoRA setup.
- Use TRL `SFTTrainer` for the training loop.

Why:

- This is the common pairing: Unsloth prepares the model efficiently, and `SFTTrainer` handles dataset formatting, batching, logging, checkpoints, and training.
- Hugging Face documents Unsloth as compatible with `SFTTrainer`.
- It keeps the script smaller than manually writing the whole PyTorch training loop.

Suggested decision:

- Use both Unsloth and TRL `SFTTrainer`.

Answer: use both

## Hardware And Memory

Your answer: Colab or Kaggle free version.

Explanation:

- Free Colab and Kaggle usually provide a single NVIDIA GPU when available.
- The available GPU can vary, commonly around 16 GB VRAM on T4-class instances.
- QLoRA matters because it loads the base model in 4-bit, reducing memory use enough for small/medium models.

Recommendation:

- Target a conservative 16 GB VRAM default.
- Use `load_in_4bit=True`.
- Use batch size 1 or 2 with gradient accumulation.
- Use `max_seq_length=1024` initially, then increase to 2048 if memory allows.

Why:

- Free GPUs are inconsistent.
- A conservative default is more likely to run without out-of-memory errors.
- Doggolingo examples are short, so 1024 tokens is enough for the first version.

Suggested decision:

- Default to 16 GB VRAM assumptions.

## QLoRA Settings

Your answer: "I'm not familiar with this."

Explanation:

- LoRA trains small adapter matrices instead of changing all base model weights.
- QLoRA combines LoRA with 4-bit quantization of the base model.
- `r` is the adapter rank. Higher can learn more but uses more memory.
- `lora_alpha` scales the adapter update.
- `lora_dropout` regularizes LoRA, but Unsloth commonly optimizes for `0`.
- Target modules are the model layers where LoRA adapters are attached.

Recommended defaults:

```python
r = 16
lora_alpha = 16
lora_dropout = 0
bias = "none"
target_modules = [
    "q_proj", "k_proj", "v_proj", "o_proj",
    "gate_proj", "up_proj", "down_proj",
]
use_gradient_checkpointing = True
```

Why:

- `r=16` is a practical first setting for style tuning.
- `lora_alpha=16` matches the rank and is a stable default.
- `lora_dropout=0` and `bias="none"` match Unsloth's optimized pattern.
- The target modules cover attention and MLP projection layers commonly used in Qwen/Llama-like models.

Suggested decision:

- Use the defaults above unless the first run underfits or runs out of memory.

Answer: yes

## Speed, Memory, Or Quality

This question was unanswered.

Recommendation:

- Prioritize low memory usage first, then training speed, then final quality.

Why:

- Colab/Kaggle free GPU availability is the main constraint.
- A style fine-tune can often work well with a smaller model and careful data.
- We can improve quality later by increasing dataset size, epochs, sequence length, model size, or LoRA rank.

Suggested decision:

- Low memory defaults.

Answer: the current setting is already low memory?

## Saving Adapters Or Merged Model

Your answer: "I'm not familiar with this."

Explanation:

- A LoRA adapter is the small trained delta. It is cheap to save, upload, and retrain.
- A merged model combines the base model plus adapter into one full model. It is larger and easier to run in some deployment setups.

Recommendation:

- Save only LoRA adapters for this stage.
- Do not merge by default.

Why:

- Adapter saving is faster and uses less disk space.
- You are not publishing to Hugging Face Hub yet.
- Keeping the adapter separate makes experimentation easier.

Suggested decision:

- Save adapters to `outputs/doggolingo-qlora-adapter`.

Answer: yes

## Output Location

This question was unanswered.

Recommendation:

- In Colab: `/content/doggolingo-qlora-adapter`
- In Kaggle: `/kaggle/working/doggolingo-qlora-adapter`
- In repo/local fallback: `outputs/doggolingo-qlora-adapter`

Why:

- These paths match the writable working areas in the target notebook environments.

Suggested decision:

- Script should auto-detect Colab/Kaggle path and otherwise use local `outputs/`.

Answer: ok

## Hugging Face Hub Publishing

Your answer: not at this stage.

Suggested decision:

- Do not implement `push_to_hub` in the first version.
- Leave token handling out of scope.

## Model Card

Your answer: "what is this?"

Explanation:

- A model card is the `README.md` that accompanies a model on Hugging Face Hub.
- It documents the model, intended use, limitations, datasets, training settings, and evaluation.

Recommendation:

- Do not include model card generation now.

Why:

- You are not publishing to Hugging Face Hub yet.
- A model card becomes useful when sharing the adapter or merged model publicly.

Suggested decision:

- Skip model card for this stage.

Answer: skip for now

## Validation

Your answer: not decided yet at this stage.

Explanation:

- Validation can mean automated loss on a held-out dataset, or qualitative prompts checked by a human.
- For a style fine-tune, qualitative checks are usually more informative than a generic metric.

Recommendation:

- Include a tiny validation split from the JSONL data.
- Log training loss only.
- Add 3 to 5 fixed sample prompts at the end of training if you later decide an inference check is acceptable.

Why:

- A validation split catches obvious overfitting or broken formatting.
- Plain console logs are enough for a first Colab/Kaggle script.
- Weights & Biases and TensorBoard add setup complexity that is not needed yet.

Suggested decision:

- Use plain console logging.
- Use a small validation split if the dataset has enough examples.

## Environment

Your answer: "I'm not familiar with this."

Explanation:

- Python version controls language/runtime compatibility.
- Dependency management controls how libraries are installed.
- CUDA/Linux matters because Unsloth and 4-bit GPU training rely on accelerated GPU libraries.

Recommendation:

- Target Colab/Kaggle's default Python runtime.
- Use notebook/script install commands instead of `requirements.txt` for the first version.
- Assume Linux plus NVIDIA CUDA GPU.
- Install with `pip install unsloth trl datasets`.

Why:

- Colab and Kaggle already provide Python, CUDA, and notebook execution.
- A single install cell/script section is simpler than packaging for this use case.
- Unsloth is designed around GPU training and documents support for Colab and Kaggle.

Suggested decision:

- No `requirements.txt` or `pyproject.toml` yet.
- Training script includes a commented install command for Colab/Kaggle.

Answer: yes

## Remaining Confirmations Needed

Please confirm or edit these before implementation starts:

1. Use `unsloth/Qwen3-1.7B` as the default base model?
2. Use synthetic JSONL Doggolingo instruction examples as the initial corpus?
3. Save only LoRA adapters, not a merged full model?
4. Use low-memory defaults for free Colab/Kaggle GPUs?
5. Skip README, model card, inference script, and Hub publishing for version 1?

## Sources Checked

- Hugging Face TRL `SFTTrainer` documentation: https://huggingface.co/docs/trl/main/sft_trainer
- Hugging Face TRL Unsloth integration: https://huggingface.co/docs/trl/unsloth_integration
- Unsloth installation documentation: https://docs.unsloth.ai/get-started/install-and-update
- Hugging Face model card documentation: https://huggingface.co/docs/hub/en/model-cards
- Qwen3 1.7B model card: https://huggingface.co/Qwen/Qwen3-1.7B
- Unsloth Qwen3 1.7B model card: https://huggingface.co/unsloth/Qwen3-1.7B
- Gemma 3 1B model card: https://huggingface.co/google/gemma-3-1b-it
- Doggolingo reference overview: https://www.dictionary.com/culture/slang/doggo
- Doggolingo style/reference overview: https://knowyourmeme.com/memes/doggolingo-doggo-speak
