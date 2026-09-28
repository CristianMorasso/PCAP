# Training

Fine-tuning script for training a LoRA / aLoRA adapter on top of a causal LM for jailbreak detection.

## Files

| Path | Description |
|---|---|
| `train.py` | Main training script |
| `requirments_train.txt` | Pinned Python dependencies |
| `data/example_ds.jsonl` | **Mock dataset** — two placeholder samples used for smoke-testing the pipeline (see note below) |

## Mock dataset

`data/example_ds.jsonl` is a **mock dataset** and is not suitable for real training.
It contains only two trivial benign exchanges (`"Hello"` / `"Hi"`) and exists solely to let you verify that the data-loading, tokenisation, and training loop run end-to-end without errors.
Replace it with your real dataset before running any meaningful experiment.

Each record is expected to follow this schema:

```json
{"input": "<user prompt>", "output": "<model response>", "type": "benign|malicious", "goal": null, "strategies": null}
```

## Installation

```bash
pip install -r requirments_train.txt
```

## Usage

Edit the `configuration` block near the bottom of `train.py`, then run:

```bash
python training/train.py
```

Key configuration options:

| Key | Description |
|---|---|
| `base_model` | Path or HuggingFace hub ID of the base causal LM |
| `dataset_paths` | Dict with `train`, `eval`, and `test` JSONL file paths |
| `save_path` | Directory where the final adapter and checkpoints are written |
| `adapter_path` | Path to an existing adapter to resume from (`False` to start fresh) |
| `aLoraConfig` | LoRA hyper-parameters: `r`, `lora_alpha`, `learning_rate`, `max_steps` |
| `checkpoint_steps` | List of training steps at which to save an intermediate checkpoint, e.g. `[100, 250, 500]`. Empty list disables intermediate saves. |

## Output

- **Final adapter** — saved to `save_path/` at the end of training.
- **Intermediate checkpoints** — saved to `save_path/checkpoint-step-{N}/` for each step listed in `checkpoint_steps`.
