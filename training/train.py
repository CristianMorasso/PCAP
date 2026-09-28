"""
Main script for training lora/alora for jailbreak detection
"""
import os
import json
import random
from typing import Any

from datasets import Dataset, concatenate_datasets
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM, LlamaTokenizer, TrainerCallback, LlamaForCausalLM
from trl import SFTConfig, SFTTrainer

# DataCollatorForCompletionOnlyLM moved between packages/versions. Try importing
# from `trl` first and fall back to Alora's implementation when needed.
try:
    from trl import DataCollatorForCompletionOnlyLM
except Exception:
    from alora.multi_collator import DataCollatorForCompletionOnlyLM_Multi as DataCollatorForCompletionOnlyLM

# lora
from peft import get_peft_model

# alora
from alora.peft_model_alora import aLoRAPeftModelForCausalLM
from alora.config import aLoraConfig

random.seed(10)
SECURITY_PROMPT = "<|start_of_role|>jailbreak<|end_of_role|>"


def get_datasets(config) -> dict:
    """
    Load the json formatted datasets
    """
    datasets = {}


    for name, file in config["dataset_paths"].items():

        # Support both JSON (array) and JSONL (one JSON object per line) files.
        with open(file, encoding="utf-8") as f:
            try:
                raw_data = json.load(f)
            except json.JSONDecodeError:
                # Fallback: parse line-delimited JSON
                f.seek(0)
                raw_data = []
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        raw_data.append(json.loads(line))
                    except json.JSONDecodeError:
                        print(f"Warning: skipping invalid JSON line in {file}: {line[:80]}")

        data = []
        malicious_count = 0
        for sample in tqdm(raw_data):
            data.append({"prompt": sample["input"], "completion": sample['output']})

        random.shuffle(data)
        print(
            f"Dataset {name}: with {len(data)} samples {len(data)}"
        )
        datasets[name] = data

    return datasets


def formatting_prompts_func(example) -> str:
    """
    Format the data for SFT — return a single string per example.
    """
    if isinstance(example["input"], list):
        # Join multiple input/target pairs into a single string
        texts = [f"{example['input'][i]}{example['target'][i]}" for i in range(len(example['input']))]
        return "".join(texts)
    else:
        return f"{example['input']}{example['target']}"


def process_dataset(dataset: list[dict[str, str]], tokenizer):
    """
    Add the chat template and the invocation string to the inputs
    Add the tatget label and end eos token to the target
    """
    proc_datasets = []
    proc_dict = {"input": [], "target": []}

    for sample in dataset:
        prompt = sample["prompt"]
        chat = [{"role": "user", "content": prompt}]
        string = tokenizer.apply_chat_template(chat, tokenize=False, add_generation_prompt=False)
        if isinstance(string, list):
            string = "".join(string)

        # Append invocation sequence here.
        proc_dict["input"].append(string + SECURITY_PROMPT)

        # Targets (that aLoRA is meant to learn to generate)
        proc_dict["target"].append(sample["completion"] + tokenizer.eos_token)

    proc_datasets.append(Dataset.from_dict(proc_dict))
    return proc_datasets


class CheckpointAtStepsCallback(TrainerCallback):
    """Save the adapter at a specific list of training steps."""

    def __init__(self, steps: list[int], base_dir: str):
        self.steps = set(steps)
        self.base_dir = base_dir

    def on_step_end(self, args, state, control, **kwargs):
        if state.global_step in self.steps:
            out = os.path.join(self.base_dir, f"checkpoint-step-{state.global_step}")
            kwargs["model"].save_pretrained(out)
            print(f"Checkpoint saved at step {state.global_step} → {out}", flush=True)


def fine_tune(model_base, tokenizer, config_dict):
    """
    Main fine-tune loop
    """
    # logger = Logger(config_dict)
    # logger.log_configs(config_dict)

    data = get_datasets(config_dict)

    tokenizer.pad_token = tokenizer.eos_token

    dataset_train = process_dataset(data["train"], tokenizer)
    dataset_eval = process_dataset(data["eval"], tokenizer)

    merged_dataset = concatenate_datasets(dataset_train)
    dataset_eval = concatenate_datasets(dataset_eval)

    # The Alora collator expects a list of response template(s); wrap the
    # SECURITY_PROMPT in a list to be compatible with both implementations.
    collator = DataCollatorForCompletionOnlyLM([SECURITY_PROMPT], tokenizer=tokenizer)

    
    peft_config = aLoraConfig(
        r=config_dict["aLoraConfig"]["r"],
        lora_alpha=config_dict["aLoraConfig"]["lora_alpha"],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        invocation_string=config_dict["intrinsic_prompt"],
        target_modules=["q_proj", "k_proj", "v_proj"], 
    )
    response_tokens = tokenizer(SECURITY_PROMPT, return_tensors="pt", add_special_tokens=False)

    adapter_path = config_dict.get("adapter_path")

    # aLoRA's __init__ crashes with AttributeError on `modules_to_save`
    # in this environment. Always construct via get_peft_model() and then
    # load saved weights separately with load_adapter() when resuming.
    try:
        peft_model = aLoRAPeftModelForCausalLM(
            model_base, peft_config, response_token_ids=response_tokens["input_ids"]
        )
    except AttributeError as e:
        if "modules_to_save" not in str(e):
            raise
        print(f"Warning: aLoRA __init__ failed ({e}), falling back to get_peft_model.", flush=True)
        peft_model = get_peft_model(model_base, peft_config)
        try:
            peft_model.response_token_ids = response_tokens["input_ids"]
        except Exception:
            pass

    if adapter_path:
        # Load the saved adapter weights onto the already-constructed model.
        # This bypasses __init__ entirely, so the modules_to_save bug is avoided,
        # and the iter-N learned weights are correctly restored for continued training.
        print(f"Resuming from existing adapter: {adapter_path}", flush=True)
        peft_model.load_adapter(adapter_path, adapter_name="default", is_trainable=True)

    training_config = SFTConfig(
        output_dir=config_dict["save_path"],
        dataset_kwargs={"add_special_tokens": False},
        max_steps=config_dict["aLoraConfig"]["max_steps"],
        num_train_epochs=30,
        learning_rate=config_dict["aLoraConfig"]["learning_rate"],
        max_length=512,
        per_device_train_batch_size=1,
        save_strategy="no",
        gradient_accumulation_steps=8,
        logging_steps=1,
        fp16=True,
        # assistant_only_loss=True
    )

    callbacks = []
    checkpoint_steps = config_dict.get("checkpoint_steps")
    if checkpoint_steps:
        callbacks.append(CheckpointAtStepsCallback(checkpoint_steps, config_dict["save_path"]))

    trainer = SFTTrainer(
        peft_model,
        train_dataset=merged_dataset,
        eval_dataset=dataset_eval,
        args=training_config,
        processing_class=tokenizer,
        formatting_func=formatting_prompts_func,
        data_collator=collator,
        callbacks=callbacks or None,
    )

    trainer.train()
    trainer.model.save_pretrained(config_dict["save_path"])
    print(f"Final model saved to {config_dict['save_path']}", flush=True)


if __name__ == "__main__":

    configuration = {
        "lora_type": "alora",
        
        'base_model': "path/to/model/snapshot",
        
        "tokenizer_padding": "left",
        "dataset_paths": {
           
            "train": "training/data/example_ds.jsonl",
            "eval": "training/data/example_ds.jsonl",
            "test": "training/data/example_ds.jsonl",
        },
        "aLoraConfig": {"r": 32, "lora_alpha": 32, "learning_rate": 6e-5, "max_steps": 5},
        "intrinsic_prompt": SECURITY_PROMPT,
    }

    assert configuration["lora_type"] in ["alora", "lora"]
    print("Configuration: ", configuration)

    configuration["save_path"] = 'training/models'
    # Path to the iter-1 final adapter — continue from it rather than starting fresh.
    configuration["adapter_path"] = False
    # Steps at which to save intermediate checkpoints (e.g. [100, 250, 500]).
    # Set to [] or omit to skip intermediate saves.
    configuration["checkpoint_steps"] = [2,3]
    print("Save path: ", configuration["save_path"])

    t = AutoTokenizer
    tokenizer = t.from_pretrained(
    configuration["base_model"], padding_side=configuration["tokenizer_padding"], trust_remote_code=True
    )

    # tokenizer = t.from_pretrained(
    #     configuration["base_model"], padding_side=configuration["tokenizer_padding"], trust_remote_code=True
    # )
    model = AutoModelForCausalLM.from_pretrained(
        configuration["base_model"],
        device_map="mps",
    )

    fine_tune(model, tokenizer, configuration)