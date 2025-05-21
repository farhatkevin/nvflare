#
# Copyright (c) 2024, NVIDIA CORPORATION.  All rights reserved.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import time
import argparse
import copy
import math
import os
import random
from itertools import chain
import glob
from dolma.core.paths import cached_path      # NEW
from tqdm import tqdm         

import datasets
import numpy as np
import torch
import wandb
from peft import (
    LoraConfig,
    get_peft_model,
    get_peft_model_state_dict,
    set_peft_model_state_dict,
    utils,
)
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    DataCollatorForLanguageModeling,
    Trainer,
    TrainingArguments,
    trainer_utils,
)

import nvflare.client as flare


import logging
logging.getLogger("torch._dynamo").setLevel(logging.ERROR)
logging.getLogger("torch._inductor").setLevel(logging.ERROR)


# ──────────────────────────────────────────────────────────────────────────────
# deterministic seeds
torch.manual_seed(0)
random.seed(0)
np.random.seed(0)

# ──────────────────────────────────────────────────────────────────────────────
# argument parser
def parse_token_count(s):
    """Parse a string like '50B', '10M', '1000000' into an integer token count."""
    s = s.strip().lower()
    if s.endswith("b"):
        return int(float(s[:-1]) * 1_000_000_000)
    elif s.endswith("m"):
        return int(float(s[:-1]) * 1_000_000)
    elif s.endswith("k"):
        return int(float(s[:-1]) * 1_000)
    else:
        return int(float(s))

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name_or_path", type=str,
                        default="meta-llama/llama-3.2-1b")
    parser.add_argument("--data_path_train", type=str,
                        default="./dataset/text/training.txt")
    parser.add_argument("--data_path_valid", type=str,
                        default="./dataset/text/validation.txt")
    parser.add_argument("--output_path", type=str,
                        default="./workspace_federated/llama-3.2-1b-pretrain")
    parser.add_argument("--train_mode", type=str, choices=["pretrain", "peft"],
                        default="pretrain",
                        help="'peft' turns on LoRA; anything else trains full weights")
    parser.add_argument("--message_mode", type=str, choices=["numpy", "tensor"],
                        default="numpy")
    parser.add_argument("--local_epoch", type=int, default=1)
    parser.add_argument("--clean_up", type=int, default=0)
    parser.add_argument("--block_size", type=int, default=512)
    parser.add_argument("--gpu", type=str, default="0")
    parser.add_argument("--max_tokens", type=str, default=None,
                        help="Maximum number of tokens to use for this client (e.g., 50B, 10M, 1000000).")
    return parser.parse_args()


# ──────────────────────────────────────────────────────────────────────────────
# dataset helpers
def tokenize_function(examples, tokenizer):
    return tokenizer(examples["text"])

def group_tokens(examples, block_size):
    """Pack token IDs into fixed-length blocks for causal-LM pre-training."""
    # Check the structure of examples["text"] to handle different formats
    if isinstance(examples["text"][0], int):
        # If text contains direct integers (tokens), no need to flatten
        all_tokens = examples["text"]
    else:
        # If text contains lists of tokens, flatten with chain
        all_tokens = list(chain.from_iterable(examples["text"]))
    
    # Calculate total length that fits evenly into blocks
    total_len = (len(all_tokens) // block_size) * block_size
    
    # Create blocks of fixed size
    result = {
        "input_ids": [all_tokens[i:i + block_size] for i in range(0, total_len, block_size)],
    }
    
    # For causal LM, labels are the same as input_ids
    result["labels"] = result["input_ids"].copy()
    
    return result

# ──────────────────────────────────────────────────────────────────────────────

def load_npy_memmap_data_as_dataset(file_path: str) -> datasets.Dataset:
    """Loads a .npy file containing text data and returns a datasets.Dataset."""
    if not os.path.exists(file_path):
        print(f"Warning: Data file not found: {file_path}. Returning empty dataset.")
        # Return an empty dataset with the expected "text" column
        return datasets.Dataset.from_dict({"text": []})

    print(f"Loading data from {file_path}...")
    try:
        # allow_pickle=True is often necessary for arrays of strings
        size = os.path.getsize(file_path)
        data_array = np.memmap(file_path, dtype='uint32', mode='r', shape=(size // 4,))
        print("data_array shape:", data_array.shape)
        print("data_array dtype:", data_array.dtype)
        print("data_array size:", data_array.size)
    except Exception as e:
        print(f"Error loading {file_path}: {e}. Returning empty dataset.")
        return datasets.Dataset.from_dict({"text": []})

    # Assuming data_array is a 1D array of strings.
    # If it's a 0-dim array containing a list (np.save might do this for lists)
    if data_array.ndim == 0 and isinstance(data_array.item(), list):
        text_list = data_array.item()
    elif data_array.ndim == 1:
        text_list = data_array.tolist()
    else:
        raise ValueError(
            f"Unsupported .npy file structure in {file_path}. "
            "Expected a 1D NumPy array of strings or a 0D array containing a list of strings."
        )

    print(len(text_list), "len text_list")
    text_list = text_list[:1_000_000] # Limit to first 100 items for testing
    print(len(text_list), "len text_list")

    # The Dataset needs a dictionary where keys are column names
    return datasets.Dataset.from_dict({"text": text_list})


def load_all_npy_in_folder_as_dataset(folder_path: str, max_tokens: int = None) -> datasets.Dataset:
    """Loads all .npy files in a folder and concatenates them into a single HuggingFace Dataset, up to max_tokens."""
    if not os.path.isdir(folder_path):
        print(f"Provided path {folder_path} is not a directory. Falling back to single file loader.")
        return load_npy_memmap_data_as_dataset(folder_path)
    npy_files = sorted(glob.glob(os.path.join(folder_path, "*.npy")))
    if not npy_files:
        print(f"No .npy files found in {folder_path}. Returning empty dataset.")
        return datasets.Dataset.from_dict({"text": []})
    all_texts = []
    total_tokens_loaded = 0
    for npy_file in npy_files:
        print(f"Loading data from {npy_file}...")
        try:
            size = os.path.getsize(npy_file)
            data_array = np.memmap(npy_file, dtype='uint32', mode='r', shape=(size // 4,))
            if data_array.ndim == 0 and isinstance(data_array.item(), list):
                text_list = data_array.item()
            elif data_array.ndim == 1:
                text_list = data_array.tolist()
            else:
                raise ValueError(
                    f"Unsupported .npy file structure in {npy_file}. "
                    "Expected a 1D NumPy array of strings or a 0D array containing a list of strings."
                )
            print(len(text_list), "len text_list")
            # Optionally limit for testing:
            # text_list = text_list[:10_000_000]
            if max_tokens is not None:
                tokens_remaining = max_tokens - total_tokens_loaded
                if tokens_remaining <= 0:
                    break
                text_list = text_list[:tokens_remaining]
            all_texts.extend(text_list)
            total_tokens_loaded += len(text_list)
            if max_tokens is not None and total_tokens_loaded >= max_tokens:
                break
        except Exception as e:
            print(f"Error loading {npy_file}: {e}. Skipping this file.")
    print(f"Total loaded tokens: {len(all_texts)}")
    return datasets.Dataset.from_dict({"text": all_texts})


class CustomTrainer(Trainer):
    def _load_rng_state(self, checkpoint):
        """Override to load RNG state with weights_only=False"""
        # Hardcode the trainer state filename - it's always "trainer_state.json"
        import json
        rng_file = os.path.join(checkpoint, "trainer_state.json")
        if not os.path.isfile(rng_file):
            print(f"No RNG state found at {rng_file}")
            return
        
        try:
            # Load JSON file instead of using torch.load
            with open(rng_file, 'r') as f:
                checkpoint_rng_state = json.load(f)
            
            # The rest follows the original implementation
            if checkpoint_rng_state is not None and "random_states" in checkpoint_rng_state:
                random_states = checkpoint_rng_state["random_states"]
                if "python" in random_states and random_states["python"] is not None:
                    random.setstate(random_states["python"])
                if "numpy" in random_states and random_states["numpy"] is not None:
                    np.random.set_state(random_states["numpy"])
                if "cpu_rng_state" in random_states:
                    torch.set_rng_state(random_states["cpu_rng_state"])
                if "cuda_rng_state" in random_states and torch.cuda.is_available():
                    devices = list(range(torch.cuda.device_count()))
                    if "cuda_rng_state_all" in random_states:
                        for i, device in enumerate(devices):
                            if i < len(random_states["cuda_rng_state_all"]):
                                with torch.cuda.device(device):
                                    torch.cuda.set_rng_state(random_states["cuda_rng_state_all"][i])
                    elif "cuda_rng_state" in random_states:
                        torch.cuda.set_rng_state(random_states["cuda_rng_state"])
            print("RNG state successfully loaded")
        except Exception as e:
            print(f"Warning: Failed to load RNG state: {e}")
            print("Continuing training with current RNG state.")

def main():
    args = parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    # Parse max_tokens argument to int if needed
    max_tokens = None
    if args.max_tokens is not None:
        max_tokens = parse_token_count(args.max_tokens)
        print(f"Limiting to {max_tokens:,} tokens for this client.")

    # ── tokenizer ────────────────────────────────────────────────────────────
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    pad_id = tokenizer.pad_token_id

    # ── raw datasets ─────────────────────────────────────────────────────────
    print(f"Attempting to load training data from: {args.data_path_train}")
    # If data_path_train is a directory, load all .npy files in it
    if os.path.isdir(args.data_path_train):
        raw_train_ds = load_all_npy_in_folder_as_dataset(args.data_path_train, max_tokens=max_tokens)
    else:
        raw_train_ds = load_npy_memmap_data_as_dataset(args.data_path_train)
        if max_tokens is not None and len(raw_train_ds) > max_tokens:
            raw_train_ds = datasets.Dataset.from_dict({"text": raw_train_ds["text"][:max_tokens]})

    # Handle validation data:
    # If a specific validation file is provided and it's different from the training file
    if args.data_path_valid and os.path.exists(args.data_path_valid) and args.data_path_valid != args.data_path_train:
        print(f"Attempting to load validation data from: {args.data_path_valid}")
        if os.path.isdir(args.data_path_valid):
            raw_valid_ds = load_all_npy_in_folder_as_dataset(args.data_path_valid)
        else:
            raw_valid_ds = load_npy_memmap_data_as_dataset(args.data_path_valid)
        raw_ds = datasets.DatasetDict({"train": raw_train_ds, "validation": raw_valid_ds})
    # If validation path is same as train, or if train is long enough to split
    elif len(raw_train_ds) > 1: # Check if there's enough data to split
        # If the launch script passed the same path for train and valid,
        # we can use the whole dataset for both training and validation (common in pretraining for perplexity check)
        # or split it. Let's choose to split for a more "true" validation.
        print(f"Validation data path is same as train or not specified separately. Splitting training data (from {args.data_path_train}) for validation (1% test size).")
        # Ensure a minimal number of samples for splitting
        test_size = 0.01
        if len(raw_train_ds) * test_size < 1: # Ensure at least one sample in test set
            if len(raw_train_ds) > 1: # If we have at least 2, take 1 for test
                test_size = 1 / len(raw_train_ds)
            else: # cannot split
                print("Not enough data to create a validation split. Using training data for validation.")
                raw_ds = datasets.DatasetDict({"train": raw_train_ds, "validation": raw_train_ds})

        if 'raw_ds' not in locals(): # If not already set by the cannot split case
            split_dataset = raw_train_ds.train_test_split(test_size=test_size, seed=42)
            raw_ds = datasets.DatasetDict({"train": split_dataset["train"], "validation": split_dataset["test"]})
    else: # Not enough data to split, use train data for validation (or empty if train is also empty)
        print("Training data is too small to split for validation. Using training data as validation data.")
        raw_ds = datasets.DatasetDict({"train": raw_train_ds, "validation": raw_train_ds})
    # --- END MODIFIED DATA LOADING ---

    print(f"Dataset size: training {len(raw_ds['train'])}, "
        f"validation {len(raw_ds['validation'])}")

    print("Raw dataset structure after loading:", raw_ds)

    # Ensure datasets are not empty before proceeding
    if len(raw_ds["train"]) == 0:
        raise ValueError("Training dataset is empty. Please check your data path and .npy file content.")
    # Validation can be empty if not enough data to split, but Trainer might complain.
    # The evaluate function handles this by returning NaN if eval_dataset is empty.

    packed_cache_path = os.path.join(args.output_path, "packed_train.arrow")
    if os.path.exists(packed_cache_path):
        print(f"Loading packed dataset from cache: {packed_cache_path}")
        lm_ds = datasets.load_from_disk(packed_cache_path)
    else:
        lm_ds = raw_ds.map(
            group_tokens,
            fn_kwargs={"block_size": args.block_size},
            batched=True,
            remove_columns=["text"],
            desc=f"Packing into {args.block_size}-token blocks",
        )
        lm_ds.save_to_disk(packed_cache_path)

    # Ensure tokenized datasets are not empty
    if len(lm_ds["train"]) == 0:
        # This can happen if all texts are shorter than block_size after tokenization and grouping
        raise ValueError("Tokenized training dataset is empty. This might happen if your texts are too short, "
                        "all texts were filtered out, or block_size is too large.")

    # ── logging steps ───────────────────────────────────────────────────────
    batch_size = 2
    gra_accu_steps = 20
    # Make sure lm_ds["train"] is not empty before calculating logging_steps
    # if len(lm_ds["train"]) > 0:
    #     logging_steps = max(
    #         1, int(len(lm_ds["train"]) / (20 * batch_size * gra_accu_steps))
    #     )
    # else:
    logging_steps = 1 # Default if no training data (though it should have failed earlier)
    # ── model ────────────────────────────────────────────────────────────────
    default_dtype = torch.get_default_dtype()
    torch.set_default_dtype(torch.bfloat16)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name_or_path,
        device_map="auto",
        use_cache=False,
        torch_dtype=torch.bfloat16,
    )
    torch.set_default_dtype(default_dtype)
    model.config.pad_token_id = pad_id
    model.config.pretraining_tp = 1

    # ── LoRA / PEFT switch ───────────────────────────────────────────────────
    use_peft = args.train_mode.lower() == "peft"
    peft_config = None
    if use_peft:
        peft_config = LoraConfig(
            lora_alpha=16,
            lora_dropout=0.1,
            r=64,
            bias="none",
            task_type="CAUSAL_LM",
        )
        model = get_peft_model(model, peft_config)

    # ── data collator ────────────────────────────────────────────────────────
    collator = DataCollatorForLanguageModeling(tokenizer, mlm=False)

    # ── training arguments ───────────────────────────────────────────────────
    train_args = TrainingArguments(
        output_dir=args.output_path,
        num_train_epochs=args.local_epoch,
        per_device_train_batch_size=batch_size,
        gradient_accumulation_steps=gra_accu_steps,
        gradient_checkpointing=False,
        optim="paged_adamw_32bit",
        logging_steps=logging_steps,
        save_strategy="epoch",
        learning_rate=5e-4,
        bf16=True,
        max_grad_norm=0.3,
        warmup_ratio=0.03,
        lr_scheduler_type="cosine",
        disable_tqdm=True,
        save_total_limit=2,
        save_safetensors=True,
        report_to=[],              # we handle W&B manually
        # device="cuda"
    )

    flare.init()

    site_name = flare.get_site_name()              # e.g. "site-oasst1"
    client_id = site_name.split("-", 1)[-1]        # yields "oasst1"

    # Path to store wandb run ID
    wandb_id_file = os.path.join(args.output_path, f"wandb_run_id_{client_id}.txt")
    resume_id = None

    # Check if we have a previous run ID
    if os.path.exists(wandb_id_file):
        try:
            with open(wandb_id_file, 'r') as f:
                resume_id = f.read().strip()
            print(f"Resuming wandb run with ID: {resume_id}")
        except Exception as e:
            print(f"Error reading wandb ID file: {e}")

    # ── W&B run (one per FL client, persists across rounds) ──────────────────
    wandb_run = wandb.init(
        entity="allenai-team1",
        project="nvflare",
        name=f"{args.train_mode}-{client_id}-{args.model_name_or_path.rsplit('/',1)[-1]}",
        id=resume_id if resume_id else f"{client_id}-{args.model_name_or_path.rsplit('/',1)[-1]}-{int(time.time())}",
        group=client_id,
        resume="allow",
        config={"client_id": client_id, **train_args.to_dict()},
    )

    # Save the run ID for future rounds
    if not resume_id:
        os.makedirs(os.path.dirname(wandb_id_file), exist_ok=True)
        with open(wandb_id_file, 'w') as f:
            f.write(wandb_run.id)

    train_args.report_to = ["wandb"]
    train_args.run_name = wandb_run.name

    # ── NVFlare federated loop ───────────────────────────────────────────────
    # flare.init()

    # site_name = flare.get_site_name()              # e.g. "site-oasst1"
    # client_id = site_name.split("-", 1)[-1]        # yields "oasst1"

    # # ── W&B run (one per FL client, persists across rounds) ──────────────────
    # wandb_run = wandb.init(
    #     entity="allenai-team1",
    #     project="nvflare",
    #     name = f"{args.train_mode}-{client_id}-{args.model_name_or_path.rsplit('/',1)[-1]}",
    #     id   = f"{client_id}-{args.model_name_or_path.rsplit('/',1)[-1]}-{int(time.time())}",   # unique per site
    #     group = client_id,                      # lets you overlay curves
    #     resume = "allow",
    #     config = {"client_id": client_id, **train_args.to_dict()},
    # )
    # train_args.report_to = ["wandb"]
    # train_args.run_name = wandb_run.name

        # ── Trainer ──────────────────────────────────────────────────────────────
    # trainer = Trainer(
    #     model=model,
    #     args=train_args,
    #     train_dataset=lm_ds["train"],
    #     eval_dataset=lm_ds["validation"],
    #     data_collator=collator,
    # )
    
    trainer = CustomTrainer(
        model=model,
        args=train_args,
        train_dataset=lm_ds["train"],
        eval_dataset=lm_ds["validation"],
        data_collator=collator,
    )
    
    # Add a file to log throughput per client
    throughput_log_file = os.path.join(args.output_path, f"throughput_{client_id}.txt")

    while flare.is_running():
        input_model = flare.receive()
        curr_round = input_model.current_round
        print(f"--- federated round {curr_round} ---")

        # rename keys to match local model
        global_state = {
            k.replace("model.", "", 1): v for k, v in input_model.params.items()
        }

        # ── evaluation helper ───────────────────────────────────────────────────────
        def evaluate(weights):
            if use_peft:
                set_peft_model_state_dict(trainer.model, weights)
            else:
                trainer.model.load_state_dict(weights)

            if trainer.eval_dataset is None or len(trainer.eval_dataset) == 0:
                # no validation data – return dummy metrics
                return {"eval_loss": float("nan"), "perplexity": float("nan")}

            metrics = trainer.evaluate()
            loss_val = metrics["eval_loss"]
            metrics["perplexity"] = float(np.exp(min(20, loss_val)))
            print("eval metrics:", metrics)
            return metrics


        eval_metrics = evaluate(global_state)
        eval_loss = float(eval_metrics["eval_loss"])
        eval_ppl = float(eval_metrics["perplexity"])

        # ── training ────────────────────────────────────────────────────────
        train_start = time.time()
        if curr_round == 0:
            trainer.train()
        else:
            ckpt_dir = trainer_utils.get_last_checkpoint(trainer.args.output_dir)
            if use_peft:
                torch.save(global_state,
                           os.path.join(ckpt_dir, utils.WEIGHTS_NAME))
            else:
                # trainer.model.save_pretrained(ckpt_dir, safe_serialization=False)
                trainer.model.save_pretrained(ckpt_dir, safe_serialization=True)

            # extend epochs so the next `trainer.train()` continues
            if args.clean_up:
                trainer.args.num_train_epochs = (curr_round + 1) * args.local_epoch
            else:
                trainer.args.num_train_epochs += args.local_epoch
            trainer.train(resume_from_checkpoint=True)
        train_end = time.time()

        # ── throughput calculation ──────────────────────────────────────────
        train_samples = len(lm_ds["train"])
        train_tokens = train_samples * args.block_size
        elapsed = train_end - train_start
        samples_per_sec = train_samples / elapsed if elapsed > 0 else float("nan")
        tokens_per_sec = train_tokens / elapsed if elapsed > 0 else float("nan")
        print(f"[THROUGHPUT] Round {curr_round}: {samples_per_sec:.2f} samples/sec, {tokens_per_sec:.2f} tokens/sec, elapsed {elapsed:.2f}s")

        # Log to wandb
        wandb_run.log({
            "throughput_samples_per_sec": samples_per_sec,
            "throughput_tokens_per_sec": tokens_per_sec,
            "train_elapsed_sec": elapsed,
            "round": curr_round,
        }, step=curr_round)

        # Save to txt file (append)
        with open(throughput_log_file, "a") as f:
            f.write(f"round {curr_round}, samples/sec: {samples_per_sec:.2f}, tokens/sec: {tokens_per_sec:.2f}, elapsed: {elapsed:.2f}s\n")

        # ── collect weights to send back ────────────────────────────────────
        if use_peft:
            out_state = get_peft_model_state_dict(trainer.model)
        else:
            out_state = trainer.model.state_dict()
            out_state = {f"model.{k}": v.cpu() for k, v in out_state.items()}

        if args.message_mode.lower() == "numpy":
            out_state = {k: v.cpu().to(torch.float32) for k, v in out_state.items()} # Ensure CPU for numpy conversion
        output_model = flare.FLModel(
            params=out_state,
            metrics={"eval_loss": eval_loss, "perplexity": eval_ppl},
            meta={"NUM_STEPS_CURRENT_ROUND": len(lm_ds["train"]), "CURRENT_ROUND": curr_round},

        )
        flare.send(output_model)

    # ── tidy up ──────────────────────────────────────────────────────────────
    wandb.finish()


if __name__ == "__main__":
    main()