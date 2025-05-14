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
    parser.add_argument("--block_size", type=int, default=1024)
    parser.add_argument("--gpu", type=str, default="0")
    return parser.parse_args()


# ──────────────────────────────────────────────────────────────────────────────
# dataset helpers
def tokenize_function(examples, tokenizer):
    return tokenizer(examples["text"])


def group_texts(examples, block_size):
    """Pack tokens into fixed-length blocks for causal-LM pre-training."""
    concatenated = {k: list(chain.from_iterable(examples[k])) for k in examples.keys()}
    total_len = (len(next(iter(concatenated.values()))) // block_size) * block_size
    result = {
        k: [t[i : i + block_size] for i in range(0, total_len, block_size)]
        for k, t in concatenated.items()
    }
    return result

def load_npy_data_as_dataset(file_path: str) -> datasets.Dataset:
    """Loads a .npy file containing text data and returns a datasets.Dataset."""
    if not os.path.exists(file_path):
        print(f"Warning: Data file not found: {file_path}. Returning empty dataset.")
        # Return an empty dataset with the expected "text" column
        return datasets.Dataset.from_dict({"text": []})

    print(f"Loading data from {file_path}...")
    try:
        # allow_pickle=True is often necessary for arrays of strings
        data_array = np.load(file_path, allow_pickle=True)
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
    
    # Ensure all items are strings
    if not all(isinstance(item, str) for item in text_list):
        print(f"Warning: Not all items in {file_path} are strings. Attempting to convert.")
        try:
            text_list = [str(item) for item in text_list]
        except Exception as e:
            raise ValueError(f"Could not convert all items in {file_path} to strings: {e}")


    # The Dataset needs a dictionary where keys are column names
    return datasets.Dataset.from_dict({"text": text_list})

# ──────────────────────────────────────────────────────────────────────────────
def main():
    args = parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    # ── tokenizer ────────────────────────────────────────────────────────────
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    pad_id = tokenizer.pad_token_id

    # ── raw datasets ─────────────────────────────────────────────────────────
    print(f"Attempting to load training data from: {args.data_path_train}")
    raw_train_ds = load_npy_data_as_dataset(args.data_path_train)

    # Handle validation data:
    # If a specific validation file is provided and it's different from the training file
    if args.data_path_valid and os.path.exists(args.data_path_valid) and args.data_path_valid != args.data_path_train:
        print(f"Attempting to load validation data from: {args.data_path_valid}")
        raw_valid_ds = load_npy_data_as_dataset(args.data_path_valid)
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

    tokenised = raw_ds.map(
        tokenize_function,
        fn_kwargs={"tokenizer": tokenizer},
        batched=True,
        remove_columns=["text"],
        desc="Tokenising",
    )

    lm_ds = tokenised.map(
        group_texts,
        fn_kwargs={"block_size": args.block_size},
        batched=True,
        desc=f"Packing into {args.block_size}-token blocks",
    )

    # Ensure tokenized datasets are not empty
    if len(lm_ds["train"]) == 0:
        # This can happen if all texts are shorter than block_size after tokenization and grouping
        raise ValueError("Tokenized training dataset is empty. This might happen if your texts are too short, "
                        "all texts were filtered out, or block_size is too large.")

    # ── logging steps ───────────────────────────────────────────────────────
    batch_size = 2
    gra_accu_steps = 20
    # Make sure lm_ds["train"] is not empty before calculating logging_steps
    if len(lm_ds["train"]) > 0:
        logging_steps = max(
            1, int(len(lm_ds["train"]) / (20 * batch_size * gra_accu_steps))
        )
    else:
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
        save_safetensors=False,
        report_to=[],              # we will handle W&B manually
    )




    # ── NVFlare federated loop ───────────────────────────────────────────────
    flare.init()

    site_name = flare.get_site_name()              # e.g. "site-oasst1"
    client_id = site_name.split("-", 1)[-1]        # yields "oasst1"

    # ── W&B run (one per FL client, persists across rounds) ──────────────────
    wandb_run = wandb.init(
        entity="allenai-team1",
        project="nvflare",
        name = f"{args.train_mode}-{client_id}-{args.model_name_or_path.rsplit('/',1)[-1]}",
        id   = f"{client_id}-{args.model_name_or_path.rsplit('/',1)[-1]}-{int(time.time())}",   # unique per site
        group = client_id,                      # lets you overlay curves
        resume = "allow",
        config = {"client_id": client_id, **train_args.to_dict()},
    )
    train_args.report_to = ["wandb"]
    train_args.run_name = wandb_run.name

        # ── Trainer ──────────────────────────────────────────────────────────────
    trainer = Trainer(
        model=model,
        args=train_args,
        train_dataset=lm_ds["train"],
        eval_dataset=lm_ds["validation"],
        data_collator=collator,
    )
    
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
        if curr_round == 0:
            trainer.train()
        else:
            ckpt_dir = trainer_utils.get_last_checkpoint(trainer.args.output_dir)
            if use_peft:
                torch.save(global_state,
                           os.path.join(ckpt_dir, utils.WEIGHTS_NAME))
            else:
                trainer.model.save_pretrained(ckpt_dir, safe_serialization=False)

            # extend epochs so the next `trainer.train()` continues
            if args.clean_up:
                trainer.args.num_train_epochs = (curr_round + 1) * args.local_epoch
            else:
                trainer.args.num_train_epochs += args.local_epoch
            trainer.train(resume_from_checkpoint=True)

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
            meta={"NUM_STEPS_CURRENT_ROUND": len(lm_ds["train"])},
        )
        flare.send(output_model)

    # ── tidy up ──────────────────────────────────────────────────────────────
    wandb.finish()


if __name__ == "__main__":
    main()


#test git