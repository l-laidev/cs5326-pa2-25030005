from argparse import ArgumentParser
import itertools
import json
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, get_cosine_schedule_with_warmup
from datasets import load_dataset
from torch.optim import AdamW

from src.common import DEVICE, apply_prompt_format, compute_logp, process_one_batch, save_ckpt, load_ckpt


MODEL_ID = "Qwen/Qwen2.5-0.5B"
MAX_LENGTH = 512
TRAIN_EXAMPLES = 30_000
VAL_EXAMPLES = 1_000
PER_DEVICE_BATCH_SIZE = 2
GRADIENT_ACCUMULATION_STEPS = 16
LEARNING_RATE = 5e-6
WARMUP_RATIO = 0.05
WEIGHT_DECAY = 0.1
MAX_GRAD_NORM = 0.01
EPOCHS = 1
VAL_STEPS = 250
CKPT_INTERVAL = 2_000
MAX_STEPS = float("inf")


def build_sft_batch(tokenizer, prompts, responses, max_length):
    batch_input_ids = []
    batch_attention_mask = []
    batch_labels = []

    for prompt, response in zip(prompts, responses):
        input_ids, attention_mask, labels = process_one_batch(tokenizer, prompt, response, max_length)

        batch_input_ids.append(input_ids)
        batch_attention_mask.append(attention_mask)
        batch_labels.append(labels)

    return {
        "input_ids": torch.stack(batch_input_ids),
        "attention_mask": torch.stack(batch_attention_mask),
        "labels": torch.stack(batch_labels),
    }

def compute_response_logprobs(model, tokenizer, prompts, responses, max_length, reduction="sum"):
    batch = build_sft_batch(tokenizer, prompts, responses, max_length)
    device = next(model.parameters()).device
    batch = {k: v.to(device) for k, v in batch.items()}

    outputs = model(
        input_ids=batch["input_ids"],
        attention_mask=batch["attention_mask"],
        labels=batch["labels"]
    )
    
    return compute_logp(outputs.logits, batch["labels"], reduction=reduction)

def compute_sft_loss(model, tokenizer, prompts, responses, max_length):
    logp = compute_response_logprobs(model, tokenizer, prompts, responses, max_length, reduction="sum")
    return -1.0 * logp.mean()

def evaluate(model, val_loader):
    was_train = model.training
    model.eval()
    total_loss = 0.0
    with torch.no_grad(), torch.inference_mode():
        for batch in tqdm.tqdm(val_loader, total=len(val_loader), desc="Evaluating", leave=False):
            prompts, responses = batch["prompt"], batch["response"]

            with torch.amp.autocast("cuda", dtype=torch.float16):
                loss = compute_sft_loss(model, tokenizer, prompts, responses, MAX_LENGTH)
            total_loss += loss.item()
    if was_train:
        model.train()
    return total_loss / len(val_loader)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--run-name", type=str, required=True)
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(MODEL_ID, dtype=torch.float32).to(DEVICE)

    dataset = load_dataset("HuggingFaceH4/ultrachat_200k", split="train_sft")
    dataset = dataset.shuffle(seed=42)
    train_subset = dataset.select(range(TRAIN_EXAMPLES))
    val_subset = dataset.select(range(TRAIN_EXAMPLES, TRAIN_EXAMPLES + VAL_EXAMPLES))

    print("Preprocessing datasets...")
    def _preproc(example):
        return {
            "prompt": apply_prompt_format(example["messages"][0]["content"]),
            "response": example["messages"][1]["content"],
        }
    train_dataset = train_subset.map(_preproc, remove_columns=train_subset.column_names)
    val_dataset = val_subset.map(_preproc, remove_columns=val_subset.column_names)

    train_generator = torch.Generator().manual_seed(42)
    train_loader = DataLoader(train_dataset, batch_size=PER_DEVICE_BATCH_SIZE, shuffle=True, generator=train_generator)
    val_loader = DataLoader(val_dataset, batch_size=PER_DEVICE_BATCH_SIZE)

    optimizer = AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    total_steps = (len(train_loader) // GRADIENT_ACCUMULATION_STEPS) * EPOCHS
    warmup_steps = int(WARMUP_RATIO * total_steps)

    scheduler = get_cosine_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps)
    scaler = torch.amp.GradScaler("cuda")
    
    print("Starting training...")
    best_val_loss = float("inf")
    losses = {"train": [], "val": []}
    global_step = 0
    resume_step = 0
    ckpt_path = Path("checkpoints") / "sft" / args.run_name / "state.pt"
    loader_iter = train_loader
    
    if ckpt_path.exists():
        global_step, resume_step, best_val_loss, losses = load_ckpt(
            model,
            optimizer,
            train_generator,
            ckpt_path,
            scheduler,
        )
        
        if resume_step > 0:
            loader_iter = itertools.islice(train_loader, resume_step, None)

    for epoch in (t1 := tqdm.trange(EPOCHS, desc="Epochs")):
        model.train()
        optimizer.zero_grad()

        step = resume_step
        for batch in (t2 := tqdm.tqdm(loader_iter, initial=resume_step, total=len(train_loader), desc="Step", leave=False)):
            step += 1
            
            prompts, responses = batch["prompt"], batch["response"]
            with torch.amp.autocast("cuda", dtype=torch.float16):
                loss = compute_sft_loss(model, tokenizer, prompts, responses, MAX_LENGTH)
                losses["train"].append(loss.item())  # plotting code will deal with accumulation etc
                loss /= GRADIENT_ACCUMULATION_STEPS

            scaler.scale(loss).backward()

            if step % GRADIENT_ACCUMULATION_STEPS == 0 or step == len(train_loader) or step >= MAX_STEPS:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), MAX_GRAD_NORM)

                old_scale = scaler.get_scale()
                scaler.step(optimizer)
                scaler.update()
                new_scale = scaler.get_scale()

                # keep scheduler in sync only if optimizer.step() wasn't skipped (scale didn't decrease)
                # avoids pytorch's warning
                if new_scale >= old_scale:
                    scheduler.step()
                
                optimizer.zero_grad()
                global_step += 1

                if global_step % 100 == 0:
                    t2.set_description_str(f"Epoch {epoch+1} | Step {global_step}/{total_steps} | Loss: {loss.item() * GRADIENT_ACCUMULATION_STEPS:.4f}")
                
            if step % CKPT_INTERVAL == 0:
                save_ckpt(
                    model,
                    optimizer,
                    train_generator,
                    global_step,
                    step,
                    best_val_loss,
                    ckpt_path,
                    scheduler,
                    losses,
                )

            if step == 1 or step % VAL_STEPS == 0 or step == len(train_loader) or step >= MAX_STEPS:
                val_loss = evaluate(model, val_loader)
                losses["val"].append(val_loss)

                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    model.save_pretrained(f"./checkpoints/best/sft/{args.run_name}")
                    tokenizer.save_pretrained(f"./checkpoints/best/sft/{args.run_name}")
                
                t1.set_description_str(f"[Step {step:04}] Validation Loss: {val_loss:.4f}, Best Loss: {best_val_loss:.4f}")

            if step >= MAX_STEPS:  # for short experiments
                break

    model.save_pretrained(f"./checkpoints/final/sft/{args.run_name}")
    tokenizer.save_pretrained(f"./checkpoints/final/sft/{args.run_name}")
    with open(f"./checkpoints/final/sft/{args.run_name}/metrics.json", "w") as f:
        f.write(json.dumps(losses))
    print("Training complete. Final checkpoint saved.")
