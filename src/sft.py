from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, get_cosine_schedule_with_warmup
from datasets import load_dataset
from torch.optim import AdamW

from src.common import compute_logp, process_one_batch, save_ckpt, load_ckpt


MODEL_ID = "Qwen/Qwen2.5-0.5B"
MAX_LENGTH = 512
TRAIN_EXAMPLES = 30_000
VAL_EXAMPLES = 1_000
PER_DEVICE_BATCH_SIZE = 2
GRADIENT_ACCUMULATION_STEPS = 16
LEARNING_RATE = 2e-5
WARMUP_RATIO = 0.03
WEIGHT_DECAY = 0.1
MAX_GRAD_NORM = 1.0
EPOCHS = 1
CKPT_INTERVAL = 100

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


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

def compute_response_logprobs(model, tokenizer, prompts, responses, max_length):
    batch = build_sft_batch(tokenizer, prompts, responses, max_length)
    device = next(model.parameters()).device
    batch = {k: v.to(device) for k, v in batch.items()}

    outputs = model(
        input_ids=batch["input_ids"],
        attention_mask=batch["attention_mask"],
        labels=batch["labels"]
    )
    
    return compute_logp(outputs.logits, batch["labels"], reduction="none")

def compute_sft_loss(model, tokenizer, prompts, responses, max_length):
    logp = compute_response_logprobs(model, tokenizer, prompts, responses, max_length)
    return -1.0 * logp.mean()

def evaluate(model, val_loader):
    model.eval()
    total_loss = 0.0
    with torch.no_grad(), torch.inference_mode():
        for batch in tqdm.tqdm(val_loader, total=len(val_loader), desc="Evaluating"):
            prompts, responses = batch

            with torch.cuda.amp.autocast(dtype=torch.float16):
                loss = compute_sft_loss(model, tokenizer, prompts, responses, MAX_LENGTH)
            total_loss += loss.item()
    return total_loss / len(val_loader)


if __name__ == "__main__":
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(MODEL_ID, torch_dtype=torch.float32)
    model.to(device)

    dataset = load_dataset("HuggingFaceH4/ultrachat_200k", split="train_sft")
    train_subset = dataset.shuffle(seed=42).select(range(TRAIN_EXAMPLES))
    val_subset = dataset.shuffle(seed=42).select(range(TRAIN_EXAMPLES, TRAIN_EXAMPLES + VAL_EXAMPLES))

    print("Preprocessing datasets...")
    train_dataset = train_subset.map(lambda example: (example["messages"][0]["content"], example["messages"][1]["content"]),
                                     remove_columns=train_subset.column_names)
    val_dataset = val_subset.map(lambda example: (example["messages"][0]["content"], example["messages"][1]["content"]),
                                 remove_columns=val_subset.column_names)

    train_generator = torch.Generator().manual_seed(42)
    train_loader = DataLoader(train_dataset, batch_size=PER_DEVICE_BATCH_SIZE, shuffle=True, generator=train_generator)
    val_loader = DataLoader(val_dataset, batch_size=PER_DEVICE_BATCH_SIZE)

    optimizer = AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    total_steps = (len(train_loader) // GRADIENT_ACCUMULATION_STEPS) * EPOCHS
    warmup_steps = int(WARMUP_RATIO * total_steps)

    scheduler = get_cosine_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps)
    scaler = torch.cuda.amp.GradScaler()
    
    print("Starting training...")
    best_val_loss = float("inf")
    global_step = 0
    resume_step = 0
    ckpt_path = Path("checkpoints") / "sft" / "state.pt"
    
    if ckpt_path.exists():
        global_step, resume_step, best_val_loss = load_ckpt(
            model,
            optimizer,
            train_generator,
            ckpt_path,
            scheduler,
        )

    for epoch in (t1 := tqdm.trange(EPOCHS, desc="Epochs")):
        model.train()
        optimizer.zero_grad()

        step = resume_step
        for batch in (t2 := tqdm.tqdm(train_loader, initial=resume_step, total=len(train_loader), desc="Step", leave=False)):
            step += 1
            
            prompts, responses = batch
            with torch.cuda.amp.autocast(dtype=torch.float16):
                loss = compute_sft_loss(model, tokenizer, prompts, responses, MAX_LENGTH)
                loss /= GRADIENT_ACCUMULATION_STEPS

            scaler.scale(loss).backward()

            if step % GRADIENT_ACCUMULATION_STEPS == 0 or step == len(train_loader):
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), MAX_GRAD_NORM)

                scaler.step(optimizer)
                scaler.update()
                scheduler.step()
                
                optimizer.zero_grad()
                global_step += 1

                if global_step % 100 == 0:
                    t2.set_description_str(f"Epoch {epoch+1} | Step {global_step}/{total_steps} | Loss: {loss.item() * GRADIENT_ACCUMULATION_STEPS:.4f}")
                
                if global_step % CKPT_INTERVAL == 0:
                    save_ckpt(
                        model,
                        optimizer,
                        train_generator,
                        global_step,
                        step,
                        best_val_loss,
                        ckpt_path,
                        scheduler,
                    )

        val_loss = evaluate(model, val_loader)
        t1.set_description_str(f"Epoch {epoch+1} Complete. Validation Loss: {val_loss:.4f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            model.save_pretrained("./best_sft_checkpoint")
            tokenizer.save_pretrained("./best_sft_checkpoint")
            t1.set_description_str("Saved best checkpoint.")

    model.save_pretrained("./final_sft_checkpoint")
    tokenizer.save_pretrained("./final_sft_checkpoint")
    print("Training complete. Final checkpoint saved.")
