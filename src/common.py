from pathlib import Path
from typing import Literal

import torch
import torch.nn.functional as F


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def apply_prompt_format(messages, add_generation_prompt=True):
    SYSTEM = """<|im_start|>system
You are a helpful assistant.<|im_end|>"""

    formatted = ""
    for msg in messages:
        role = "user"
        content = msg
        if isinstance(msg, dict):
            role = msg['role']
            content = msg['content']
        
        formatted += f"""<|im_start|>{role}
{content}<|im_end|>
"""
    generation_prompt = "\n<|im_start|>assistant\n" if add_generation_prompt else ""
    return SYSTEM + '\n' + formatted + generation_prompt

@torch.no_grad
def generate(model, tokenizer, prompt):
    inputs = tokenizer(
        prompt,
        max_length=256,
        truncation=True,
        padding="max_length",
        return_tensors="pt",
    ).to(model.device)
    outputs = model.generate(
        **inputs,
        max_new_tokens=256,
        do_sample=False,
        repetition_penalty=1.05,
        pad_token_id=tokenizer.eos_token_id,
    )
    return inputs, outputs

def process_one_batch(tokenizer, formatted_prompt_text, response, max_length):
    eos = tokenizer.eos_token if tokenizer.eos_token else ""
    full_text = formatted_prompt_text + response + eos

    encoded = tokenizer(
        full_text,
        max_length=max_length,
        truncation=True,
        padding="max_length",
        return_tensors="pt",
    )
    input_ids = encoded.input_ids[0]
    attention_mask = encoded.attention_mask[0]

    prompt_encoded = tokenizer(formatted_prompt_text, max_length=max_length, truncation=True, return_tensors="pt")
    prompt_len = len(prompt_encoded.input_ids[0])

    # response-only masking with -100
    labels = input_ids.clone()
    labels[:prompt_len] = -100
    labels[attention_mask == 0] = -100

    return input_ids, attention_mask, labels

def compute_logp(logits, labels, reduction: Literal["none"] | Literal["sum"] = "sum"):
    # logits at i predict labels at i+1
    shift_logits = logits[..., :-1, :].contiguous()
    shift_labels = labels[..., 1:].contiguous()
    
    if logits.requires_grad:  # reference model does not require grad
        shift_logits.retain_grad()
    
    mask = (shift_labels == -100)
    shift_labels[mask] = 0  # pick the first one, will be discarded
    
    log_probs = F.log_softmax(shift_logits, dim=-1)
    token_log_probs = log_probs.gather(-1, shift_labels.unsqueeze(-1)).squeeze(-1)
    
    token_log_probs = token_log_probs * (~mask)
    
    if reduction == "none":
        pass
    elif reduction == "sum":
        token_log_probs = token_log_probs.sum(dim=-1)
    else:
        raise ValueError(f"Unknown reduction: {reduction}")

    return token_log_probs

def save_ckpt(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    train_generator: torch.Generator,
    global_step: int,
    resume_step: int,
    best_metric: float,
    path: Path,
    scheduler: torch.optim.lr_scheduler.LambdaLR | None = None,
):
    path.parent.mkdir(parents=True, exist_ok=True)
    
    torch.save(
        {
            "generator": train_generator.get_state(),
            "model": model.state_dict(),
            "optim": optimizer.state_dict(),
            "global_step": global_step,
            "resume_step": resume_step,
            "best_metric": best_metric,
            "scheduler": scheduler.state_dict() if scheduler is not None else None,
        },
        path
    )

def load_ckpt(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    train_generator: torch.Generator,
    path: Path,
    scheduler: torch.optim.lr_scheduler.LambdaLR | None = None,
) -> int:
    assert path.exists()
    
    states = torch.load(path)
    
    train_generator.set_state(states["generator"])
    model.load_state_dict(states["model"])
    optimizer.load_state_dict(states["optim"])
    if scheduler is not None:
        scheduler.load_state_dict(states["scheduler"])
    
    return states["global_step"], states["resume_step"], states["best_metric"]
