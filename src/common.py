from typing import Literal

import torch.nn.functional as F


def process_one_batch(tokenizer, prompt, response, max_length):
    prompt_messages = [{"role": "user", "content": prompt}]
    prompt_text = tokenizer.apply_chat_template(prompt_messages, tokenize=False, add_generation_prompt=True)
    
    eos = tokenizer.eos_token if tokenizer.eos_token else ""
    full_text = prompt_text + response + eos

    encoded = tokenizer(
        full_text,
        max_length=max_length,
        truncation=True,
        padding="max_length",
        return_tensors="pt",
    )
    input_ids = encoded["input_ids"][0]
    attention_mask = encoded["attention_mask"][0]

    prompt_encoded = tokenizer(prompt_text, max_length=max_length, truncation=True, return_tensors="pt")
    prompt_len = len(prompt_encoded["input_ids"][0])

    # response-only masking with -100
    labels = input_ids.clone()
    labels[:prompt_len] = -100
    labels[attention_mask == 0] = -100

    return input_ids, attention_mask, labels

def compute_logp(logits, labels, reduction: Literal["none"] | Literal["sum"] = "sum"):
    # logits at i predict labels at i+1
    shift_logits = logits[..., :-1, :].contiguous()
    shift_labels = labels[..., 1:].contiguous()
    
    flat_logits = shift_logits.view(-1, shift_logits.size(-1))
    flat_labels = shift_labels.view(-1)
    
    log_probs = F.log_softmax(flat_logits, dim=-1)
    token_log_probs = log_probs.gather(1, flat_labels.unsqueeze(1)).squeeze(1)
    
    mask = (flat_labels != -100).float()
    token_log_probs = token_log_probs * mask
    
    if reduction == "none":
        token_log_probs = token_log_probs[mask]
    elif reduction == "sum":
        token_log_probs = token_log_probs.view(labels.size(0), -1).sum(dim=-1)
    else:
        raise ValueError(f"Unknown reduction: {reduction}")
    
    return token_log_probs
