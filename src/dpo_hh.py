import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

from common import DEVICE, apply_prompt_format
from src.dpo import train_dpo


MODEL_ID = "Qwen/Qwen2.5-0.5B"
TRAIN_EXAMPLES = 15_000


if __name__ == "__main__":
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    model = AutoModelForCausalLM.from_pretrained("./best_sft_checkpoint", dtype=torch.bfloat16).to(DEVICE)
    ref = AutoModelForCausalLM.from_pretrained("./best_sft_checkpoint", dtype=torch.bfloat16).to(DEVICE)
    for p in ref.parameters():
        p.requires_grad = False
    ref.eval()
    
    train_dataset = load_dataset("Anthropic/hh-rlhf", split="train")
    train_dataset = train_dataset.shuffle(seed=42).select(range(TRAIN_EXAMPLES))
    val_dataset = load_dataset("json", data_files="./eval_sets/hh_rlhf_eval_300_seed42.json", split="train")
    
    print("Preprocessing datasets...")
    def _preproc(example):
        prompt, chosen = example["chosen"].split(" Assistant: ")[0] 
        prompt, rejected = example["rejected"].split(" Assistant: ")[0] 
        
        prompt, chosen, rejected = prompt.strip(), chosen.strip(), rejected.strip()
        return apply_prompt_format(prompt), chosen, rejected
    
    train_dataset = train_dataset.map(_preproc)
    
    train_dpo(model, ref, tokenizer, train_dataset, val_dataset, "dpo-hh")
