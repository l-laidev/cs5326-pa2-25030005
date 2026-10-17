from argparse import ArgumentParser

import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

from src.common import DEVICE, apply_prompt_format
from src.dpo import train_dpo


MODEL_ID = "Qwen/Qwen2.5-0.5B"
TRAIN_EXAMPLES = 15_000
VAL_EXAMPLES = 1_000


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--run-name", type=str, required=True)
    args = parser.parse_args()
    
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    model = AutoModelForCausalLM.from_pretrained("./checkpoints/best/sft/final-run", dtype=torch.float32).to(DEVICE)
    ref = AutoModelForCausalLM.from_pretrained("./checkpoints/best/sft/final-run", dtype=torch.bfloat16).to(DEVICE)
    for p in ref.parameters():
        p.requires_grad = False
    ref.eval()
    
    dataset = load_dataset("Anthropic/hh-rlhf", split="train")
    dataset = dataset.shuffle(seed=42)
    train_dataset = dataset.select(range(TRAIN_EXAMPLES))
    val_dataset = dataset.select(range(TRAIN_EXAMPLES, TRAIN_EXAMPLES + VAL_EXAMPLES))
    
    print("Preprocessing datasets...")
    def _preproc(example):
        def _extract_first_turn(text):
            first_turn = text.split("Human:")[1]  # element 0 is empty text
            return first_turn.split("Assistant:")[:2]  # response can have multiple "Assistant:", so keep the 1st one

        prompt, chosen = _extract_first_turn(example["chosen"])
        _, rejected = _extract_first_turn(example["rejected"])

        prompt, chosen, rejected = prompt.strip(), chosen.strip(), rejected.strip()
        return {
            "prompt": apply_prompt_format(prompt),
            "chosen": chosen,
            "rejected": rejected,
        }
    
    train_dataset = train_dataset.map(_preproc, remove_columns=train_dataset.column_names)
    val_dataset = val_dataset.map(_preproc, remove_columns=val_dataset.column_names)
    
    train_dpo(model, ref, tokenizer, train_dataset, val_dataset, f"hh_{args.run_name}")
