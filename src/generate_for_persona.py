from argparse import ArgumentParser
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from datasets import load_dataset
import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from src.common import DEVICE, apply_prompt_format, generate


MODEL_ID = "HuggingFaceTB/SmolLM2-360M-Instruct"

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--saved-model", type=str, required=True)
    parser.add_argument("--name", type=str, required=True)
    
    args = parser.parse_args()
    
    model = AutoModelForCausalLM.from_pretrained(
        args.saved_model,
        dtype=torch.bfloat16,
    ).to(DEVICE)
    
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    
    def _preproc(example):
        return {
            "instruction": apply_prompt_format(example["prompt"]),
            "chosen": example["chosen"],
            "rejected": example["rejected"],
        }
    
    dataset = load_dataset("json", data_files="data/persona_lora/ramsay_style_preferences.json", split="train")
    dataset = dataset.map(_preproc)
    
    dataset = DataLoader(dataset, batch_size=1, shuffle=False)
    
    print("Starting generation...")
    results = []
    with torch.no_grad(), torch.inference_mode():
        for idx, batch in tqdm.tqdm(enumerate(dataset), total=len(dataset), desc="Generating"):
            instruction = batch["instruction"]

            inputs, outputs = generate(model, tokenizer, instruction)
        
            prompt_len = len(inputs.input_ids[0])
            response = outputs[0][prompt_len:]
            response = tokenizer.decode(response, skip_special_tokens=True)
            
            results.append({
                "id": idx+1,
                "instruction": instruction,
                f"{args.name}": response,
                "model": args.name,
                "chosen": batch["chosen"],
                "rejected": batch["rejected"],
            })

    base = Path("generations")
    base.mkdir(parents=True, exist_ok=True)
    output_path = base / f"ramsay_eval_{args.name}.json"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print(f"Successfully saved {len(results)} {args.name} entries to {output_path}")
