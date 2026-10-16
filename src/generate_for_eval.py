from argparse import ArgumentParser
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from datasets import load_dataset
import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from src.common import DEVICE, apply_prompt_format, generate


MODEL_ID = "Qwen/Qwen2.5-0.5B"

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--saved-model", type=str, required=True)
    parser.add_argument("--name", type=str, required=True)
    parser.add_argument("--dataset", type=str, required=True, choices=["alpaca", "hh"])
    
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
            "id": example["id"],
            "instruction": apply_prompt_format(example["instruction"]),
            "baseline": example["output"],
        }
    
    if args.dataset == "alpaca":
        dataset = load_dataset("json", data_files="generations/alpaca_eval_baseline.json", split="train")
    elif args.dataset == "hh":
        dataset = load_dataset("json", data_files="generations/hh_eval_baseline.json", split="train")
    else:
        raise ValueError(f"Unknown dataset: {args.dataset}")
    dataset = dataset.map(_preproc)
    
    dataset = DataLoader(dataset, batch_size=1, shuffle=False)
    
    print("Starting generation...")
    results = []
    with torch.no_grad(), torch.inference_mode():
        for idx, batch in tqdm.tqdm(enumerate(dataset), total=len(dataset), desc="Generating"):
            id, instruction, baseline_response = batch["id"], batch["instruction"], batch["baseline"]

            inputs, outputs = generate(model, tokenizer, instruction)
        
            prompt_len = len(inputs.input_ids[0])
            response = outputs[0][prompt_len:]
            response = tokenizer.decode(response, skip_special_tokens=True)
            
            results.append({
                "id": id,
                "instruction": instruction,
                f"{args.name}": response,
                "model": args.name,
                "baseline": baseline_response,
            })

    base = Path("generations")
    base.mkdir(parents=True, exist_ok=True)
    output_path = base / f"{args.dataset}_eval_{args.name}.json"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print(f"Successfully saved {len(results)} {args.name} entries to {output_path}")
