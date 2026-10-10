from argparse import ArgumentParser
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from datasets import load_dataset
import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer


MODEL_ID = "Qwen/Qwen2.5-0.5B"
device = "cuda" if torch.cuda.is_available() else "cpu"

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--saved-model", type=str, required=True)
    parser.add_argument("--name", type=str, required=True)
    parser.add_argument("--dataset", type=str, required=True, choices=["alpaca", "hh"])
    
    args = parser.parse_args()
    
    model = AutoModelForCausalLM.from_pretrained(
        args.saved_model,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )
    
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    
    def _preproc(example):
        return example["id"], example["instruction"], example["output"]
    
    if args.dataset == "alpaca":
        dataset = load_dataset("json", data_files="generations/alpaca_eval_baseline.json")
    else:
        dataset = load_dataset("json", data_files="generations/hh_eval_baseline.json")
    dataset = dataset.map(_preproc)
    
    dataset = DataLoader(dataset, batch_size=1, shuffle=False)
    
    print("Starting generation...")
    results = []
    with torch.no_grad(), torch.inference_mode():
        for idx, (id, instruction, baseline_response) in tqdm.tqdm(enumerate(dataset), desc="Generating"):
            messages = [{"role": "user", "content": instruction}]
            prompt = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
            inputs = tokenizer([instruction], return_tensors="pt").to(model.device)
            
            outputs = model.generate(
                **inputs,
                max_new_tokens=512,
                do_sample=False,
                repetition_penalty=1.05,
                pad_token_id=tokenizer.eos_token_id,
            )
        
        prompt_len = len(inputs.input_ids[0])
        response = outputs[0][prompt_len:]
        response = tokenizer.decode(response, skip_special_tokens=True)
        
        results.append({
            "id": id,
            "instruction": instruction,
            f"{args.name}": response,
            "baseline": baseline_response,
        })

    base = Path("generations")
    base.mkdir(parents=True, exist_ok=True)
    output_path = base / f"alpaca_eval_{args.name}.json"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print(f"Successfully saved {len(results)} {args.name} entries to {output_path}")
