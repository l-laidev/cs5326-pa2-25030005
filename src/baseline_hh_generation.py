import json
from pathlib import Path
import torch
from datasets import load_dataset
import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from src.common import apply_prompt_format


MODEL_ID = "Qwen/Qwen2.5-0.5B"
if __name__ == "__main__":
    eval_set = load_dataset("json", data_files="eval_sets/hh_rlhf_eval_300_seed42.json")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.bfloat16,
        device_map="auto"
    )

    results = []
    for idx, example in tqdm.tqdm(enumerate(eval_set), desc="Generating baseline responses"):
        instruction = example["instruction"]

        messages = [{"role": "user", "content": instruction}]
        prompt = apply_prompt_format(messages, add_generation_prompt=True)
        inputs = tokenizer([prompt], return_tensors="pt").to(model.device)

        with torch.no_grad(), torch.inference_mode():
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
            "id": str(idx),
            "instruction": instruction,
            "output": response,
            "generator": "qwen2.5-0.5b-base",
        })

    base = Path("generations")
    base.mkdir(parents=True, exist_ok=True)
    output_path = base / "hh_eval_baseline.json"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print(f"Successfully saved {len(results)} baseline entries to {output_path}")
