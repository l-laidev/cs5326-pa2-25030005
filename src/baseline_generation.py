import json
from pathlib import Path
import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer


eval_set = load_dataset("tatsu-lab/alpaca_eval", "alpaca_eval", split="eval")

model_id = "Qwen/Qwen2.5-0.5B"
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(
    model_id, torch_dtype=torch.bfloat16, device_map="auto"
)

results = []
for idx, example in enumerate(eval_set):
    instruction = example["instruction"]

    messages = [{"role": "user", "content": instruction}]
    prompt = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )

    inputs = tokenizer([prompt], return_tensors="pt").to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=512,
            do_sample=False,
            repetition_penalty=1.05,
            pad_token_id=tokenizer.eos_token_id,
        )

    prompt_len = len(inputs.input_ids[0])
    generated_ids = outputs[0][prompt_len:]
    response = tokenizer.decode(generated_ids, skip_special_tokens=True)

    results.append({
        "id": str(idx),
        "instruction": instruction,
        "output": response,
        "generator": "qwen2.5-0.5b-base",
        "dataset": "alpaca_eval"
    })

base = Path("outputs")
base.mkdir(parents=True, exist_ok=True)
output_path = base / "alpaca_eval_baseline.json"
with open(output_path, "w", encoding="utf-8") as f:
    json.dump(results, f, indent=2, ensure_ascii=False)

print(f"Successfully saved {len(results)} baseline entries to {output_path}")
