from transformers import AutoTokenizer
from trl import DPOTrainer, DPOConfig
from peft import LoraConfig, TaskType
from datasets import load_dataset

from common import apply_prompt_format


def to_conversational(example):
    def _convert(msg, role):
        return [
            {
                "role": role,
                "content": msg,
            }
        ]
    
    return {
        "prompt": apply_prompt_format(_convert(example["prompt"], "user"), add_generation_prompt=False),
        "chosen": apply_prompt_format(_convert(example["chosen"], "assistant"), add_generation_prompt=False),
        "rejected": apply_prompt_format(_convert(example["rejected"], "assistant"), add_generation_prompt=False),
    }

if __name__ == "__main__":
    dataset = load_dataset("json", data_files="data/persona_lora/ramsay_style_preferences.json")
    dataset = dataset.map(to_conversational)
    
    tokenizer = AutoTokenizer.from_pretrained("HuggingFaceTB/SmolLM2-360M-Instruct")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    trainer = DPOTrainer(
        model="HuggingFaceTB/SmolLM2-360M-Instruct",
        processing_class=tokenizer,
        train_dataset=dataset,
        # eval_dataset=...,
        peft_config=LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=8,
            target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],
            lora_alpha=8,
            lora_dropout=0.05,
        ),
        args=DPOConfig(
            output_dir="checkpoints/dpo/ramsay",
            
            # eval_strategy="steps",
            # eval_steps=100,
            # include_num_input_tokens_seen="non_padding",
            
            save_strategy="steps",
            save_steps=100,
            load_best_model_at_end=True,
            
            seed=42,
            bf16=True,
            max_length=512,
            num_train_epochs=1,
            beta=0.1,
            
            per_device_train_batch_size=1,
            per_device_eval_batch_size=1,
            gradient_accumulation_steps=32,
            
            learning_rate=1e-6,
            lr_scheduler_type="constant",
            max_grad_norm=1.0,
            
            optim="adamw_torch",
            weight_decay=0.1,
            adam_beta1=0.9,
            adam_beta2=0.999,
        )
    )
    
    trainer.train()
    trainer.save_model("./best_ramsay_checkpoint")
 