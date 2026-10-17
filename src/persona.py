from transformers import AutoTokenizer
from trl import DPOTrainer, DPOConfig
from peft import LoraConfig, TaskType
from datasets import load_dataset


def to_conversational(example):
    def _convert(msg, role):
        return [
            {
                "role": role,
                "content": msg,
            }
        ]
    
    return {
        "prompt": _convert(example["prompt"], "user"),
        "chosen": _convert(example["chosen"], "assistant"),
        "rejected": _convert(example["rejected"], "assistant"),
    }

if __name__ == "__main__":
    dataset = load_dataset("json", data_files="data/persona_lora/ramsay_style_preferences.json", split="train")
    train_dataset = dataset.map(to_conversational).shuffle(seed=42)
    
    tokenizer = AutoTokenizer.from_pretrained("HuggingFaceTB/SmolLM2-360M-Instruct")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    lora_rank = 8
    trainer = DPOTrainer(
        model="HuggingFaceTB/SmolLM2-360M-Instruct",
        processing_class=tokenizer,
        train_dataset=train_dataset,
        peft_config=LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=lora_rank,
            target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],  # qkvo -> qv, 0.3575 -> 0.4410 (worse)
            lora_alpha=2*lora_rank,  # 8 -> 16, 0.4675 -> 0.3575
            lora_dropout=0.05,
        ),
        args=DPOConfig(
            output_dir="checkpoints/dpo/ramsay",
            
            include_num_input_tokens_seen="non_padding",
            
            seed=42,
            bf16=True,
            max_length=512,
            num_train_epochs=1,
            beta=0.1,
            
            per_device_train_batch_size=32,
            per_device_eval_batch_size=32,
            
            learning_rate=1e-3,  # 1e-6 -> 1e-3, loss: 0.6943 -> 0.4761
            lr_scheduler_type="constant",
            max_grad_norm=1.0,
            
            optim="adamw_torch",
            weight_decay=1.1,  # 0.1 -> 1.1, 0.4678 -> 0.4675
            adam_beta1=0.5,  # 0.9 -> 0.8 -> 0.5, 0.4761 -> 0.4749 -> 0.4678
            adam_beta2=0.555,  # 0.999 -> 0.888 -> 0.555, 0.4761 -> 0.4749 -> 0.4678
        )
    )
    
    trainer.train()
    trainer.save_model("./checkpoints/final/persona/ramsay_checkpoint")
