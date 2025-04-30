# ===================== Install Libraries =====================
# !pip install -q datasets
# !pip install -i https://pypi.org/simple/ bitsandbytes
# !pip install -q transformers
# !pip install -q peft
# !pip install -q accelerate

# ===================== Qwen-1.5-7B QLoRA Training Script =====================
import torch
from torch.nn.utils.rnn import pad_sequence
from dataclasses import dataclass
from typing import Sequence, Dict
import copy
import os
from os import listdir

from transformers import (
    AutoTokenizer,
    PreTrainedTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
    TrainingArguments,
    Trainer
)
from peft import (
    get_peft_model,
    prepare_model_for_kbit_training,
    LoraConfig,
    TaskType
)

from datasets import load_from_disk

# ==== Constants ====
IGNORE_INDEX = -100
DEFAULT_PAD_TOKEN = "[PAD]"
model_name = "Qwen/CodeQwen1.5-7B"
dataset_path = "/content/drive/MyDrive/packaged_dataset/merged_dataset"

# ==== Data Collator ====
@dataclass
class DataCollatorForCausalLM:
    tokenizer: PreTrainedTokenizer
    source_max_len: int
    target_max_len: int
    train_on_source: bool = False
    predict_with_generate: bool = False

    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        sources = [f"{self.tokenizer.bos_token}{ex['input']}" for ex in instances]
        targets = [f"{ex['output']}{self.tokenizer.eos_token}" for ex in instances]

        tokenized_sources = self.tokenizer(sources, max_length=self.source_max_len, truncation=True, add_special_tokens=False)
        tokenized_targets = self.tokenizer(targets, max_length=self.target_max_len, truncation=True, add_special_tokens=False)

        input_ids, labels = [], []
        for src, tgt in zip(tokenized_sources["input_ids"], tokenized_targets["input_ids"]):
            input_ids.append(torch.tensor(src + tgt))
            labels.append(torch.tensor([IGNORE_INDEX] * len(src) + tgt))

        input_ids = pad_sequence(input_ids, batch_first=True, padding_value=self.tokenizer.pad_token_id)
        labels = pad_sequence(labels, batch_first=True, padding_value=IGNORE_INDEX)
        return {
            "input_ids": input_ids,
            "labels": labels,
            "attention_mask": input_ids.ne(self.tokenizer.pad_token_id)
        }

# ==== Training Function ====
def train():
    # Load dataset
    dataset = load_from_disk(dataset_path)
    dataset = dataset.rename_columns({"description": "input", "code": "output"})
    dataset = dataset.train_test_split(test_size=0.1)

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token

    # Load 4-bit model and prepare for QLoRA
    quant_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16
    )
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        device_map="auto",
        quantization_config=quant_config,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True
    )

    model = prepare_model_for_kbit_training(model)

    lora_config = LoraConfig(
        r=64,
        lora_alpha=16,
        target_modules=["q_proj", "v_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type=TaskType.CAUSAL_LM
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    # Training arguments
    training_args = TrainingArguments(
        output_dir="./output_qwen_qlora",
        per_device_train_batch_size=1,
        gradient_accumulation_steps=8,
        num_train_epochs=1,
        eval_strategy="steps",
        eval_steps=100,
        save_strategy="steps",
        save_steps=200,
        save_total_limit=3,
        logging_steps=10,
        learning_rate=2e-4,
        bf16=True,
        report_to="none",
        remove_unused_columns=False,
        load_best_model_at_end=False
    )

    # Data collator
    collator = DataCollatorForCausalLM(
        tokenizer=tokenizer,
        source_max_len=2048,
        target_max_len=1024
    )

    # Trainer
    trainer = Trainer(
        model=model,
        tokenizer=tokenizer,
        args=training_args,
        train_dataset=dataset["train"],
        eval_dataset=dataset["test"],
        data_collator=collator
    )

    # Resume or train
    if os.path.isdir(training_args.output_dir) and any("checkpoint" in f for f in listdir(training_args.output_dir)):
      trainer.train(resume_from_checkpoint=True)
    else:
      trainer.train()

    # Save LoRA adapter + tokenizer
    model.save_pretrained("./output_qwen_qlora")
    tokenizer.save_pretrained("./output_qwen_qlora")

# ==== Run ====
if __name__ == "__main__":
    train()
