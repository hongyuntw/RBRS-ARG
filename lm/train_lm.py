import torch

import numpy as np
from transformers import DataCollatorForLanguageModeling
from transformers import AutoModelForMaskedLM, AutoModelForCausalLM
from transformers import Trainer, TrainingArguments
from transformers import AutoTokenizer
import pandas as pd
from datasets import list_datasets, load_dataset
from datasets import Dataset

# data_set= "Musical_Instruments_data"
# data_set= "Office_Products_data"
# data_set = 'Video_Games_data'

csv_path = f'dataset/{data_set}/data_train.csv'

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

block_size = 128
max_length = 128
exp_name = 'distilgpt2_128'
# exp_name = 'gpt_metric'
def group_texts(examples):

    examples['labels'] = examples["input_ids"].copy()
    return examples


def tokenize_function(examples):
    return tokenizer(examples["reviews"], padding=True, truncation=True, max_length=max_length)

# model_checkpoint = "distilgpt2"
model_checkpoint = 'gpt2'
tokenizer = AutoTokenizer.from_pretrained(model_checkpoint, use_fast=True)
tokenizer.pad_token = tokenizer.eos_token
model = AutoModelForCausalLM.from_pretrained(model_checkpoint)

df = pd.read_csv(csv_path)
df = df[df['reviews'].notna()]
df = df[df['reviews'] != '']
df['reviews_len'] = df['reviews'].map(str).apply(len)
df = df[df['reviews_len'] > 20]
datasets = Dataset.from_pandas(df)

tokenized_datasets = datasets.map(tokenize_function, batched=True, num_proc=4, remove_columns=['item_id', 'user_id', 'reviews', 'ratings', 'reviews_len'])

lm_datasets = tokenized_datasets.map(
    group_texts,
    batched=True,
    batch_size=1000,
    num_proc=4,
)

print(lm_datasets)
print(len(lm_datasets))
lm_datasets = lm_datasets.train_test_split(test_size=0.15)
print(len(lm_datasets['train']))
print(len(lm_datasets['test']))

model_name = model_checkpoint.split("/")[-1]
training_args = TrainingArguments(
    output_dir=f"{exp_name}-{data_set}_reviews",
    do_eval=True,
    evaluation_strategy = "epoch",
    per_device_train_batch_size=8,
    per_device_eval_batch_size=16,
    gradient_accumulation_steps=8,
    learning_rate=3e-5,
    weight_decay=0.01,
    num_train_epochs=2,
    save_total_limit=10,
    save_strategy = "epoch",
    logging_dir='./logs', 
)

trainer = Trainer(
    model=model.to(device),
    args=training_args,
    train_dataset=lm_datasets["train"],
    eval_dataset=lm_datasets["test"],
)
trainer.train()
