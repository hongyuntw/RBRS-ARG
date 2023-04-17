from transformers import Seq2SeqTrainingArguments, Seq2SeqTrainer, AutoTokenizer
import torch
from torch.utils.data import random_split
import json
from datasets import load_metric
from datasets import load_dataset
import numpy as np
import editdistance as ed
from datasets import load_metric
from transformers import BertTokenizer, BartForConditionalGeneration
import nltk

metric = load_metric("rouge")

data_pre_fix = 'review_generate/unsupervised/data'
train_path = f'{data_pre_fix}/train.json'
eval_path = f'{data_pre_fix}/eval.json'

with open(train_path) as f:
    train_dataset = json.load(f)
with open(eval_path) as f:
    eval_dataset = json.load(f)


print(len(train_dataset))
print(len(eval_dataset))


max_input_length = 512
max_target_length = 128

rating_tokens_map = {
    '1.0' : '<rating_1>',
    '2.0' : '<rating_2>',
    '3.0' : '<rating_3>',
    '4.0' : '<rating_4>',
    '5.0' : '<rating_5>',
}



# defining collator functioon for preparing batches on the fly ..
def data_collator(features:list):
    # list of string
    labels = [f["target"] for f in features]
    # list of list string
    inputs = [f["reviews"] for f in features]
    # list of list score
    reviews_rating = [f['reviews_rating'] for f in features]

    for batch_idx in range(len(inputs)):
        for i in range(len(inputs[batch_idx])):
            rating_token = rating_tokens_map[reviews_rating[batch_idx][i]]
            inputs[batch_idx][i] = rating_token + ' ' + inputs[batch_idx][i]

    inputs = [f' {tokenizer.sep_token} '.join(x) for x in inputs]

    
    batch = tokenizer.prepare_seq2seq_batch(src_texts=inputs, tgt_texts=labels, max_length=max_input_length, max_target_length=max_target_length)

    for k in batch:
        batch[k] = torch.tensor(batch[k])


    return batch


pretrained_model = 'ainize/bart-base-cnn'
tokenizer = AutoTokenizer.from_pretrained(pretrained_model)

special_tokens_dict = {'additional_special_tokens': list(rating_tokens_map.values())}
tokenizer.add_special_tokens(special_tokens_dict)
model = BartForConditionalGeneration.from_pretrained(pretrained_model)
model.resize_token_embeddings(len(tokenizer))

# defining trainer using 🤗

run_name = 'bart-base-cnn-rating-tokens-256'

args = Seq2SeqTrainingArguments(
                        output_dir=f"./checkpoints/{run_name}/",
                        run_name=f'{run_name}',
                        do_train=True,
                        do_eval=True,
                        evaluation_strategy="epoch",
                        # eval_steps=1000,
                        per_device_train_batch_size=8,
                        per_device_eval_batch_size=16,
                        gradient_accumulation_steps=8,
                        learning_rate=5e-5,
                        num_train_epochs=10,
                        logging_dir=f"./logs/{run_name}",
                        save_strategy='epoch',
                        save_total_limit=10,
                        # eval_accumulation_steps=8,
                        prediction_loss_only=True,
                        fp16=True,
                        warmup_steps=500,)


trainer = Seq2SeqTrainer(model=model, 
                args=args, 
                data_collator=data_collator, 
                train_dataset=train_dataset, 
                eval_dataset=eval_dataset,
                tokenizer=tokenizer,)

trainer.train()
# trainer.evaluate()
