import json
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import argparse
from datasets import load_dataset, load_metric, concatenate_datasets, Dataset
from transformers import Trainer, TrainingArguments
import json
import evaluate
import numpy as np


metric = evaluate.load("f1")

parser = argparse.ArgumentParser()

args = parser.parse_args()

def compute_metrics_bin(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    return metric.compute(predictions=predictions, references=labels)

def compute_metrics_multi(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    return metric.compute(predictions=predictions, references=labels, average='macro')
compute_metrics = [compute_metrics_bin, compute_metrics_multi]

dataset = [load_dataset('json', data_files={'test': '/home/lzj/lzj/plug4MSG/data/yelp/yelp_over_sample/sample_sentiment/senti_sents_test.txt'}),
           load_dataset('json', data_files={'test': '/home/lzj/lzj/plug4MSG/data/yelp/yelp_over_sample/sample_topic/topic_sents_test.txt'})]

tokenizer = AutoTokenizer.from_pretrained("roberta-large")

def tokenize_function(examples):
    return tokenizer(examples["text"], padding="max_length", truncation=True)

model_list = ['/home/lzj/lzj/plug4MSG/pluginMSG/train_classifier/sentiment_finetuned_model', '/home/lzj/lzj/plug4MSG/pluginMSG/train_classifier/topic_finetuned_model']
label_num = [2, 3]

tokenized_datasets = [i.map(tokenize_function, batched=True) for i in dataset]
tokenized_datasets = [i.rename_column("label", "labels") for i in tokenized_datasets]

test_args = TrainingArguments(
    output_dir='logs',
    do_train = False,
    do_predict = True,
    no_cuda = False,
    per_device_eval_batch_size=64,
    dataloader_drop_last = False,
    report_to='none'
)

train_out = []
for i in range(2):
    model = AutoModelForSequenceClassification.from_pretrained(model_list[i], num_labels=label_num[i])
    eval_dataset = tokenized_datasets[i]

    trainer = Trainer(
        model=model,
        args = test_args,
        compute_metrics=compute_metrics[i],
    )
    train_out.append(trainer.evaluate(eval_dataset['test'])['eval_f1'])

print(train_out)
