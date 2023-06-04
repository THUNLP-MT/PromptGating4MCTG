import json
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import argparse
from datasets import load_dataset, load_metric, concatenate_datasets, Dataset
from transformers import Trainer, TrainingArguments
import json
import evaluate
import numpy as np


metric = evaluate.load("accuracy")

parser = argparse.ArgumentParser()
parser.add_argument("--file_loc", type=str, default="./test.txt")
parser.add_argument(
    "--specify",
    type=str,
    default=None
)

args = parser.parse_args()

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    return metric.compute(predictions=predictions, references=labels)

dataset = {'label':[], 'text':[]}
with open(args.file_loc, 'r') as f:
    for line in f.readlines():
        label, sent = json.loads(args.specify), line.strip()
        dataset['label'].append(label)
        dataset['text'].append(sent.strip())

src_txt = dataset['text']
dataset = Dataset.from_dict(dataset)
tokenizer = AutoTokenizer.from_pretrained("roberta-large")

def tokenize_function(examples):
    return tokenizer(examples["text"], padding="max_length", truncation=True)

model_list = ['/home/lzj/lzj/plug4MSG/pluginMSG/train_classifier/sentiment_finetuned_model', '/home/lzj/lzj/plug4MSG/pluginMSG/train_classifier/topic_finetuned_model']
topics = ["asian","usa","mexi"]
task_list = ['sentiment', 'topic']

if args.specify is not None:
    specify = json.loads(args.specify)

label_num = [2, 3]
sample_rate = [1.0 for _ in src_txt]
for i in range(2):
    if args.specify is not None:
        if specify[i] == -1:
            continue
    model = AutoModelForSequenceClassification.from_pretrained(model_list[i], num_labels=label_num[i])
    eval_dataset = None
    eval_dataset = dataset.map(tokenize_function, batched=True)
    eval_dataset = eval_dataset.map(lambda e: {'labels': e['label'][i]})

    # classify
    test_args = TrainingArguments(
        output_dir='logs',
        do_train = False,
        do_predict = True,
        no_cuda = False,
        per_device_eval_batch_size=64,
        dataloader_drop_last = False,
        report_to='none'
    )
    trainer = Trainer(
        model=model,
        args=test_args,
        compute_metrics=compute_metrics,
        eval_dataset=eval_dataset
    )
    train_out = {}
    train_out['predict'] = trainer.predict(eval_dataset)
    labels = train_out['predict'].predictions.argmax(-1)
    
    for lid, line in enumerate(src_txt):
        if labels[lid] != specify[i]:
            sample_rate[lid] *= 0

# sample from sample rate
cnt = 0
for lid, line in enumerate(src_txt):
    if cnt == 25:
        print()
        cnt = 0
    cnt += 1
    if sample_rate[lid] < 1:
        if np.random.rand() > sample_rate[lid]:
            continue        
    print(line)
