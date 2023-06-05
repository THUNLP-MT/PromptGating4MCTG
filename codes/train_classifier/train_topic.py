from datasets import load_dataset
from transformers import AutoTokenizer
from transformers import AutoModelForSequenceClassification, RobertaForSequenceClassification, BertForSequenceClassification
from transformers import TrainingArguments
import numpy as np
import evaluate

metric = evaluate.load("accuracy")

dataset = load_dataset('json', 
                       data_files={'train': '/path/to/processed/yelp/yelp_over_sample/sample_topic/topic_sents_train.txt', 
                                   'test': '/path/to/processed/yelp/yelp_over_sample/sample_topic/topic_sents_test.txt',
                                   'validation': '/path/to/processed/yelp/yelp_over_sample/sample_topic/topic_sents_valid.txt'},
                       streaming=True)

tokenizer = AutoTokenizer.from_pretrained("roberta-large")

def tokenize_function(examples):
    return tokenizer(examples["text"], padding="max_length", truncation=True)

tokenized_datasets = dataset.map(tokenize_function, batched=True)
tokenized_datasets = tokenized_datasets.rename_column("label", "labels")

model = AutoModelForSequenceClassification.from_pretrained("roberta-large", num_labels=3)

training_args = TrainingArguments(
    output_dir='./train_classifier/topic_finetuned_model',          # output directory
    num_train_epochs=4,              # total # of training epochs
    per_device_train_batch_size=8,  # batch size per device during training
    per_device_eval_batch_size=32,   # batch size for evaluation
    warmup_steps=2000,                # number of warmup steps for learning rate scheduler
    weight_decay=0.01,               # strength of weight decay
    logging_dir='./train_classifier/topic_finetuned_model',            # directory for storing logs
    logging_steps=100,
    save_steps=2500,
    evaluation_strategy='steps',
    load_best_model_at_end=True,
    learning_rate=1e-5,
    do_train=True,
    do_eval=True,
    eval_steps=2500,
    fp16=True,
    max_steps=100000,
)

from transformers import Trainer

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    return metric.compute(predictions=predictions, references=labels)

trainer = Trainer(
    model=model,                         # the instantiated 🤗 Transformers model to be trained
    args=training_args,                  # training arguments, defined above
    train_dataset=tokenized_datasets["train"],         # training dataset
    eval_dataset=tokenized_datasets["validation"],            # evaluation dataset
    compute_metrics=compute_metrics,
)

trainer.train()

trainer.evaluate()

trainer.save_model()
