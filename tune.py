import os
import torch
import transformers
from transformers import (
    AutoTokenizer, AutoModelForSequenceClassification, AutoModelForMaskedLM,
    Trainer,
    TrainingArguments,
)
from datasets import load_dataset
import evaluate
import argparse

import numpy as np
import random

import GPUtil
import json
import sys


''' 
== Argument Parsing ==
'''
def parse_args():
    parser = argparse.ArgumentParser()
    
    parser.add_argument("--project_dir", type=str, required=True) #directory of project
    
    parser.add_argument("--model", type=str, default="google-bert/bert-base-uncased") #name of model
    
    parser.add_argument("--dataset", type=str, default="yelp_polarity") # name of dataset
    
    parser.add_argument("--text_col", type=str, default="text") #column of text
    parser.add_argument("--label_col", type=str, default="label") #column of labels
    
    parser.add_argument("--batch", type=int, default=None) #set batch size
    parser.add_argument("--epochs", type=int, default=1) #set epoch size

    parser.add_argument("--lr", type=float, default=4e-5) #learning rate

    parser.add_argument("--task", type=str, default="yelp_polarity") #set task type
    parser.add_argument("--train_split", type=str, default="train") #splitting training set
    parser.add_argument("--eval_split", type=str, default="test") #splitting evaluation set

    parser.add_argument("--skip_label_map", action='store_true')
    parser.add_argument("--skip_tokenization", action='store_true')

    #parser.add_argument("--sampling", type=float, default=1) #sampling factor: between 0 and 1--decides how much portion of dataset is sampled
    parser.add_argument("--train_size", type=int, default=None)
    parser.add_argument("--eval_size", type=int, default=None)
    parser.add_argument("--from_local", type=str, default=None, help="Path to .pt file with saved model wrights") #functionality to resume from local
    parser.add_argument('--eval_interval', type=int, default=100)
    parser.add_argument('--metric', type=str, default='accuracy', help="Metric to evaluate (e.g., accuracy, f1, precision, recall)")
    parser.add_argument('--eval_only', action="store_true")
    parser.add_argument('--eval_batch', type=int, default=None)

    if '--arg_file' in sys.argv:
        index = sys.argv.index('--arg_file')
        arg_file_path = sys.argv[index + 1]
        with open(arg_file_path, 'r') as f:
            file_args = [line.strip() for line in f if line.strip() and not line.strip().startswith('#')]
        return parser.parse_args(file_args)
    
    return parser.parse_args()
    
'''
== device & mem capacity check ==
'''
def check_device(batch_size, model_name):
    gpus=GPUtil.getGPUs()
    if not gpus:
        print("[WARNING] No GPU found. Using CPU instead.")
        return 'cpu', batch_size or 4, False
    gpu = gpus[0]
    free_mem = gpu.memoryFree
    print(f"[INFO] Detected GPU: {gpu.name}, Free Memory: {free_mem:.2f} MB.")

    if batch_size is None:
        suggested_batch_size = max(2, int(free_mem/(1024*8)))
        print(f"[INFO] No batch specified. Suggested batch={suggested_batch_size}.")
        return 'cuda', suggested_batch_size, True
    if batch_size>free_mem/(1024*10):
        print("[WARNING] Specified batch size may exceed memory capacity. Consider lowering it.")
    return 'cuda', batch_size, True

'''
== load dataset ==
1.load dataset from dataset_name
2.split training and eval sets
3.down sample each portion with sample rate
'''
def load_data(dataset_name, train_split, eval_split, text_column, label_column, train_size, eval_size, skip_label_map):
    raw_train_ds, raw_eval_ds = load_dataset(
        dataset_name, split=[train_split, eval_split]
    )
    
    if train_size is None or train_size > len(raw_train_ds):
        raise ValueError(f"--train_size must be <= {len(raw_train_ds)}")
    train_indices = random.sample(range(len(raw_train_ds)), train_size)
    train_ds = raw_train_ds.select(train_indices)

    if eval_size is None or eval_size > len(raw_eval_ds):
        raise ValueError(f"--eval_size must be <= {len(raw_eval_ds)}")
    eval_indices = random.sample(range(len(raw_eval_ds)), eval_size)
    eval_ds = raw_eval_ds.select(eval_indices)

    if (not skip_label_map):
        train_ds = train_ds.map(lambda x: {'label': int(x[label_column])}, remove_columns=[label_column])
        eval_ds = eval_ds.map(lambda x: {'label': int(x[label_column])}, remove_columns=[label_column])
    return train_ds, eval_ds

def load_eval_data(dataset_name, eval_split, text_column, label_column, eval_size, skip_label_map):
    raw_eval_ds = load_dataset(
        dataset_name, split=eval_split
    )

    if eval_size is None or eval_size > len(raw_eval_ds):
        raise ValueError(f"--eval_size must be <= {len(raw_eval_ds)}")
    eval_indices = random.sample(range(len(raw_eval_ds)), eval_size)
    eval_ds = raw_eval_ds.select(eval_indices)

    if (not skip_label_map):
        eval_ds = eval_ds.map(lambda x: {'label': int(x[label_column])}, remove_columns=[label_column])
    return eval_ds

'''
========
'''

def preprocess(train_ds,eval_ds,tokenizer, text_col):
    def tokenize_fn(examples):
        return tokenizer(examples[text_col],truncation=True)
    tokenized_train_ds = train_ds.map(tokenize_fn, batched=True)
    tokenized_eval_ds = eval_ds.map(tokenize_fn, batched=True)
    return tokenized_train_ds, tokenized_eval_ds

def preprocess_eval(eval_ds, tokenizer, text_col):
    def tokenize_fn(examples):
        return tokenizer(examples[text_col],truncation=True)
    tokenized_eval_ds = eval_ds.map(tokenize_fn, batched=True)
    return tokenized_eval_ds

'''
========
'''

def create_new_task_dir(project_dir):
    os.makedirs(project_dir, exist_ok=True)
    task_id = 1
    while os.path.exists(os.path.join(project_dir, f"task{task_id}")):
        task_id += 1
    task_dir = os.path.join(project_dir, f"task{task_id}")
    os.makedirs(task_dir)
    os.makedirs(os.path.join(task_dir, "checkpoints"))
    return task_dir

def save_eval_results(output_dir, step, metrics):
    path = os.path.join(output_dir, "test_outputs.jsonl")
    record = {"step": step, **metrics}
    with open(path, 'a') as f:
        f.write(json.dumps(record) + "\n")

'''
== initialize model ==
'''
def load_model_and_tokenizer(model_name, resume_path, device):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)
    model.to(device)

    if resume_path:
        metadata_path = resume_path.replace(".pt","_meta.json")
        if os.path.exists(metadata_path):
            with open(metadata_path,'r') as f:
                meta=json.load(f)
            if meta.get("model_name") != model_name:
                raise ValueError(f"Model name mismatch: checkpoint trained with {meta.get('model_name')}, current model is {model_name}")
        state_dict = torch.load(resume_path, map_location=device)
        model.load_state_dict(state_dict)
        print(f"[INFO] Populated model weights from {resume_path}")

    return model, tokenizer

'''
== training routine ==
'''
def train(model, tokenizer, train_ds, eval_ds, args, device, task_dir):
    if (not args.skip_tokenization):
        train_data, eval_data = preprocess(train_ds, eval_ds, tokenizer, args.text_col)
    else:
        train_data, eval_data = train_ds, eval_ds

    training_args = TrainingArguments(
        output_dir = os.path.join(task_dir, "checkpoints"),
        per_device_train_batch_size=args.batch,
        per_device_eval_batch_size=args.batch,
        eval_strategy="steps",
        eval_steps=args.eval_interval,
        save_strategy="steps",
        save_steps=args.eval_interval,       
        learning_rate = args.lr,
        num_train_epochs=args.epochs,
        weight_decay=0.01,
        save_total_limit = 2,
        load_best_model_at_end=True,
        metric_for_best_model = "accuracy"
    )

    metric=evaluate.load(args.metric, trust_remote_code=True)
    def compute_metrics(eval_pred):
        logits,labels = eval_pred
        predictions = np.argmax(logits, axis = -1)
        return metric.compute(predictions=predictions,references=labels)
    
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_data,
        eval_dataset=eval_data,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics
    )

    trainer.train()

    history = trainer.state.log_history
    out_path = os.path.join(task_dir, "test_outputs.jsonl")
    with open(out_path, "w") as fout:
        for rec in history:
            if "eval_loss" in rec:
                step = rec["step"]
                metrics = {k.replace("eval_", ""): v
                           for k, v in rec.items() if k.startswith("eval_")}
                fout.write(json.dumps({"step": step, **metrics}) + "\n")
    print(f"[INFO] Eval history saved to", out_path)


    model_path = os.path.join(task_dir, "final_model.pt")
    torch.save(model.state_dict(), model_path)
    with open(model_path.replace(".pt","_meta.json"), 'w') as f:
        json.dump({"model_name": args.model}, f)
    print(f"[INFO] Model saved to {model_path}")

    return model

def evaluate_model(model,tokenizer, eval_ds, args, device, task_dir):
    if (not args.skip_tokenization):
        eval_data = preprocess_eval(eval_ds, tokenizer, args.text_col)
    else:
        eval_data = eval_ds
    
    training_args = TrainingArguments(
        output_dir = os.path.join(task_dir),
        per_device_eval_batch_size=args.eval_batch,
        do_eval=True,
        logging_strategy="no",
        eval_strategy="no",
    )
    metric = evaluate.load(args.metric, trust_remove_code=True)
    def compute_metrics(eval_pred):
        logits,labels = eval_pred
        predictions = np.argmax(logits, axis = -1)
        return metric.compute(predictions=predictions,references=labels)

    trainer = Trainer(
        model=model,
        args=training_args,
        eval_dataset=eval_data,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics
    )

    res= trainer.evaluate()
    print(f"Evaluation results on '{args.eval_split}':", res)
    return res


def main():
    args=parse_args()
    #device, batch_size, proceed = check_device(args.batch, args.model)
    #args.batch=batch_size
    #if (not proceed):
        #sys.exit(f"[WARNING] Insufficient resources; termintating.")
    
    if args.eval_only:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        batch_size = args.eval_batch
        model, tokenizer = load_model_and_tokenizer(args.model, args.from_local, device)
        task_dir = create_new_task_dir(args.project_dir)
        eval_ds = load_eval_data(
            args.dataset, 
            args.eval_split, 
            args.text_col, 
            args.label_col, 
            args.eval_size, 
            args.skip_label_map
        )
        _ = evaluate_model(model, tokenizer, eval_ds, args, device, task_dir)
        return
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        batch_size = args.batch
        model, tokenizer = load_model_and_tokenizer(args.model, args.from_local, device)
        task_dir = create_new_task_dir(args.project_dir)
        train_ds, eval_ds=load_data(
            args.dataset, 
            args.train_split, 
            args.eval_split, 
            args.text_col, 
            args.label_col, 
            args.train_size,
            args.eval_size,
            args.skip_label_map
        )
        model=train(model,tokenizer,train_ds,eval_ds, args, device, task_dir)

if __name__ == '__main__':
    main()