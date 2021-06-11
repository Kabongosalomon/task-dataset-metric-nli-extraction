# Imports 

import os
import json
import argparse
import time
import ipdb
import spacy
import torch
import optuna
import pickle


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sb
import numpy as np
from tqdm import tqdm
from collections import deque

import torch.optim as optim
from torchtext import data
import torch.nn as nn
import torch.nn.functional as F
from torchsummary import summary
from torch.utils.data import Dataset, TensorDataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
from transformers import RobertaTokenizer, BertModel, TransfoXLTokenizer, TransfoXLModel, AdamW
from transformers import BigBirdTokenizer, BigBirdForSequenceClassification
from transformers import BertTokenizer, BertForSequenceClassification
from transformers import LongformerTokenizer, LongformerForSequenceClassification
from transformers import XLNetTokenizer, XLNetForSequenceClassification

from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

# Internal inport 
from utils.helpers import tokenize_and_cut, count_parameters, epoch_time, AverageMeter
from utils.helpers import train, evaluate, predict_TDM_from_pdf, get_top_n_prediction_label, write_evaluation_result

from model.transformer import TransformersNLI

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Run TDM")
    parser.add_argument("-ptrain", "--path_train", default="/nfs/home/kabenamualus/Research/task-dataset-metric-extraction/data/paperwithcode/new/60Neg800unk/twofoldwithunk/fold1/train.tsv", help="path to train file")
    parser.add_argument("-pvalid", "--path_valid", default="/nfs/home/kabenamualus/Research/task-dataset-metric-extraction/data/paperwithcode/new/60Neg800unk/twofoldwithunk/fold1/dev.tsv", help="Path to the dev file")
    parser.add_argument("-m", "--model_name", default="SciBert", help="Huggingface model name")
    parser.add_argument("-ne", "--numb_epochs", default=2, help="Number of Epochs")
    parser.add_argument("-bs", "--batch_size", default=32, help="Batch size")
    parser.add_argument("-o", "--output", default="/nfs/home/kabenamualus/Research/task-dataset-metric-extraction/data/paperwithcode/new/60Neg800unk/twofoldwithunk/fold1/", help="Output Path to save the trained model and other metadata")

    args = parser.parse_args()

    train_path = args.path_train
    valid_path = args.path_valid
    N_EPOCHS = args.numb_epochs
    model_name = args.model_name
    output_path = args.output
    bs = args.batch_size

    processors = {
      "Bert": [BertTokenizer, BertForSequenceClassification, "bert-base-uncased"],
      "SciBert": [BertTokenizer, BertForSequenceClassification, "allenai/scibert_scivocab_uncased"],
      "XLNet": [XLNetTokenizer, XLNetForSequenceClassification, "xlnet-base-cased"],
      "BigBird": [BigBirdTokenizer, BigBirdForSequenceClassification, "google/bigbird-roberta-base"],
      "Longformer": [LongformerTokenizer, LongformerForSequenceClassification, "allenai/longformer-base-4096"],
    }

    if model_name in processors.keys():
        selected_processor = processors[model_name]
    else:
        Print(f"Model not available check selected model only {list(processors.keys())} as supported")
        quit()

    start_time = time.time()

    if model_name == "SciBert":
        tokenizer = selected_processor[0].from_pretrained("bert-base-uncased")
    else:
        tokenizer = selected_processor[0].from_pretrained(selected_processor[2])
    

    init_token = tokenizer.cls_token
    eos_token = tokenizer.sep_token
    pad_token = tokenizer.pad_token
    unk_token = tokenizer.unk_token

    init_token_idx = tokenizer.convert_tokens_to_ids(init_token)
    eos_token_idx = tokenizer.convert_tokens_to_ids(eos_token)
    pad_token_idx = tokenizer.convert_tokens_to_ids(pad_token)
    unk_token_idx = tokenizer.convert_tokens_to_ids(unk_token)

    if model_name == "SciBert":
        max_input_length = tokenizer.max_model_input_sizes["bert-base-uncased"]
    else:
        max_input_length = tokenizer.max_model_input_sizes[selected_processor[2]]
    

    print(f"Maximun sequence lenght {max_input_length}")

    train_df = pd.read_csv(f"{train_path}", 
                    sep="\t", names=["label", "title", "TDM", "Context"])

    valid_df = pd.read_csv(f"{valid_path}", 
                   sep="\t", names=["label", "title", "TDM", "Context"])

    print(train_df.head())
    print(valid_df.head())

    TDM_dataset = TransformersNLI(tokenizer, max_input_length)

    train_loader = TDM_dataset.get_train_data(train_df, batch_size=bs, shuffle=True)
    valid_loader = TDM_dataset.get_valid_data(valid_df, batch_size=bs, shuffle=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    # model = BertForSequenceClassification.from_pretrained(model_key, num_labels=2)
    model = selected_processor[1].from_pretrained(selected_processor[2], num_labels=2)
    model = model.to(device)

    param_optimizer = list(model.named_parameters())

    no_decay = ['bias', 'gamma', 'beta']

    optimizer_grouped_parameters = [
        {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)],
        'weight_decay_rate': 0.01},
        {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)],
        'weight_decay_rate': 0.0}
    ]

    optimizer = AdamW(optimizer_grouped_parameters, lr=2e-5, correct_bias=False)

    print(f'The model has {count_parameters(model):,} trainable parameters')

    best_valid_loss = 0.30 #float('inf')
    best_valid_metric_avg = 0.3

    for epoch in range(N_EPOCHS):

        start_time_inner = time.time()
        
        train_loss, train_acc, train_macro_avg_p, train_macro_avg_r, train_macro_avg_f1, train_micro_avg_p, train_micro_avg_r, train_micro_avg_f1 = train(model, train_loader, optimizer, epoch)
        val_macro_avg_p, val_macro_avg_r, val_macro_avg_f1, val_micro_avg_p, val_micro_avg_r, val_micro_avg_f1 = evaluate(model, valid_loader, optimizer)
        
        end_time_inner = time.time()

        epoch_mins, epoch_secs = epoch_time(start_time_inner, end_time_inner)
        
        print(f'Epoch: {epoch+1:02} Final | Epoch Time: {epoch_mins}m {epoch_secs}s')
        print('------------------------------------------------------------')
        print(f"Train Accuracy Score: {train_acc}; Train loss : {train_loss}")
        print(f"Macro Precision: {train_macro_avg_p}; Macro Recall : {train_macro_avg_r}; Macro F1 : {train_macro_avg_f1}")
        print(f"Micro Precision: {train_micro_avg_p}; Micro Recall : {train_micro_avg_r}; Micro F1 : {train_micro_avg_f1}")
        print('------------------------------------------------------------')

        valid_metric_avg = (val_macro_avg_p, val_macro_avg_r+val_macro_avg_f1+val_micro_avg_p+val_micro_avg_r+val_micro_avg_f1)/6
        if valid_metric_avg > best_valid_metric_avg : #and abs(valid_loss - best_valid_loss) < 1e-1
            best_valid_metric_avg = valid_metric_avg
            print('Saving Model ...')
            torch.save(model.state_dict(), f'{output_path}Model_{model_name}_avg_metric_{str(best_valid_metric_avg)[:4]}.pt')
            print('****************************************************************************')
            print('best record: [epoch %d], [val loss %.5f], [val acc %.5f], [val f1 %.5f]' % (epoch, valid_loss, valid_acc, valid_f1))

            print(f"Validation Accuracy Score : {val_acc.avg}; Vadidation loss : {val_loss.avg}")
            print(f"Macro Precision : {val_macro_avg_p}; Macro Recall : {val_macro_avg_r}; Macro F1 : {val_macro_avg_f1}")
            print(f"Micro Precision : {val_micro_avg_p}; Micro Recall : {val_micro_avg_r}; Micro F1 : {val_micro_avg_f1}")
            print('****************************************************************************')

    runtime = round(time.time() - start_time, 3)
    print("runtime: %s seconds " % (runtime))
    print('done.')