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
import logging

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sb
import numpy as np
from tqdm import tqdm
from collections import deque


import torch.optim as optim
from torchtext import data

import torch.nn.functional as F
from torchsummary import summary
from torch.utils.data import Dataset, TensorDataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
from transformers import RobertaTokenizer, BertModel, TransfoXLTokenizer, TransfoXLModel, AdamW, get_linear_schedule_with_warmup
from transformers import BigBirdTokenizer, BigBirdForSequenceClassification
from transformers import BertTokenizer, BertForSequenceClassification
from transformers import LongformerTokenizer, LongformerForSequenceClassification
from transformers import XLNetTokenizer, XLNetForSequenceClassification

from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

# Internal inport 
from utils.helpers import count_parameters, epoch_time, AverageMeter, processors #,tokenize_and_cut
from utils.helpers import train, evaluate, predict_TDM_from_pdf, get_top_n_prediction_label, write_evaluation_result

from model.transformer import TransformersNLI

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Run TDM")
    parser.add_argument("-ptrain", "--path_train", default="/nfs/home/kabenamualus/Research/task-dataset-metric-extraction/data/paperwithcode/new/60Neg800unk/twofoldwithunk/fold1/train.tsv", help="path to train file")
    parser.add_argument("-pvalid", "--path_valid", default="/nfs/home/kabenamualus/Research/task-dataset-metric-extraction/data/paperwithcode/new/60Neg800unk/twofoldwithunk/fold1/dev.tsv", help="Path to the dev file")
    parser.add_argument("-m", "--model_name", default="SciBert", help="Huggingface model name")
    parser.add_argument("-init_pt", "--model_init_checkpoint", default=None, help="A checkpoint to start training the model from")

    parser.add_argument("-ne", "--numb_epochs", default=2, help="Number of Epochs")
    parser.add_argument("-bs", "--batch_size", default=32, help="Batch size")
    parser.add_argument("-maxl", "--max_input_len", default=512, help="Manual insert of the max input lenght in case this is not encoded in the model (e.g. XLNet)")
    parser.add_argument("-o", "--output", default="/nfs/home/kabenamualus/Research/task-dataset-metric-extraction/data/paperwithcode/new/60Neg800unk/twofoldwithunk/fold1/", help="Output Path to save the trained model and other metadata")

    args = parser.parse_args()

    train_path = args.path_train
    valid_path = args.path_valid
    N_EPOCHS = int(args.numb_epochs)
    model_name = args.model_name
    model_init_checkpoint = args.model_init_checkpoint
    output_path = args.output
    bs = int(args.batch_size)
    max_input_len = int(args.max_input_len)

    if not os.path.exists(f"{output_path}"):
        os.mkdir(f"{output_path}")

    logging.info(args)

    if model_name in processors.keys():
        selected_processor = processors[model_name]
    else:
        print(f"Model not available check selected model only {list(processors.keys())} as supported")
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
    
    if not max_input_length:
        max_input_length = max_input_len

    print(f"Maximun sequence lenght {max_input_length}")

    train_df = pd.read_csv(f"{train_path}", 
                    sep="\t", names=["label", "title", "TDM", "Context"])

    valid_df = pd.read_csv(f"{valid_path}", 
                   sep="\t", names=["label", "title", "TDM", "Context"])

    print(train_df.head())
    print(valid_df.head())

    TDM_dataset = TransformersNLI(tokenizer, max_input_length)
    
    if os.path.exists(f'{output_path}train_loader_{bs}_seq_{max_input_length}.pth'):
        train_loader = torch.load(f'{output_path}train_loader_{bs}_seq_{max_input_length}.pth')
        # os.remove(f'{output_path}train_loader_{bs}_seq_{max_input_length}.pth')
    else:
        train_loader = TDM_dataset.get_train_data(train_df, batch_size=bs, shuffle=True)
        # Save dataloader
        torch.save(train_loader, f'{output_path}train_loader_{bs}_seq_{max_input_length}.pth')

    if os.path.exists(f'{output_path}valid_loader_{bs}_seq_{max_input_length}.pth'):
        # valid_loader = torch.load(f'{output_path}valid_loader_{bs}_seq_{max_input_length}.pth')
        os.remove(f'{output_path}valid_loader_{bs}_seq_{max_input_length}.pth')
    # else:
    #     valid_loader = TDM_dataset.get_valid_data(valid_df, batch_size=bs, shuffle=True)
    #     # Save dataloader
    #     torch.save(valid_loader, f'{output_path}valid_loader_{bs}_seq_{max_input_length}.pth')

    # train_loader = TDM_dataset.get_train_data(train_df, batch_size=bs, shuffle=True)
    valid_loader = TDM_dataset.get_valid_data(valid_df, batch_size=bs, shuffle=False)


    # Model loading
    # model = BertForSequenceClassification.from_pretrained(model_key, num_labels=2)
    model = selected_processor[1].from_pretrained(
                                    selected_processor[2], num_labels=2)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    if torch.cuda.device_count() > 1:
        print("Let's use", torch.cuda.device_count(), "GPUs!")
        model = torch.nn.DataParallel(model)
    else:
        print(f"Device: {device}")

    model = model.to(device)

    if model_init_checkpoint:
        # Reload from given checkpoint
        model.load_state_dict(torch.load(model_init_checkpoint))

    param_optimizer = list(model.named_parameters())

    no_decay = ['bias', 'gamma', 'beta']

    optimizer_grouped_parameters = [
        {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)],
        'weight_decay_rate': 0.01},
        {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)],
        'weight_decay_rate': 0.0}
    ]

    optimizer = AdamW(optimizer_grouped_parameters, lr=5e-5, correct_bias=False)
    # optimizer = AdamW(model.parameters(), lr=5e-5, correct_bias=False)

    total_steps = len(train_loader) * N_EPOCHS
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=0,
        num_training_steps=total_steps
        )

    print(f'The model has {count_parameters(model)[0]:,} trainable parameters')
    print(f'The model has {count_parameters(model)[1]:,} non-trainable parameters')

    best_valid_loss = 0.30 #float('inf')
    best_valid_metric_avg = 0.3

    for epoch in range(N_EPOCHS):

        start_time_inner = time.time()
        
        train_loss, train_acc, train_macro_avg_p, train_macro_avg_r, train_macro_avg_f1, train_micro_avg_p, train_micro_avg_r, train_micro_avg_f1 = train(model, train_loader, optimizer, scheduler, epoch)
        valid_loss, valid_acc, val_macro_avg_p, val_macro_avg_r, val_macro_avg_f1, val_micro_avg_p, val_micro_avg_r, val_micro_avg_f1 = evaluate(model, valid_loader, optimizer)
        
        end_time_inner = time.time()

        epoch_mins, epoch_secs = epoch_time(start_time_inner, end_time_inner)
        
        print(f'Epoch: {epoch+1:02} Final | Epoch Time: {epoch_mins}m {epoch_secs}s')
        print('------------------------------------------------------------')
        print(f"Train Accuracy Score: {train_acc}; Train loss : {train_loss}")
        print(f"Macro Precision: {train_macro_avg_p}; Macro Recall : {train_macro_avg_r}; Macro F1 : {train_macro_avg_f1}")
        print(f"Micro Precision: {train_micro_avg_p}; Micro Recall : {train_micro_avg_r}; Micro F1 : {train_micro_avg_f1}")
        print('------------------------------------------------------------')

        # TODO: Shoul we focus only on precision or only f1 score ?
        # valid_metric_avg = (val_macro_avg_p + val_macro_avg_r + val_macro_avg_f1 + val_micro_avg_p + val_micro_avg_r + val_micro_avg_f1)/6
        valid_metric_avg = (val_macro_avg_f1 + val_micro_avg_f1)/2
        # valid_metric_avg = val_macro_avg_p

        if valid_metric_avg > best_valid_metric_avg : #and abs(valid_loss - best_valid_loss) < 1e-1
            best_valid_metric_avg = valid_metric_avg

            print('Saving Model ...')
            torch.save(model.state_dict(), f'{output_path}Model_{model_name}_Epoch_{epoch}_avg_metric_{round(best_valid_metric_avg, 4)}.pt')

            print('****************************************************************************')
            print('best record: [epoch %d], [val loss %.5f], [val acc %.5f], [val avg. metric %.5f]' % (epoch, valid_loss, valid_acc, valid_metric_avg))
            print(f"Macro Precision : {val_macro_avg_p}; Macro Recall : {val_macro_avg_r}; Macro F1 : {val_macro_avg_f1}")
            print(f"Micro Precision : {val_micro_avg_p}; Micro Recall : {val_micro_avg_r}; Micro F1 : {val_micro_avg_f1}")
            print('****************************************************************************')

    runtime = round(time.time() - start_time, 3)
    print("runtime: %s minutes " % (runtime/60))
    print("#####################################")
    print(args)
    print("#####################################")
    print('done.')