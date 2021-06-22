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
from transformers import XLNetTokenizer, TFXLNetForSequenceClassification

from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

# Internal inport 
from utils.helpers import count_parameters, epoch_time, AverageMeter, processors, compute_metrics
from utils.helpers import train, evaluate, predict_TDM_from_pdf, get_top_n_prediction_label, write_evaluation_result

from model.transformer import TransformersNLI


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Run TDM testing on save model")
    parser.add_argument("-pvalid", "--path_valid", default="/nfs/home/kabenamualus/Research/task-dataset-metric-extraction/data/paperwithcode/new/60Neg800unk/twofoldwithunk/fold1/dev.tsv", help="Path to the dev file")
    parser.add_argument("-ptest_res", "--path_test_prediction", default=None, help="Path to the result file")

    parser.add_argument("-pt", "--model_checkpoint", default="/nfs/home/kabenamualus/Research/task-dataset-metric-extraction/Longformer/Model_f1_0.93.pt", help="Path to the best saved model checkpoint")
    parser.add_argument("-bs", "--batch_size", default=6, help="Batch size")
    parser.add_argument("-maxl", "--max_input_len", default=512, help="Manual insert of the max input lenght in case this is not encoded in the model (e.g. XLNet)")
    parser.add_argument("-m", "--model_name", default='bert-base-uncased', help="The name of the model used")
    parser.add_argument("-o", "--output", default="/nfs/home/kabenamualus/Research/task-dataset-metric-extraction/data/paperwithcode/new/jar/10Neg20unk/", help="Output Path to save the trained model and other metadata")

    args = parser.parse_args()

    valid_path = args.path_valid
    model_pt_path = args.model_checkpoint
    model_name = args.model_name
    output_path = args.output
    bs = int(args.batch_size)
    max_input_len = int(args.max_input_len)

    test_prediction_path = args.path_test_prediction

    if not os.path.exists(f"{output_path}"):
        os.mkdir(f"{output_path}")
        

    if test_prediction_path:
        with open(f"{valid_path}") as f:
            list_test_file = f.read().splitlines()

        with open(f"{test_prediction_path}") as f:
            list_test_pred_file = f.read().splitlines()

        assert len(list_test_file) == len(list_test_pred_file), "Error in Lenght valid path <> predict"

        compute_metrics(list_gold=list_test_file, 
                list_pred=list_test_pred_file, 
                model_name=model_name, output_path=output_path)

        print("Done.")
        quit()

    if model_name in processors.keys():
        selected_processor = processors[model_name]
    else:
        print(f"Model not available check selected model only {list(processors.keys())} as supported")
        quit()

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
    
    TDM_dataset = TransformersNLI(tokenizer, max_input_length)

    # Model loading 
    model = selected_processor[1].from_pretrained(
                                    selected_processor[2], num_labels=2)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    if torch.cuda.device_count() > 1:
        print("Let's use", torch.cuda.device_count(), "GPUs!")
        model = nn.DataParallel(model)
    else:
        print(f"Device: {device}")

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

    print(f'The model has {count_parameters(model)[0]:,} trainable parameters')
    print(f'The model has {count_parameters(model)[1]:,} non-trainable parameters')

    start_time = time.time()

    # Reload the best model
    model.load_state_dict(torch.load(model_pt_path))
    
    valid_df = pd.read_csv(f"{valid_path}", 
                   sep="\t", names=["label", "title", "TDM", "Context"])

    valid_df.head()

    if os.path.exists(f'{output_path}valid_loader_{bs}_seq_{max_input_length}.pth'):
        # valid_loader = torch.load(f'{output_path}valid_loader_{bs}_seq_{max_input_length}.pth')
        os.remove(f'{output_path}valid_loader_{bs}_seq_{max_input_length}.pth')
    # else:
    #     valid_loader = TDM_dataset.get_valid_data(valid_df, batch_size=bs, shuffle=True)
    #     # Save dataloader
    #     torch.save(valid_loader, f'{output_path}valid_loader_{bs}_seq_{max_input_length}.pth')

    valid_loader = TDM_dataset.get_valid_data(valid_df, batch_size=bs, shuffle=False)
    
    valid_loss, valid_acc, val_macro_avg_p, val_macro_avg_r, val_macro_avg_f1, val_micro_avg_p, val_micro_avg_r, val_micro_avg_f1 = evaluate(model, valid_loader, optimizer)

    write_evaluation_result(val_macro_avg_p, val_macro_avg_r, val_macro_avg_f1, val_micro_avg_p, val_micro_avg_r, val_micro_avg_f1, model_name, output_path)

    runtime = round(time.time() - start_time, 3)
    print("runtime: %s seconds " % (runtime))
    print('done.')