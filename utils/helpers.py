# Imports 
import os
import json
import argparse
import time
# import ipdb
import spacy
import torch
import optuna
import pickle
import re

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sb
import numpy as np
from tqdm import tqdm
from collections import deque

from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, classification_report

from transformers import RobertaTokenizer, BertModel, TransfoXLTokenizer, TransfoXLModel, AdamW
from transformers import BigBirdTokenizer, BigBirdForSequenceClassification
from transformers import BertTokenizer, BertForSequenceClassification
from transformers import LongformerTokenizer, LongformerForSequenceClassification
from transformers import XLNetTokenizer, XLNetForSequenceClassification

from torch import nn, optim


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# def tokenize_and_cut(sentence):
#     tokens = tokenizer.tokenize(sentence) 
#     tokens = tokens[:max_input_length-2]
#     return tokens

processors = {
      "Bert": [BertTokenizer, BertForSequenceClassification, "bert-base-uncased"],
      "SciBert": [BertTokenizer, BertForSequenceClassification, "allenai/scibert_scivocab_uncased"],
      "XLNet": [XLNetTokenizer, XLNetForSequenceClassification, "xlnet-base-cased"],
      "BigBird": [BigBirdTokenizer, BigBirdForSequenceClassification, "google/bigbird-roberta-base"],
      "Longformer": [LongformerTokenizer, LongformerForSequenceClassification, "allenai/longformer-base-4096"],
    }

def count_parameters(model):
    return (sum(p.numel() for p in model.parameters() if p.requires_grad), sum(p.numel() for p in model.parameters() if not p.requires_grad))

def epoch_time(start_time, end_time):
    elapsed_time = end_time - start_time
    elapsed_mins = int(elapsed_time / 60)
    elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
    return elapsed_mins, elapsed_secs 


class AverageMeter(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
    
    
def train(model, iterator, optimizer, scheduler, epoch):
    
    model.train()
    
    train_loss = AverageMeter()
    train_acc = AverageMeter()
    train_macro_p = AverageMeter()
    train_macro_r = AverageMeter()
    train_macro_f1 = AverageMeter()
    train_micro_p = AverageMeter()
    train_micro_r = AverageMeter()
    train_micro_f1 = AverageMeter()
    
    for batch_idx, (pair_token_ids, mask_ids, seg_ids, y) in tqdm(enumerate(iterator), total=len(iterator)):
        optimizer.zero_grad()
        
        pair_token_ids = pair_token_ids.to(device)
        mask_ids = mask_ids.to(device)
        seg_ids = seg_ids.to(device)
        labels = y.to(device)

        outputs = model(pair_token_ids, 
                        token_type_ids=seg_ids, 
                        attention_mask=mask_ids, 
                        labels=labels)
            
        loss = outputs.loss
        prediction = outputs.logits

        # loss, prediction = model(pair_token_ids, 
        #                     token_type_ids=seg_ids, 
        #                     attention_mask=mask_ids, 
        #                     labels=labels).values()          
        
        # loss.backward()

        if torch.cuda.device_count() > 1:
            loss.sum().backward()
        else:
            loss.backward()   
        # New
        # nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        optimizer.step()
        # New
        # scheduler.step()
        # New
        # optimizer.zero_grad()
                
        prediction = torch.log_softmax(prediction, dim=1).argmax(dim=1)       

        train_acc.update(prediction.eq(labels.view_as(prediction)).sum().item()/len(labels)) # accuracy_score(labels.cpu(), prediction.cpu())
        # train_loss.update(loss.item())  
        
        if torch.cuda.device_count() > 1:
            train_loss.update(loss.sum().item())
        else:
            train_loss.update(loss.item())

        train_macro_p.update(precision_score(labels.cpu(), prediction.cpu(), average ='macro'))
        train_macro_r.update(recall_score(labels.cpu(), prediction.cpu(), average ='macro'))
        train_macro_f1.update(f1_score(labels.cpu(), prediction.cpu(), average ='macro'))
        # metrics_dict = classification_report(labels.cpu(), prediction.cpu(), labels=[1, 0], output_dict=True)['macro avg']
        # train_macro_p.update(metrics_dict['precision'])
        # train_macro_r.update(metrics_dict['recall'])
        # train_macro_f1.update(metrics_dict['f1-score'])
        train_micro_p.update(precision_score(labels.cpu(), prediction.cpu(), average ='micro'))
        train_micro_r.update(recall_score(labels.cpu(), prediction.cpu(), average ='micro'))
        train_micro_f1.update(f1_score(labels.cpu(), prediction.cpu(), average ='micro'))
        
        if (batch_idx + 1) % 1000 == 0:
            print(f"[epoch {epoch+1}] [iter {(batch_idx + 1)}/{len(iterator)}]")
            print('------------------------------------------------------------')
            print(f"Train Accuracy Score: {train_acc.avg}; Train loss : {train_loss.avg}")
            print(f"Macro Precision: {train_macro_p.avg}; Macro Recall : {train_macro_r.avg}; Macro F1 : {train_macro_f1.avg}")
            print(f"Micro Precision: {train_micro_p.avg}; Micro Recall : {train_micro_r.avg}; Micro F1 : {train_micro_f1.avg}")
            print('------------------------------------------------------------')


    print(f"[epoch {epoch+1}] [iter {(batch_idx + 1)}/{len(iterator)}]")
    print('------------------------------------------------------------')
    print(f"Macro Precision: {train_macro_p.avg}; Macro Recall : {train_macro_r.avg}; Macro F1 : {train_macro_f1.avg}")
    print(f"Micro Precision: {train_micro_p.avg}; Micro Recall : {train_micro_r.avg}; Micro F1 : {train_micro_f1.avg}")    
    print('------------------------------------------------------------')

    return train_loss.avg, train_acc.avg, train_macro_p.avg, train_macro_r.avg, train_macro_f1.avg, train_micro_p.avg, train_micro_r.avg, train_micro_f1.avg



def evaluate(model, iterator, optimizer):
        
    model.eval()
    val_loss = AverageMeter()
    val_acc = AverageMeter()
    val_macro_p = AverageMeter()
    val_macro_r = AverageMeter()
    val_macro_f1 = AverageMeter()
    val_micro_p = AverageMeter()
    val_micro_r = AverageMeter()
    val_micro_f1 = AverageMeter()

    with torch.no_grad():
    
        for batch_idx, (pair_token_ids, mask_ids, seg_ids, y) in tqdm(enumerate(iterator), total=len(iterator)):
#             optimizer.zero_grad()
            pair_token_ids = pair_token_ids.to(device)
            mask_ids = mask_ids.to(device)
            seg_ids = seg_ids.to(device)
            labels = y.to(device)

            outputs = model(pair_token_ids, 
                        token_type_ids=seg_ids, 
                        attention_mask=mask_ids, 
                        labels=labels)
            
            loss = outputs.loss
            prediction = outputs.logits

            # loss, prediction = model(pair_token_ids, 
            #                     token_type_ids=seg_ids, 
            #                     attention_mask=mask_ids, 
            #                     labels=labels).values()

            prediction = torch.log_softmax(prediction, dim=1).argmax(dim=1)

            val_acc.update(prediction.eq(labels.view_as(prediction)).sum().item()/len(labels)) # accuracy_score(labels.cpu(), prediction.cpu())
            val_loss.update(loss.item())  

            val_macro_p.update(precision_score(labels.cpu(), prediction.cpu(), average ='macro'))
            val_macro_r.update(recall_score(labels.cpu(), prediction.cpu(), average ='macro'))
            val_macro_f1.update(f1_score(labels.cpu(), prediction.cpu(), average ='macro'))
            # metrics_dict = classification_report(labels.cpu(), prediction.cpu(), labels=[1, 0], output_dict=True)['macro avg']
            # val_macro_p.update(metrics_dict['precision'])
            # val_macro_r.update(metrics_dict['recall'])
            # val_macro_f1.update(metrics_dict['f1-score'])
            val_micro_p.update(precision_score(labels.cpu(), prediction.cpu(), average ='micro'))
            val_micro_r.update(recall_score(labels.cpu(), prediction.cpu(), average ='micro'))
            val_micro_f1.update(f1_score(labels.cpu(), prediction.cpu(), average ='micro'))
                      
    val_macro_avg_p, val_macro_avg_r, val_macro_avg_f1 = val_macro_p.avg, val_macro_r.avg, val_macro_f1.avg 
    val_micro_avg_p, val_micro_avg_r, val_micro_avg_f1 = val_micro_p.avg, val_micro_r.avg, val_micro_f1.avg 

    print('------------------------------------------------------------')
    print(f"Validation Accuracy Score : {val_acc.avg}; Vadidation loss : {val_loss.avg}")
    print(f"Macro Precision : {val_macro_avg_p}; Macro Recall : {val_macro_avg_r}; Macro F1 : {val_macro_avg_f1}")
    print(f"Micro Precision : {val_micro_avg_p}; Micro Recall : {val_micro_avg_r}; Micro F1 : {val_micro_avg_f1}")
    print('------------------------------------------------------------')
    
    return val_loss.avg, val_acc.avg, val_macro_avg_p, val_macro_avg_r, val_macro_avg_f1, val_micro_avg_p, val_micro_avg_r, val_micro_avg_f1

def predict_TDM_from_pdf(model, tokenizer, iterator, model_name, output_path):
    model.eval()
    with torch.no_grad():
    
        for batch_idx, (pair_token_ids, mask_ids, seg_ids, y) in tqdm(enumerate(iterator), total=len(iterator)):
            pair_token_ids = pair_token_ids.to(device)
            mask_ids = mask_ids.to(device)
            seg_ids = seg_ids.to(device)
            labels = y.to(device)

            outputs = model(pair_token_ids, 
                        token_type_ids=seg_ids, 
                        attention_mask=mask_ids, 
                        labels=labels)
            
            loss = outputs.loss
            prediction = outputs.logits

            # loss, prediction = model(pair_token_ids, 
            #                     token_type_ids=seg_ids, 
            #                     attention_mask=mask_ids, 
            #                     labels=labels).values()

            prediction_scalled = torch.sigmoid(prediction)
            # prediction_scalled = torch.log_softmax(prediction, dim=1).argmax(dim=1)
            # prediction = torch.log_softmax(prediction, dim=1).argmax(dim=1)
            
            with open(f"{output_path}test_results_{model_name}.tsv", "a+", encoding="utf-8") as text_file:
                for true, false in prediction_scalled.cpu():
                    text_file.write(str(true.item())+"\t"+str(false.item())+"\n")
                # text_file.write(str(prediction.cpu())+"\n")

def inference(model, tokenizer, test_path, max_input_length, model_name, output_path):
    model.eval()
    with open(test_path) as f:
        list_prediction_inputs = f.read().splitlines()
    
    with torch.no_grad():
        for input_text in tqdm(list_prediction_inputs, total=len(list_prediction_inputs)):
            
            encoded_review = tokenizer.encode_plus(
                input_text,
                max_length=max_input_length,
                add_special_tokens=True,
                return_token_type_ids=False,
                pad_to_max_length=True,
                return_attention_mask=True,
                return_tensors='pt',
            )
            
            input_ids = encoded_review['input_ids'].to(device)
            attention_mask = encoded_review['attention_mask'].to(device)

            outputs = model(input_ids, attention_mask)
            
            loss = outputs.loss
            prediction = outputs.logits

            prediction_scalled = torch.sigmoid(prediction)

            # loss, prediction = model(pair_token_ids, 
            #                     token_type_ids=seg_ids, 
            #                     attention_mask=mask_ids, 
            #                     labels=labels).values()
            
            with open(f"{output_path}test_results_{model_name}.tsv", "a+", encoding="utf-8") as text_file:
                for true, false in prediction_scalled.cpu():
                    text_file.write(str(true.item())+"\t"+str(false.item())+"\n")

                    
def get_top_n_prediction_label(path_to_test_file, model_name, output_path, n = 5):
    """
    This function return the label with the highest proba
    """
    top5 = deque()
    with open(f"{path_to_test_file}") as f:
        txt_test_files = f.read().splitlines()
    with open(f"{output_path}test_results_{model_name}.tsv") as f:
        txt_prediction_files = f.read().splitlines()
    
    for example, prediction in zip(txt_test_files, txt_prediction_files):
        true_prob, false_prob = prediction.split("\t")
        true_prob, false_prob = float(true_prob), float(false_prob)
        if true_prob > false_prob:
            label = example.split("\t")[2]
            top5.append((label, true_prob))
    results = deque(sorted(top5, key=lambda x: x[1] if x else x, reverse=False), n)
    with open(f"{output_path}test_top_{n}_tdm.tsv", "w+", encoding="utf-8") as text_file:
        for tdm in results:
            text_file.write(f"{tdm[0]}\t{tdm[1]}\n")
    return results

def write_evaluation_result(val_macro_avg_p, val_macro_avg_r, val_macro_avg_f1, val_micro_avg_p, val_micro_avg_r, val_micro_avg_f1, model_name, output_path):
    with open(f"{output_path}evaluation_tdm_{model_name}_results.tsv", "w+", encoding="utf-8") as text_file:
        text_file.write(f"Macro P\tMacro R\t Macro F1\t Micro P\t Micro R\t Micro F1\n")
        text_file.write(f"{val_macro_avg_p}\t{val_macro_avg_r}\t{val_macro_avg_f1}\t{val_micro_avg_p}\t{val_micro_avg_r}\t{val_micro_avg_f1}\n")

def compute_metrics(list_gold, list_pred, model_name, output_path):
    tp, fn, tn, fp = 0, 0, 0, 0
    y = []
    y_pred = []
    # here 0: true, 1: false
    for idx in range(len(list_gold)):
        true_label = list_gold[idx].split("\t")[0]
        true, false = list_pred[idx].split("\t")
        true, false = float(true), float(false)
        if true_label=='true' :
            y.append(0)
            if true > false:
                tp += 1
                y_pred.append(0)
            else:
                fn += 1
                y_pred.append(1)
        else:
            y.append(1)
            if false > true:
                tn += 1
                y_pred.append(1)
            else:
                fp += 1    
                y_pred.append(0)

    print(f"classification_report sklearn  here 0: true, 1: false")

    val_macro_avg_p = precision_score(y, y_pred, average ='macro')
    val_macro_avg_r = recall_score(y, y_pred, average ='macro')
    val_macro_avg_f1 = f1_score(y, y_pred, average ='macro')
    val_micro_avg_p = precision_score(y, y_pred, average ='micro')
    val_micro_avg_r = recall_score(y, y_pred, average ='micro')
    val_micro_avg_f1 = f1_score(y, y_pred, average ='micro')
    
    with open(f"{output_path}evaluation_tdm_{model_name}_results.tsv", "w+", encoding="utf-8") as text_file:
        text_file.write(f"Macro P\tMacro R\t Macro F1\t Micro P\t Micro R\t Micro F1\n")
        text_file.write(f"{val_macro_avg_p}\t{val_macro_avg_r}\t{val_macro_avg_f1}\t{val_micro_avg_p}\t{val_micro_avg_r}\t{val_micro_avg_f1}\n")

        text_file.write(f"{classification_report(y, y_pred, target_names=['true', 'false'], output_dict=False)}")

def get_start_lenght(dictionary, limit="", title="", plot=False):
    # Stats
    len_context= []
    for context in dictionary.values():
        len_context.append(len(context.split()))
    
    print(f"Context TDM limit {limit}:")
    print(f"Mean lenght: {np.mean(len_context)}")
    print(f"Max lenght: {np.max(len_context)}")
    print(f"Min lenght: {np.min(len_context)}")
    print(f"Std lenght: {np.std(len_context)}")
    
    if plot:
        x = np.arange(1, len(len_context)+1, 1)
        y = len_context

        plt.plot(x, y)

        plt.title(title)
        plt.xlabel("number of papers")
        plt.ylabel("lenght DocTAET")
        plt.savefig(fname=re.sub(r"[0-9]+", '', title).strip())
        plt.show()
    
    return len_context
