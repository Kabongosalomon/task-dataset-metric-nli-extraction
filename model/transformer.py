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
from sklearn.metrics import accuracy_score, f1_score, precision_recall_fscore_support


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class TransformersNLI(Dataset):
    def __init__(self, tokenizer, max_input_length):
        self.label_dict = {'True': 0, 'False': 1} # Default {'entailment': 0, 'contradiction': 1, 'neutral': 2}
        self.tokenizer = tokenizer
        self.max_input_length = max_input_length        
        
    def _truncate_seq_pair(self, tokens_a, tokens_b, max_length):
        """Truncates a sequence pair in place to the maximum length."""

        # This is a simple heuristic which will always truncate the longer sequence
        # one token at a time. This makes more sense than truncating an equal percent
        # of tokens from each, since if one sequence is very short then each token
        # that's truncated likely contains more information than a longer sequence.
        while True:
            total_length = len(tokens_a) + len(tokens_b)
            if total_length <= max_length:
                break
            if len(tokens_a) > len(tokens_b):
                tokens_a.pop()
            else:
                tokens_b.pop()
        
    def load_data(self, df):
        MAX_LEN = self.max_input_length
        token_ids = []
        mask_ids = []
        seg_ids = []
        y = []

        premise_list = df['TDM'].to_list()           # df['sentence1'].to_list()
        hypothesis_list = df['Context'].to_list()    # df['sentence2'].to_list()
        label_list = df['label'].to_list()           # df['gold_label'].to_list()

        for (premise, hypothesis, label) in tqdm(zip(premise_list, hypothesis_list, label_list), total=len(label_list)):
            premise_id = self.tokenizer.encode(premise, add_special_tokens = False)
            hypothesis_id = self.tokenizer.encode(hypothesis, add_special_tokens = False)
            # ignore the warning as the ong sequence issuw is taken care of here 
            self._truncate_seq_pair(premise_id, hypothesis_id, MAX_LEN-3) # -3 to account for the special characters 
            
            pair_token_ids = [self.tokenizer.cls_token_id] + premise_id \
                            + [self.tokenizer.sep_token_id] + hypothesis_id \
                            + [self.tokenizer.sep_token_id]
            premise_len = len(premise_id)
            hypothesis_len = len(hypothesis_id)

            segment_ids = torch.tensor([0] * (premise_len + 2) + [1] * (hypothesis_len + 1))  # sentence 0 and sentence 1
            attention_mask_ids = torch.tensor([1] * (premise_len + hypothesis_len + 3))  # mask padded values

            token_ids.append(torch.tensor(pair_token_ids))
            seg_ids.append(segment_ids)
            mask_ids.append(attention_mask_ids)
            # we have str(label) to have the key work proprely 
            y.append(self.label_dict[str(label)]) # y.append(self.label_dict[label]) 
            
        token_ids = pad_sequence(token_ids, batch_first=True)
        mask_ids = pad_sequence(mask_ids, batch_first=True)
        seg_ids = pad_sequence(seg_ids, batch_first=True)
        y = torch.tensor(y)
        dataset = TensorDataset(token_ids, mask_ids, seg_ids, y)

        print(len(dataset))

        return dataset

    def get_train_data(self, train_df, batch_size=32, shuffle=True):
        train_data = self.load_data(train_df)
                    
        train_loader = DataLoader(
            train_data,
            shuffle=shuffle,
            batch_size=batch_size
            )

        return train_loader

    def get_valid_data(self, valid_df, batch_size=32, shuffle=True):
        valid_data = self.load_data(valid_df)
                    
        valid_loader = DataLoader(
            valid_data,
            shuffle=shuffle,
            batch_size=batch_size
            )

        return valid_loader

    def get_inference_data(self, test_df, batch_size=32, shuffle=False):
        test_data = self.load_data(test_df)
                    
        test_loader = DataLoader(
            test_data,
            shuffle=shuffle,
            batch_size=batch_size
            )

        return test_loader