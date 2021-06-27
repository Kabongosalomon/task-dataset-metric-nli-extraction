# Imports 
import os
import json
import argparse
import time
import ipdb
import pickle
import logging
from shutil import copyfile, rmtree, copytree
from filecmp import dircmp

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sb
import numpy as np
from tqdm import tqdm
from collections import deque

# # Internal inport 
from utils.helpers import get_start_lenght

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Run TDM")
    parser.add_argument("-psource", "--path_source_folder", 
        default=None, 
        help="Path to source folder to be cloned in the format of folder clone")
    parser.add_argument("-ptgtclone", "--path_target_clone", 
        default=None, 
        help="Path to the folder to be cloned with a different context from source")
    
    args = parser.parse_args()

    source_path = args.path_source_folder
    target_path = args.path_target_clone

    start_time = time.time()

    if os.path.exists(f"{target_path}twofoldwithunk/") :
        # https://stackoverflow.com/questions/48892772/how-to-remove-a-directory-is-os-removedirs-and-os-rmdir-only-used-to-delete-emp
        rmtree(f"{target_path}twofoldwithunk/")
    
    
    copytree(f"{source_path}twofoldwithunk/", f"{target_path}twofoldwithunk/")
    # !cp ${source_path}/twofoldwithunk ${target_path}

    train_target_f1_pd = pd.read_csv(f"{target_path}twofoldwithunk/fold1/train.tsv", 
                    sep="\t", names=["label", "title", "TDM", "Context"])

    dev_target_f1_pd = pd.read_csv(f"{target_path}twofoldwithunk/fold1/dev.tsv", 
                    sep="\t", names=["label", "title", "TDM", "Context"])

    train_target_f2_pd = pd.read_csv(f"{target_path}twofoldwithunk/fold2/train.tsv", 
                    sep="\t", names=["label", "title", "TDM", "Context"])

    dev_target_f2_pd = pd.read_csv(f"{target_path}twofoldwithunk/fold2/dev.tsv", 
                    sep="\t", names=["label", "title", "TDM", "Context"])

    trainOutput_source_pd = pd.read_csv(f"{source_path}trainOutput.tsv", 
                        sep="\t", names=["label", "title", "TDM", "Context"])

    trainOutput_target_pd = pd.read_csv(f"{target_path}trainOutput.tsv", 
                        sep="\t", names=["label", "title", "TDM", "Context"])


    list_trainOutput_source_pd_uniq = list(trainOutput_source_pd.title.unique())

    list_trainOutput_target_pd_uniq = list(trainOutput_target_pd.title.unique())

    dict_source_paper_context = {}
    for paper in list_trainOutput_source_pd_uniq:
        dict_source_paper_context[paper]=trainOutput_source_pd[trainOutput_source_pd.title==paper].Context.values[0]
        
    dict_target_paper_context = {}
    for paper in list_trainOutput_target_pd_uniq:
        dict_target_paper_context[paper]=trainOutput_target_pd[trainOutput_target_pd.title==paper].Context.values[0]

    print("---------------------------------------------")

    len_context_source = get_start_lenght(dict_source_paper_context, 
                                   limit="Source", 
                                   title="Source")

    len_context_target = get_start_lenght(dict_target_paper_context, 
                                        limit="Target",
                                       title="Target")

    train_target_f1_pd["Context"] = train_target_f1_pd.apply(lambda x : dict_target_paper_context[x['title']] if x['title'] else "None", axis=1)
    dev_target_f1_pd["Context"] = dev_target_f1_pd.apply(lambda x : dict_target_paper_context[x['title']] if x['title'] else "None", axis=1)

    train_target_f2_pd["Context"] = train_target_f2_pd.apply(lambda x : dict_target_paper_context[x['title']] if x['title'] else "None", axis=1)
    dev_target_f2_pd["Context"] = dev_target_f2_pd.apply(lambda x : dict_target_paper_context[x['title']] if x['title'] else "None", axis=1)

    trainOutput_target_pd["Context"] = trainOutput_target_pd.apply(lambda x : dict_target_paper_context[x['title']] if x['title'] else "None", axis=1)


    train_target_f1_pd.to_csv(path_or_buf=f"{target_path}twofoldwithunk/fold1/train.tsv", 
                 sep="\t", header=None, index=False)
    dev_target_f1_pd.to_csv(path_or_buf=f"{target_path}twofoldwithunk/fold1/dev.tsv", 
                 sep="\t", header=None, index=False)

    train_target_f2_pd.to_csv(path_or_buf=f"{target_path}twofoldwithunk/fold2/train.tsv", 
                 sep="\t", header=None, index=False)
    dev_target_f2_pd.to_csv(path_or_buf=f"{target_path}twofoldwithunk/fold2/dev.tsv", 
                 sep="\t", header=None, index=False)
    
    list_train_target_f1_pd_uniq = list(train_target_f1_pd.title.unique())
    list_dev_target_f1_pd_uniq = list(dev_target_f1_pd.title.unique())

    list_train_target_f2_pd_uniq = list(train_target_f2_pd.title.unique())
    list_dev_target_f2_pd_uniq = list(dev_target_f2_pd.title.unique())

    dict_check_target_train_test_paper_context = {}
    for paper in list_train_target_f1_pd_uniq:
        dict_check_target_train_test_paper_context[paper]=train_target_f1_pd[train_target_f1_pd.title==paper].Context.values[0]
    for paper in list_dev_target_f1_pd_uniq:
        dict_check_target_train_test_paper_context[paper]=dev_target_f1_pd[dev_target_f1_pd.title==paper].Context.values[0]
    for paper in list_train_target_f2_pd_uniq:
        dict_check_target_train_test_paper_context[paper]=train_target_f2_pd[train_target_f2_pd.title==paper].Context.values[0]
    for paper in list_dev_target_f2_pd_uniq:
        dict_check_target_train_test_paper_context[paper]=dev_target_f2_pd[dev_target_f2_pd.title==paper].Context.values[0]

    print("---------------------------------------------")

    len_context_source = get_start_lenght(dict_source_paper_context, 
                                   limit="Source", 
                                   title="Source")

    len_context_target = get_start_lenght(dict_check_target_train_test_paper_context, 
                                        limit="Target After Update",
                                       title="Target After Update")

    print("---------------------------------------------")


    runtime = round(time.time() - start_time, 3)
    print("runtime: %s minutes " % (runtime/60))
    print("#####################################")
    print(args)
    print("#####################################")
    print('done.')