import json
import requests
import ipdb
from collections import defaultdict
import os
import argparse
import time
import ipdb
import pickle
import logging

implementations = {}
notfound = []
tasks = {}
categories = {}
evaluations = {}
datasets = {}
counter = 1112
metrics = {}
models = {}

TDM_taxonomy = defaultdict(lambda: 0 )
TDMs_taxonomy = defaultdict(lambda: 0 )

# keep_uniq_TDMs = set()
uniq_paper_link = set()
uniq_datasetAnnotation = defaultdict(lambda: [] )
uniq_taskAnnotation = defaultdict(lambda: [] )
uniq_resultsAnnotation = defaultdict(lambda: [] )


def readFile(path):
    """
    Reads the json file and returns a dictionary containing the json object
    
    Parameters
    ----------
    path : str
        The local path of the json file
    """
    
    file = open(path)
    json_string = json.load(file)
    file.close()
    return json_string


# this is to keep track of occurance of a leaderboard  
def get_TDM_taxonomy(task, dataset, metric):
    TDM_taxonomy[task+"#"+dataset+"#"+metric] += 1

def get_TDMs_taxonomy(task, dataset, metric, score):
    TDMs_taxonomy[task+"#"+dataset+"#"+metric+"#"+score] += 1

def get_title_taxonomy(paper_title):
    paper_name_taxonomy[paper_title] += 1

def parse_TDM_taxonomy(TDM_taxonomy):      
    with open(f"{target_path}TDM_taxonomy.tsv", "a+", encoding="utf-8") as text_file:
        for key, value in TDM_taxonomy.items():
            text_file.write(key+"\t"+str(value)+"\n")

def parse_TDMs_taxonomy(TDMs_taxonomy):      
    with open(f"{target_path}TDMs_taxonomy.tsv", "a+", encoding="utf-8") as text_file:
        for key, value in TDMs_taxonomy.items():
            text_file.write(key+"\t"+str(value)+"\n")

def parse_title_taxonomy(paper_name_taxonomy):
    with open(f"{target_path}paper_name_taxonomy.tsv", "a+", encoding="utf-8") as text_file:
        for key, value in paper_name_taxonomy.items():
            text_file.write(key+"\t"+str(value)+"\n")

def resultsAnnotation(paper_name, task, dataset, metric, score):
    path = f"{target_path}resultsAnnotation.tsv"
    keep_uniq_TDMs = set()
    with open(path, "a+", encoding="utf-8") as text_file:
        if paper_name not in paper_name_taxonomy:
            # if first:
            text_file.write(paper_name+"\t"+task+"#"+dataset+"#"+metric+"#"+score+"\n")
            keep_uniq_TDMs.add(task+"#"+dataset+"#"+metric+"#"+score)
            
        else:
            # TODO: This approach is not optimal nor scalable, need to redo this
            with open(path, 'r',encoding="utf-8") as file:
                # read a list of lines into data
                data = file.readlines()

            for i, key in enumerate(reversed(data)):
                if key.split("\t")[0] == paper_name:
                    if task+"#"+dataset+"#"+metric+"#"+score in keep_uniq_TDMs:
                        continue
                    else:
                        data[len(data)-i-1] = data[len(data)-i-1].replace("\n", '')+\
                            '$'+task+"#"+dataset+"#"+metric+"#"+score+"\n"
                        keep_uniq_TDMs.add(task+"#"+dataset+"#"+metric+"#"+score)
                        break

            # # and write everything back
            with open(path, 'w', encoding="utf-8") as file:
                file.writelines( data )

        # this is to keep track of occurance of a leaderboard     
        get_TDM_taxonomy(task, dataset, metric)
        get_TDMs_taxonomy(task, dataset, metric, score)

        paper_name_taxonomy[paper_name]+=1


def datasetAnnotation(paper_title, dataset):      

    path = f"{target_path}datasetAnnotation.tsv"

    with open(path, "a+", encoding="utf-8") as text_file:

        if paper_title not in paper_title_taxonomy:
            # if first:
            text_file.write(paper_title+"\t"+dataset+"\n")
        else:
            # TODO: This approach is not optimal nor scalable, need to redo this
            with open(path, 'r',encoding="utf-8") as file:
                # read a list of lines into data
                data = file.readlines()

            for i, key in enumerate(reversed(data)):
                if key.split("\t")[0] == paper_title:
                    data[len(data)-i-1] = data[len(data)-i-1].replace("\n", '')+\
                            '#'+dataset+"\n"
                    break

            # # and write everything back
            with open(path, 'w', encoding="utf-8") as file:
                file.writelines( data )

        paper_title_taxonomy[paper_title]+=1


def taskAnnotation(paper_title, task):     
    path = f"{target_path}taskAnnotation.tsv"
    if paper_title not in paper_title_taskAnnotation: 
        with open(path, "a+", encoding="utf-8") as text_file:
            text_file.write(paper_title+"\t"+task)
            text_file.write("\n")
        paper_title_taskAnnotation[paper_title]+=1
        
def paper_links(paper_title, paper_url):      
    with open(f"{target_path}paper_links.tsv", "a+", encoding="utf-8") as text_file:
        text_file.write(paper_title+"\t"+paper_url)
        text_file.write("\n")


def parseTask(obj):
    """
    parses the Task json object and add the information into the graph
    
    Note: it might be called recursively
    
    Parameters
    ----------
    obj : dict
        the json representation of the Task object
    """

    datasets = obj["datasets"]

    task =  obj["task"].strip()
    
    if task.strip() == "":
        return 

    for dt in datasets:
        dataset = dt["dataset"].strip()
        
        if dataset == "":
            print("Empty dataset found")
            continue

        for _, row in enumerate(dt["sota"]["rows"]):
            
            paper_url = row['paper_url']

            if paper_url == "" or paper_url[-4:] == "html":
                continue

            if paper_url[-1] == '/':
                paper_url = paper_url[:-1]+'.pdf'

            elif paper_url[-3:] == "pdf":
                paper_url = paper_url
            else:
                paper_url = paper_url+'.pdf'

            paper_title = paper_url.split('/')[-1]

            # if (paper_title in paper_name_taxonomy) or (paper_url.split("//")[1][:5] != 'arxiv'):
            #     # TODO need to find a clever way to deal with the case of repeated paper
            #     # at different part of the json
            #     continue

            if (paper_url.split("//")[1][:5] != 'arxiv'):
                # TODO need to find a clever way to deal with paper not from arxiv
                continue

            # TODO I need to improve this for many metrics 
            for i, (metric, score) in  enumerate(row['metrics'].items()):
                # metric = dt["sota"]["metrics"][0]
                # rows = dt["sota"]["rows"]
                
                if not metric :
                    continue
                
                if not score :
                    score = ""
                    continue
                
                score = score.strip()
                metric = metric.strip()           
                
                # not all the metrics in sota are in metrics rows
                if f"{task}#{dataset}#{metric.strip()}#{score.strip()}" not in uniq_resultsAnnotation[paper_title]:
                    resultsAnnotation(paper_title, task, \
                                    dataset, metric.strip(), score)
                    uniq_resultsAnnotation[paper_title].append(f"{task}#{dataset}#{metric.strip()}#{score.strip()}")

            if f"{task}" not in uniq_taskAnnotation[paper_title]:
                taskAnnotation(paper_title, task)
                uniq_taskAnnotation[paper_title].append(f"{task}")

            if f"{dataset}" not in uniq_datasetAnnotation[paper_title]:
                datasetAnnotation(paper_title, dataset)
            
            if paper_title not in uniq_paper_link:
                paper_links(paper_title, paper_url)
                uniq_paper_link.add(paper_title)


    subtasks = obj["subtasks"]

    for subtask in subtasks:
        parseTask(subtask)

    
def createEvaluationSubgraph(obj):
    """
    Create the subgraph relating to the provided json strucutre,
    this subgraph corresponds to one of the paperswithcode data files
    (evaluation-tables.json').
    
    Parameters
    ----------
    obj : dict
        the json representation of the eu.tib.sre.evaluation of the paper
    """
    parseTask(obj)
    

paper_name_taxonomy = defaultdict(lambda : 0)
paper_title_taxonomy = defaultdict(lambda : 0)
paper_title_taskAnnotation = defaultdict(lambda : 0)

if __name__ == '__main__':
    """
    The current version of the code support only open access paper that are available on arxiv, 
    
    TODO
    -----
    - we can modify the code to support other open access source 
    """
    parser = argparse.ArgumentParser(description="Run TDM")

    parser.add_argument("-psource", "--path_source", 
        default="source_files/evaluation-tables.json", 
        help="Path to source evaluation-tables.json from PWC")
    parser.add_argument("-ptgt", "--path_target", 
        default="annotations_final/", 
        help="Path to the target folder to save the metadata")
    
    args = parser.parse_args()

    source_path = args.path_source
    target_path = args.path_target

    if not os.path.exists(target_path):
        os.makedirs(target_path)

    start_time = time.time()

    evalTables = readFile(source_path)

    filePath_result = f"{target_path}resultsAnnotation.tsv"
    filePath_dataset = f"{target_path}datasetAnnotation.tsv"
    filePath_task = f"{target_path}taskAnnotation.tsv"
    filePath_links = f"{target_path}paper_links.tsv"
    filePath_TDM = f"{target_path}TDM_taxonomy.tsv"
    filePath_TDMs = f"{target_path}TDMs_taxonomy.tsv"
    filePath_PaperName = f"{target_path}paper_name_taxonomy.tsv"


    for filePath in [filePath_result, filePath_dataset, \
                    filePath_task, filePath_links, filePath_TDM, filePath_TDMs, filePath_PaperName]:
        if os.path.exists(filePath):
            os.remove(filePath)
    
    for entry in evalTables:
        createEvaluationSubgraph(entry)
   #-------------------------------------------------

    # Saving the file from default dic to .tsv 
    parse_TDM_taxonomy(TDM_taxonomy)
    parse_TDMs_taxonomy(TDMs_taxonomy)
    parse_title_taxonomy(paper_name_taxonomy)

    # Clean the previously downloaded files for a minimum number of paper 



       