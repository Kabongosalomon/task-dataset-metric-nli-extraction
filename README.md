<!-- ## Steps to run the program
 
The following procedure, suppose that you are using a linux OS, if otherwise kindly just clone the repo and rebuild the project. 

1. Clone this repository (https://github.com/Kabongosalomon/task-dataset-metric-extraction/tree/trainTest) or clone a particular branch `git clone -b trainTest https://github.com/Kabongosalomon/task-dataset-metric-extraction.git`.
2. move to the cloned directory `cd task-dataset-metric-extraction`
3. run the command `bash starter.sh` -->
# Task Dataset Metric NLI Extraction

This repository is the official implementation of the paper [`Automated Mining of  Leaderboards forEmpirical AI Research`.]().

![pipeline](https://user-images.githubusercontent.com/13535078/81287158-33e01000-905a-11ea-8573-d716373efbdd.png)

If you need to generate the train data yourself, you can clone [`https://github.com/Kabongosalomon/task-dataset-metric-extraction`](https://github.com/Kabongosalomon/task-dataset-metric-extraction) and follow the instructions. 

## Requirements

Install requirements run:

```setup
pip install -r requirements.txt
```

## Datasets
We publish the following datasets:
* [2-fold TDM]()
<!-- * [Zero-shot 2-fold TDM]() -->

See [datasets](notebooks/datasets.ipynb) notebook for an example of how to load the datasets provided below. The [extraction](notebooks/extraction.ipynb) notebook shows how to use `axcell` to extract text and tables from papers.

## Training

Examples running `Bert`. `SciBert` and `XLNet`:
* `train_tdm.py -m Bert -bs 25 -ne 15 -ptrain path_to/task-dataset-metric-nli-extraction/data/pwc_ibm_150_5_10_10000/10Neg10000unk/twofoldwithunk/fold2/train.tsv -pvalid path_to/task-dataset-metric-nli-extraction/data/pwc_ibm_150_5_10_10000/10Neg10000unk/twofoldwithunk/fold2/dev.tsv -o path_to/task-dataset-metric-nli-extraction/data/pwc_ibm_150_5_10_10000/10Neg10000unk/twofoldwithunk/fold2/torch/Bert/`
* `train_tdm.py -m SciBert -bs 25 -ne 15 -ptrain path_to/task-dataset-metric-nli-extraction/data/pwc_ibm_150_5_10_10000/10Neg10000unk/twofoldwithunk/fold2/train.tsv -pvalid path_to/task-dataset-metric-nli-extraction/data/pwc_ibm_150_5_10_10000/10Neg10000unk/twofoldwithunk/fold2/dev.tsv -o path_to/task-dataset-metric-nli-extraction/data/pwc_ibm_150_5_10_10000/10Neg10000unk/twofoldwithunk/fold2/torch/SciBert/`
* `train_tdm.py -m XLNet -bs 25 -ne 15 -ptrain path_to/task-dataset-metric-nli-extraction/data/pwc_ibm_150_5_10_10000/10Neg10000unk/twofoldwithunk/fold2/train.tsv -pvalid path_to/task-dataset-metric-nli-extraction/data/pwc_ibm_150_5_10_10000/10Neg10000unk/twofoldwithunk/fold2/dev.tsv -o path_to/task-dataset-metric-nli-extraction/data/pwc_ibm_150_5_10_10000/10Neg10000unk/twofoldwithunk/fold2/torch/XLNet/`

Note: the following setup will also evaluate and write the `dev.tsv` predictions into a subdirecory of the output path. 

<!-- * [pre-training language model](notebooks/training/lm.ipynb) on the ArxivPapers dataset 
* [table type classifier](notebooks/training/table-type-classifier.ipynb) and [table segmentation](notebooks/training/table-segmentation.ipynb) on the SegmentedResults dataset  -->

## Evaluation

See the [evaluation](notebooks/evaluation.ipynb) notebook for the full example on how to evaluate AxCell on the PWCLeaderboards dataset. 

* `python evaluate_tdm.py -m SciBert -bs 16 -pt path_to/task-dataset-metric-extraction/data/paperwithcode/new/60Neg800unk/twofoldwithunk/fold1/Model_SciBert_avg_metric_0.95.pt -o path_to/task-dataset-metric-extraction/data/paperwithcode/new/60Neg800unk/twofoldwithunk/fold1/`

## Pre-trained Models

### Prediction

- `sbatch tdm.sh python predict_tdm.py -m SciBert -bs 32 -ptest path_to/task-dataset-metric-nli-extraction/data/ibm/exp/few-shot-setup/NLP-TDMS/paperVersion/test.tsv -pt path_to/task-dataset-metric-nli-extraction/data/ibm/exp/few-shot-setup/NLP-TDMS/paperVersion/torch/SciBert/Model_SciBert_Epoch_12_avg_metric_0.8212.pt -n 5 -o path_to/task-dataset-metric-nli-extraction/data/ibm/exp/few-shot-setup/NLP-TDMS/paperVersion/torch/SciBert/Model_SciBert_Epoch_12_avg_metric_0.8212/`

You can download pretrained models here:

- [axcell](https://github.com/paperswithcode/axcell/releases/download/v1.0/models.tar.xz) &mdash; an archive containing the taxonomy, abbreviations, table type classifier and table segmentation model. See the [results-extraction](notebooks/results-extraction.ipynb) notebook for an example of how to load and run the models 
- [language model](https://github.com/paperswithcode/axcell/releases/download/v1.0/lm.pth.xz) &mdash; [ULMFiT](https://arxiv.org/abs/1801.06146) language model pretrained on the ArxivPapers dataset

## Results

AxCell achieves the following performance:

<!-- ### 


| Dataset | Macro F1 | Micro F1 |
| ---------- |---------------- | -------------- |
| [PWC Leaderboards](https://paperswithcode.com/sota/scientific-results-extraction-on-pwc)     |     21.1         |      28.7       |
| [NLP-TDMS](https://paperswithcode.com/sota/scientific-results-extraction-on-nlp-tdms-exp)    |     19.7         |      25.8       |
| [ORKG-TDM](https://paperswithcode.com/sota/scientific-results-extraction-on-nlp-tdms-exp)    |     19.7         |      25.8       | -->



## License

[Apache 2.0 license](LICENSE).

## Citation
```bibtex
@inproceedings{axcell,
    title={Automated Mining of  Leaderboards forEmpirical AI Research},
    author={Salomon Kabongo, Jennifer D’Souza and Sören Auer},
    year={2021}
}
```