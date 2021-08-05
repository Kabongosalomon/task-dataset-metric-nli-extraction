<!-- ## Steps to run the program
 
The following procedure, suppose that you are using a linux OS, if otherwise kindly just clone the repo and rebuild the project. 

1. Clone this repository (https://github.com/Kabongosalomon/task-dataset-metric-extraction/tree/trainTest) or clone a particular branch `git clone -b trainTest https://github.com/Kabongosalomon/task-dataset-metric-extraction.git`.
2. move to the cloned directory `cd task-dataset-metric-extraction`
3. run the command `bash starter.sh` -->
# Task Dataset Metric NLI Extraction

This repository is the official implementation of the paper [`Automated Mining of  Leaderboards forEmpirical AI Research`.]().

![pipeline](https://raw.githubusercontent.com/Kabongosalomon/task-dataset-metric-nli-extraction/main/images/overview.png)

If you need to generate the train and test data yourself, you can clone [`https://github.com/Kabongosalomon/task-dataset-metric-extraction`](https://github.com/Kabongosalomon/task-dataset-metric-extraction) and follow the instructions. 

## Requirements

Install requirements run:

```setup
pip install -r requirements-cpu.txt
```

or if yuou have cuda installed in your OS.

```setup
pip install -r requirements.txt
```

## Datasets
We publish the following datasets:
* [ORKG-TDM](http://doi.org/10.5281/zenodo.5105798)

* Run `bash data.sh` to download and copy the data in the appopriate location.     


Note: The metrics evaluation used in the paper uses a modified version of the original [IBM Science-result-extractor](https://github.com/IBM/science-result-extractor) java code base that can be accessed [here](https://github.com/Kabongosalomon/task-dataset-metric-extraction/blob/master/src/main/java/eu/tib/sre/tdmsie/TEModelEvalOnNLPTDMS.java)

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