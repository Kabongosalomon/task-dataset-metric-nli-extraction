# task-dataset-metric-nli-extraction

This program produces the test data for classification over a set of predefined task#dataset#metrics#software labels.
Given input a pdf file, it scrapes the text from the file using the Grobid parser, subsequently generating the test data file for input to the neural network classifier.

## Train

- Using singularity
    - `sbatch tdm.sh python train_tdm.py -m XLNet -bs 16 -ne 2`
    - `sbatch tdm.sh python train_tdm.py -m BigBird -bs 12 -ne 3 -ptrain /nfs/home/kabenamualus/Research/task-dataset-metric-extraction/data/paperwithcode/newFull/60Neg800unk/twofoldwithunk/fold1/train.tsv -pvalid /nfs/home/kabenamualus/Research/task-dataset-metric-extraction/data/paperwithcode/newFull/60Neg800unk/twofoldwithunk/fold1/dev.tsv -o /nfs/home/kabenamualus/Research/task-dataset-metric-extraction/data/paperwithcode/newFull/60Neg800unk/twofoldwithunk/fold1/`
<!-- data/paperwithcode/newFull/60Neg800unk/twofoldwithunk/fold1/train.tsv -->
## Evaluation

- Using singularity
    - `sbatch tdm.sh python evaluate_tdm.py -m SciBert -bs 16 -pt /nfs/home/kabenamualus/Research/task-dataset-metric-extraction/data/paperwithcode/new/60Neg800unk/twofoldwithunk/fold1/Model_SciBert_avg_metric_0.95.pt -o /nfs/home/kabenamualus/Research/task-dataset-metric-extraction/data/paperwithcode/new/60Neg800unk/twofoldwithunk/fold1/`

## Prediction

- Using singularity
    - `sbatch tdm.sh python predict_tdm.py -m SciBert -bs 16 -ptest /nfs/home/kabenamualus/Research/task-dataset-metric-extraction/data/paperwithcode/new/60Neg800unk/twofoldwithunk/fold1/dev.tsv -pt /nfs/home/kabenamualus/Research/task-dataset-metric-extraction/data/paperwithcode/new/60Neg800unk/twofoldwithunk/fold1/Model_SciBert_avg_metric_0.95.pt -n 5 -o /nfs/home/kabenamualus/Research/task-dataset-metric-extraction/data/paperwithcode/new/60Neg800unk/twofoldwithunk/fold1/`

## Acknowledgement: 
This program reuses code modules from IBM's science-result-extractor (https://github.com/IBM/science-result-extractor). A reference url to their paper on the ACL anthology is https://www.aclweb.org/anthology/P19-1513