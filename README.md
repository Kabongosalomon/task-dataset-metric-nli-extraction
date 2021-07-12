# task-dataset-metric-nli-extraction

This is the official github code for the paper `Automated Mining of  Leaderboards forEmpirical AI Research`.

This program assume the availability of training and testing data, and allows to do Sequence Classification for `task-dataset-metric` extraction using different transformer models. 

If you need to generate the train data yourself, you can clone `https://github.com/Kabongosalomon/task-dataset-metric-extraction` and follow the instructions. 

<!-- produces the test data for classification over a set of predefined task#dataset#metrics#software labels.
Given input a pdf file, it scrapes the text from the file using the Grobid parser, subsequently generating the test data file for input to the neural network classifier. -->

## Train

- Using singularity

    - `sbatch Bert_80_Neg_600_full_test_v2.sh python train_tdm.py -m Bert -bs 24 -ne 5 -ptrain /nfs/home/kabenamualus/Research/task-dataset-metric-extraction/data/new_pwc_imb_150/80Neg600unk/trainOutput.tsv -pvalid /nfs/home/kabenamualus/Research/task-dataset-metric-extraction/data/ibm/exp/few-shot-setup/NLP-TDMS/paperVersion/test_v2.tsv -o /nfs/home/kabenamualus/Research/task-dataset-metric-extraction/data/new_pwc_imb_150/80Neg600unk/`

    - `sbatch SciBert_80_Neg_600_full_test_v2.sh python train_tdm.py -m SciBert -bs 24 -ne 5 -ptrain /nfs/home/kabenamualus/Research/task-dataset-metric-extraction/data/new_pwc_ibm_150/600Neg600unk/twofoldwithunk/fold1/train.tsv -pvalid /nfs/home/kabenamualus/Research/task-dataset-metric-extraction/data/ibm/exp/few-shot-setup/NLP-TDMS/paperVersion/test.tsv -o /nfs/home/kabenamualus/Research/task-dataset-metric-extraction/data/new_pwc_ibm_150/600Neg600unk/twofoldwithunk/fold1/`

    - `sbatch XLNet_80_Neg_600_full_test_v2.sh python train_tdm.py -m XLNet -bs 16 -ne 5 -ptrain /nfs/home/kabenamualus/Research/task-dataset-metric-extraction/data/new_pwc_imb_150/80Neg600unk/trainOutput.tsv -pvalid /nfs/home/kabenamualus/Research/task-dataset-metric-extraction/data/ibm/exp/few-shot-setup/NLP-TDMS/paperVersion/test_v2.tsv -o /nfs/home/kabenamualus/Research/task-dataset-metric-extraction/data/new_pwc_imb_150/80Neg600unk/`

    - `sbatch tdm.sh python train_tdm.py -m SciBert -bs 32 -ne 5 -ptrain /nfs/home/kabenamualus/Research/task-dataset-metric-extraction/data/new_pwc_imb_150/80Neg600unk/trainOutput.tsv -pvalid /nfs/home/kabenamualus/Research/task-dataset-metric-extraction/data/ibm/exp/few-shot-setup/NLP-TDMS/paperVersion/test_v2.tsv -o /nfs/home/kabenamualus/Research/task-dataset-metric-extraction/data/new_pwc_imb_150/80Neg600unk/`

    - `sbatch XLNet_80_Neg_600_full_test_v2.sh python train_tdm.py -m XLNet -bs 16 -ne 5 -ptrain /nfs/home/kabenamualus/Research/task-dataset-metric-extraction/data/new_pwc_imb_150/80Neg600unk/trainOutput.tsv -pvalid /nfs/home/kabenamualus/Research/task-dataset-metric-extraction/data/ibm/exp/few-shot-setup/NLP-TDMS/paperVersion/test_v2.tsv -o /nfs/home/kabenamualus/Research/task-dataset-metric-extraction/data/new_pwc_imb_150/80Neg600unk/`

    - `sbatch SciBertIBM.sh python train_tdm.py -m Bert -bs 16 -ne 5 -ptrain /nfs/home/kabenamualus/Research/task-dataset-metric-extraction/data/ibm/exp/few-shot-setup/NLP-TDMS/paperVersion/train.tsv -pvalid /nfs/home/kabenamualus/Research/task-dataset-metric-extraction/data/ibm/exp/few-shot-setup/NLP-TDMS/paperVersion/test.tsv -o /nfs/home/kabenamualus/Research/task-dataset-metric-extraction/data/ibm/exp/few-shot-setup/NLP-TDMS/paperVersion/torch/Bert/`

    - `sbatch SciBertIBM.sh python train_tdm.py -m SciBert -bs 16 -ne 10 -ptrain /nfs/home/kabenamualus/Research/task-dataset-metric-extraction/data/ibm/exp/few-shot-setup/NLP-TDMS/paperVersion/train.tsv -pvalid /nfs/home/kabenamualus/Research/task-dataset-metric-extraction/data/ibm/exp/few-shot-setup/NLP-TDMS/paperVersion/test.tsv -o /nfs/home/kabenamualus/Research/task-dataset-metric-extraction/data/ibm/exp/few-shot-setup/NLP-TDMS/paperVersion/torch/SciBert_scheduler/`

    ### train from pretrained 
    - `sbatch tdm.sh python train_tdm.py -m SciBert -bs 24 -ne 3 -init_pt /nfs/home/kabenamualus/Research/task-dataset-metric-extraction/data/paperwithcode/new/60Neg800unk/twofoldwithunk/fold1/Model_SciBert_avg_metric_0.95.pt`



    - `sbatch tdm.sh python train_tdm.py -m BigBird -bs 12 -ne 3 -ptrain /nfs/home/kabenamualus/Research/task-dataset-metric-extraction/data/paperwithcode/newFull/60Neg800unk/twofoldwithunk/fold1/train.tsv -pvalid /nfs/home/kabenamualus/Research/task-dataset-metric-extraction/data/paperwithcode/newFull/60Neg800unk/twofoldwithunk/fold1/dev.tsv -o /nfs/home/kabenamualus/Research/task-dataset-metric-extraction/data/paperwithcode/newFull/60Neg800unk/twofoldwithunk/fold1/`

    - `sbatch tdm.sh python train_tdm.py -m XLNet -bs 24 -ne 5 -ptrain /nfs/home/kabenamualus/Research/task-dataset-metric-extraction/data/new_pwc_imb_150/80Neg600unk/twofoldwithunk/fold1/train.tsv -pvalid /nfs/home/kabenamualus/Research/task-dataset-metric-extraction/data/new_pwc_imb_150/80Neg600unk/twofoldwithunk/fold1/dev.tsv -o /nfs/home/kabenamualus/Research/task-dataset-metric-extraction/data/new_pwc_imb_150/80Neg600unk/twofoldwithunk/fold1/`

    - `sbatch tdm.sh python train_tdm.py -m XLNet -bs 24 -ne 5 -ptrain /nfs/home/kabenamualus/Research/task-dataset-metric-extraction/data/new_pwc_imb_450/80Neg600unk/twofoldwithunk/fold1/train.tsv -pvalid /nfs/home/kabenamualus/Research/task-dataset-metric-extraction/data/new_pwc_imb_450/80Neg600unk/twofoldwithunk/fold1/dev.tsv -o /nfs/home/kabenamualus/Research/task-dataset-metric-extraction/data/new_pwc_imb_450/80Neg600unk/twofoldwithunk/fold1/ -maxl 2000`
<!-- data/paperwithcode/newFull/60Neg800unk/twofoldwithunk/fold1/train.tsv -->
<!-- /nfs/home/kabenamualus/Research/task-dataset-metric-extraction/data/new_pwc_imb_150/80Neg600unk/twofoldwithunk/fold1/train.tsv -->
## Evaluation

- Using singularity
    - `sbatch tdm.sh python evaluate_tdm.py -m SciBert -bs 16 -pt /nfs/home/kabenamualus/Research/task-dataset-metric-extraction/data/paperwithcode/new/60Neg800unk/twofoldwithunk/fold1/Model_SciBert_avg_metric_0.95.pt -o /nfs/home/kabenamualus/Research/task-dataset-metric-extraction/data/paperwithcode/new/60Neg800unk/twofoldwithunk/fold1/`

    - `sbatch eval.sh python evaluate_tdm.py -m SciBert -bs 16 -pvalid /nfs/home/kabenamualus/Research/task-dataset-metric-extraction/data/ibm/exp/few-shot-setup/NLP-TDMS/paperVersion/test.tsv -pt /nfs/home/kabenamualus/Research/task-dataset-metric-extraction/data/ibm/exp/few-shot-setup/NLP-TDMS/paperVersion/torch/SciBert/Model_SciBert_avg_metric_0.8946.pt -o /nfs/home/kabenamualus/Research/task-dataset-metric-extraction/data/ibm/exp/few-shot-setup/NLP-TDMS/paperVersion/torch/SciBert/`

    - `sbatch eval.sh python evaluate_tdm.py -m SciBert -bs 32 -pvalid /nfs/home/kabenamualus/Research/task-dataset-metric-extraction/data/ibm/exp/few-shot-setup/NLP-TDMS/paperVersion/test.tsv -ptest_res /nfs/home/kabenamualus/Research/task-dataset-metric-extraction/data/ibm/exp/few-shot-setup/NLP-TDMS/paperVersion/torch/SciBert/test_results_SciBert.tsv -o /nfs/home/kabenamualus/Research/task-dataset-metric-extraction/data/ibm/exp/few-shot-setup/NLP-TDMS/paperVersion/torch/SciBert/`

    - `sbatch eval.sh python evaluate_tdm.py -m Original_IBM -pvalid /nfs/home/kabenamualus/Research/task-dataset-metric-extraction/data/ibm/exp/few-shot-setup/NLP-TDMS/paperVersion/test_v2.tsv -ptest_res /nfs/home/kabenamualus/Research/task-dataset-metric-extraction/data/ibm/exp/few-shot-setup/NLP-TDMS/paperVersion/test_results.tsv -o /nfs/home/kabenamualus/Research/task-dataset-metric-extraction/data/ibm/exp/few-shot-setup/NLP-TDMS/paperVersion/`
    
    - `sbatch eval.sh python evaluate_tdm.py -m SciBert_tf -pvalid /nfs/home/kabenamualus/Research/task-dataset-metric-extraction/data/ibm/exp/few-shot-setup/NLP-TDMS/paperVersion/test_v2.tsv -ptest_res /nfs/home/kabenamualus/Research/task-dataset-metric-extraction/data/paperwithcode/new/60Neg800unk/twofoldwithunk/fold1/models/SciBERT/test_results.tsv -o /nfs/home/kabenamualus/Research/task-dataset-metric-extraction/data/paperwithcode/new/60Neg800unk/twofoldwithunk/fold1/models/SciBERT/`

## Prediction

- Using singularity
    - `sbatch tdm.sh python predict_tdm.py -m SciBert -bs 32 -ptest /nfs/home/kabenamualus/Research/task-dataset-metric-nli-extraction/data/ibm/exp/few-shot-setup/NLP-TDMS/paperVersion/test.tsv -pt /nfs/home/kabenamualus/Research/task-dataset-metric-nli-extraction/data/ibm/exp/few-shot-setup/NLP-TDMS/paperVersion/torch/SciBert/Model_SciBert_Epoch_12_avg_metric_0.8212.pt -n 5 -o /nfs/home/kabenamualus/Research/task-dataset-metric-nli-extraction/data/ibm/exp/few-shot-setup/NLP-TDMS/paperVersion/torch/SciBert/Model_SciBert_Epoch_12_avg_metric_0.8212/`

    - `sbatch eval.sh python evaluate_tdm.py -m SciBert -bs 16 -pvalid /nfs/home/kabenamualus/Research/task-dataset-metric-nli-extraction/data/ibm/exp/few-shot-setup/NLP-TDMS/paperVersion/test.tsv -pt /nfs/home/kabenamualus/Research/task-dataset-metric-extraction/data/ibm/exp/few-shot-setup/NLP-TDMS/paperVersion/torch/SciBert/Model_SciBert_avg_metric_0.8946.pt -o /nfs/home/kabenamualus/Research/task-dataset-metric-nli-extraction/data/ibm/exp/few-shot-setup/NLP-TDMS/paperVersion/torch/SciBert/`

    - `sbatch tdm.sh python predict_tdm.py -m Bert -bs 32 -ptest /nfs/home/kabenamualus/Research/task-dataset-metric-extraction/data/ibm/exp/few-shot-setup/NLP-TDMS/paperVersion/test.tsv -pt /nfs/home/kabenamualus/Research/task-dataset-metric-extraction/data/ibm/exp/few-shot-setup/NLP-TDMS/paperVersion/torch/Bert/Model_Bert_avg_metric_0.9004.pt -n 5 -o /nfs/home/kabenamualus/Research/task-dataset-metric-extraction/data/ibm/exp/few-shot-setup/NLP-TDMS/paperVersion/torch/Bert/`
    
    /nfs/home/kabenamualus/Research/task-dataset-metric-extraction/data/paperwithcode/new/60Neg800unk/twofoldwithunk/fold1/Model_SciBert_avg_metric_0.92.pt
    data/paperwithcode/new/60Neg800unk/twofoldwithunk/fold1/Model_SciBert_avg_metric_0.66.pt

    - `sbatch tdm.sh python predict_tdm.py -m Bert -bs 16 -ptest /nfs/home/kabenamualus/Research/task-dataset-metric-extraction/data/ibm/exp/few-shot-setup/NLP-TDMS/paperVersion/test.tsv -pt /nfs/home/kabenamualus/Research/task-dataset-metric-extraction/data/new_pwc_ibm_150/80Neg600unk/Model_Bert_avg_metric_0.73.pt -n 3 -o /nfs/home/kabenamualus/Research/task-dataset-metric-extraction/data/new_pwc_ibm_150/80Neg600unk/`

    - `sbatch tdm.sh python predict_tdm.py -m SciBert -bs 6 -ptest /nfs/home/kabenamualus/Research/task-dataset-metric-extraction/data/ibm/exp/few-shot-setup/NLP-TDMS/paperVersion/test_v2.tsv -pt /nfs/home/kabenamualus/Research/task-dataset-metric-extraction/data/ibm/exp/few-shot-setup/NLP-TDMS/paperVersion/Model_SciBert_avg_metric_0.73.pt -n 3 -o /nfs/home/kabenamualus/Research/task-dataset-metric-extraction/data/ibm/exp/few-shot-setup/NLP-TDMS/paperVersion/`

    - `sbatch tdm.sh python predict_tdm.py -m BigBird -bs 16 -ptest /nfs/home/kabenamualus/Research/task-dataset-metric-extraction/data/ibm/exp/few-shot-setup/NLP-TDMS/paperVersion/test_v2.tsv -pt /nfs/home/kabenamualus/Research/task-dataset-metric-extraction/data/paperwithcode/newFull/60Neg800unk/twofoldwithunk/fold1/Model_BigBird_avg_metric_0.79.pt -n 6 -o /nfs/home/kabenamualus/Research/task-dataset-metric-extraction/data/ibm/exp/few-shot-setup/NLP-TDMS/paperVersion/`

    <!-- /nfs/home/kabenamualus/Research/task-dataset-metric-extraction/data/new_pwc_ibm_150/600Neg600unk -->

`sbatch SciBert_80_Neg_600_full_test_v2.sh python train_tdm.py -m SciBert -bs 24 -ne 5 -ptrain /nfs/home/kabenamualus/Research/task-dataset-metric-extraction/data/new_pwc_ibm_150/600Neg600unk/trainOutput.tsv -pvalid /nfs/home/kabenamualus/Research/task-dataset-metric-extraction/data/ibm/exp/few-shot-setup/NLP-TDMS/paperVersion/test_v2.tsv -o /nfs/home/kabenamualus/Research/task-dataset-metric-extraction/data/new_pwc_ibm_150/600Neg600unk/`


## Steps to run the program
 
The following procedure, suppose that you are using a linux OS, if otherwise kindly just clone the repo and rebuild the project. 

1. Clone this repository (https://github.com/Kabongosalomon/task-dataset-metric-extraction/tree/trainTest) or clone a particular branch `git clone -b trainTest https://github.com/Kabongosalomon/task-dataset-metric-extraction.git`.
2. move to the cloned directory `cd task-dataset-metric-extraction`
3. run the command `bash starter.sh`


## Acknowledgement: 
This program reuses code modules from IBM's science-result-extractor (https://github.com/IBM/science-result-extractor). A reference url to their paper on the ACL anthology is https://www.aclweb.org/anthology/P19-1513