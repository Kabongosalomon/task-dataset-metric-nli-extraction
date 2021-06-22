# Train
sbatch BertTrain.sh python train_tdm.py -m Bert -bs 24 -ne 10 -ptrain /nfs/home/kabenamualus/Research/task-dataset-metric-nli-extraction/data/ibm/exp/few-shot-setup/NLP-TDMS/paperVersion/train.tsv -pvalid /nfs/home/kabenamualus/Research/task-dataset-metric-nli-extraction/data/ibm/exp/few-shot-setup/NLP-TDMS/paperVersion/test.tsv -o /nfs/home/kabenamualus/Research/task-dataset-metric-nli-extraction/data/ibm/exp/few-shot-setup/NLP-TDMS/paperVersion/torch/Bert/

# Predict
sbatch tdm.sh python predict_tdm.py -m Bert -bs 32 -ptest /nfs/home/kabenamualus/Research/task-dataset-metric-nli-extraction/data/ibm/exp/few-shot-setup/NLP-TDMS/paperVersion/test.tsv -pt /nfs/home/kabenamualus/Research/task-dataset-metric-nli-extraction/data/ibm/exp/few-shot-setup/NLP-TDMS/paperVersion/torch/Bert/Model_Bert_Epoch_0_avg_metric_0.9033.pt -n 5 -o /nfs/home/kabenamualus/Research/task-dataset-metric-nli-extraction/data/ibm/exp/few-shot-setup/NLP-TDMS/paperVersion/torch/Bert/Bert_Epoch_0_avg_metric_0.9033/

# Valid 
sbatch eval.sh python evaluate_tdm.py -m SciBert -bs 16 -pvalid /nfs/home/kabenamualus/Research/task-dataset-metric-nli-extraction/data/ibm/exp/few-shot-setup/NLP-TDMS/paperVersion/test.tsv -pt /nfs/home/kabenamualus/Research/task-dataset-metric-extraction/data/ibm/exp/few-shot-setup/NLP-TDMS/paperVersion/torch/SciBert/Model_SciBert_avg_metric_0.8946.pt -o /nfs/home/kabenamualus/Research/task-dataset-metric-nli-extraction/data/ibm/exp/few-shot-setup/NLP-TDMS/paperVersion/torch/SciBert/


