# Train
sbatch BertTrain.sh python train_tdm.py -m SciBert -bs 24 -ne 15 -ptrain /nfs/home/kabenamualus/Research/task-dataset-metric-nli-extraction/data/ibm/exp/few-shot-setup/NLP-TDMS/paperVersion/train.tsv -pvalid /nfs/home/kabenamualus/Research/task-dataset-metric-nli-extraction/data/ibm/exp/few-shot-setup/NLP-TDMS/paperVersion/test.tsv -o /nfs/home/kabenamualus/Research/task-dataset-metric-nli-extraction/data/ibm/exp/few-shot-setup/NLP-TDMS/paperVersion/torch/SciBert/

sbatch BertTrain.sh python train_tdm.py -m Bert -bs 24 -ne 15 -ptrain /nfs/home/kabenamualus/Research/task-dataset-metric-nli-extraction/data/ibm/exp/few-shot-setup/NLP-TDMS/paperVersion/train.tsv -pvalid /nfs/home/kabenamualus/Research/task-dataset-metric-nli-extraction/data/ibm/exp/few-shot-setup/NLP-TDMS/paperVersion/test.tsv -o /nfs/home/kabenamualus/Research/task-dataset-metric-nli-extraction/data/ibm/exp/few-shot-setup/NLP-TDMS/paperVersion/torch/Bert/

sbatch SciBertTrain.sh python train_tdm.py -m SciBert -bs 24 -ne 15 -ptrain /nfs/home/kabenamualus/Research/task-dataset-metric-nli-extraction/data/pwc_ibm_150_10_80_800/80Neg800unk/trainOutput.tsv -pvalid /nfs/home/kabenamualus/Research/task-dataset-metric-nli-extraction/data/ibm/exp/few-shot-setup/NLP-TDMS/paperVersion/test.tsv -o /nfs/home/kabenamualus/Research/task-dataset-metric-nli-extraction/data/pwc_ibm_150_10_80_800/80Neg800unk/torch/SciBert/

sbatch XLNet.sh python train_tdm.py -m XLNet -bs 24 -ne 15 -ptrain /nfs/home/kabenamualus/Research/task-dataset-metric-nli-extraction/data/pwc_ibm_150_10_80_800/80Neg800unk/trainOutput.tsv -pvalid /nfs/home/kabenamualus/Research/task-dataset-metric-nli-extraction/data/ibm/exp/few-shot-setup/NLP-TDMS/paperVersion/test.tsv -o /nfs/home/kabenamualus/Research/task-dataset-metric-nli-extraction/data/pwc_ibm_150_10_80_800/80Neg800unk//torch/XLNet/

# Predict
sbatch tdm.sh python predict_tdm.py -m SciBert -bs 32 -ptest /nfs/home/kabenamualus/Research/task-dataset-metric-nli-extraction/data/ibm/exp/zero-shot-setup/NLP-TDMS/test.tsv -pt /nfs/home/kabenamualus/Research/task-dataset-metric-nli-extraction/data/ibm/exp/few-shot-setup/NLP-TDMS/paperVersion/torch/SciBert/Model_SciBert_Epoch_1_avg_metric_0.9397.pt -n 5 -o /nfs/home/kabenamualus/Research/task-dataset-metric-nli-extraction/data/ibm/exp/zero-shot-setup/NLP-TDMS/torch/Epoch_1_avg_metric_0.9397.pt/

sbatch tdm.sh python predict_tdm.py -m Bert -bs 32 -ptest /nfs/home/kabenamualus/Research/task-dataset-metric-nli-extraction/data/ibm/exp/few-shot-setup/NLP-TDMS/paperVersion/test.tsv -pt /nfs/home/kabenamualus/Research/task-dataset-metric-nli-extraction/data/ibm/exp/few-shot-setup/NLP-TDMS/paperVersion/torch/Bert/Model_Bert_Epoch_5_avg_metric_0.9483.pt -n 5 -o /nfs/home/kabenamualus/Research/task-dataset-metric-nli-extraction/data/ibm/exp/few-shot-setup/NLP-TDMS/paperVersion/torch/Bert/Bert_Epoch_5_avg_metric_0.9483.pt/

sbatch XLNet.sh python predict_tdm.py -m XLNet -bs 24 -ptest /nfs/home/kabenamualus/Research/task-dataset-metric-nli-extraction/data/ibm/exp/few-shot-setup/NLP-TDMS/paperVersion/test.tsv -pt /nfs/home/kabenamualus/Research/task-dataset-metric-nli-extraction/data/ibm/exp/few-shot-setup/NLP-TDMS/paperVersion/torch/XLNet/Model_XLNet_Epoch_8_avg_metric_0.955.pt -n 5 -o /nfs/home/kabenamualus/Research/task-dataset-metric-nli-extraction/data/ibm/exp/few-shot-setup/NLP-TDMS/paperVersion/torch/XLNet/XLNet_Epoch_8_avg_metric_0.955.p/


# Valid 
sbatch eval.sh python evaluate_tdm.py -m SciBert -bs 16 -pvalid /nfs/home/kabenamualus/Research/task-dataset-metric-nli-extraction/data/ibm/exp/few-shot-setup/NLP-TDMS/paperVersion/test.tsv -pt /nfs/home/kabenamualus/Research/task-dataset-metric-extraction/data/ibm/exp/few-shot-setup/NLP-TDMS/paperVersion/torch/SciBert/Model_SciBert_avg_metric_0.8946.pt -o /nfs/home/kabenamualus/Research/task-dataset-metric-nli-extraction/data/ibm/exp/few-shot-setup/NLP-TDMS/paperVersion/torch/SciBert/


