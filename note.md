# Train
sbatch BertTrain.sh python train_tdm.py -m SciBert -bs 24 -ne 15 -ptrain /nfs/home/kabenamualus/Research/task-dataset-metric-nli-extraction/data/ibm/exp/few-shot-setup/NLP-TDMS/paperVersion/train.tsv -pvalid /nfs/home/kabenamualus/Research/task-dataset-metric-nli-extraction/data/ibm/exp/few-shot-setup/NLP-TDMS/paperVersion/test.tsv -o /nfs/home/kabenamualus/Research/task-dataset-metric-nli-extraction/data/ibm/exp/few-shot-setup/NLP-TDMS/paperVersion/torch/SciBert/

sbatch BertTrain.sh python train_tdm.py -m Bert -bs 24 -ne 15 -ptrain /nfs/home/kabenamualus/Research/task-dataset-metric-nli-extraction/data/ibm/exp/few-shot-setup/NLP-TDMS/paperVersion/train.tsv -pvalid /nfs/home/kabenamualus/Research/task-dataset-metric-nli-extraction/data/ibm/exp/few-shot-setup/NLP-TDMS/paperVersion/test.tsv -o /nfs/home/kabenamualus/Research/task-dataset-metric-nli-extraction/data/ibm/exp/few-shot-setup/NLP-TDMS/paperVersion/torch/Bert/

sbatch SciBertTrain.sh python train_tdm.py -m SciBert -bs 24 -ne 10 -ptrain /nfs/home/kabenamualus/Research/task-dataset-metric-nli-extraction/data/pwc_ibm_150_5_50_800/50Neg800unk/trainOutput.tsv -pvalid /nfs/home/kabenamualus/Research/task-dataset-metric-nli-extraction/data/ibm/exp/few-shot-setup/NLP-TDMS/paperVersion/test.tsv -o /nfs/home/kabenamualus/Research/task-dataset-metric-nli-extraction/data/pwc_ibm_150_5_50_800/50Neg800unk/torch/SciBert/

sbatch LargeXLNet.sh python train_tdm.py -m XLNet -bs 11 -ne 10 -ptrain /nfs/home/kabenamualus/Research/task-dataset-metric-nli-extraction/data/pwc_ibm_full_5_10_800/twofoldwithunk/fold1/train.tsv -pvalid /nfs/home/kabenamualus/Research/task-dataset-metric-nli-extraction/data/pwc_ibm_full_5_10_800/twofoldwithunk/fold1/dev.tsv -o /nfs/home/kabenamualus/Research/task-dataset-metric-nli-extraction/data/pwc_ibm_full_5_10_800/twofoldwithunk/fold1/torch/XLNet/ -maxl 3000

sbatch XLNet.sh python train_tdm.py -m XLNet -bs 3 -ne 10 -ptrain /nfs/home/kabenamualus/Research/task-dataset-metric-nli-extraction/data/pwc_ibm_full_5_50_800/50Neg800unk/twofoldwithunk/fold1/train.tsv -pvalid /nfs/home/kabenamualus/Research/task-dataset-metric-nli-extraction/data/pwc_ibm_full_5_50_800/50Neg800unk/twofoldwithunk/fold1/dev.tsv -o /nfs/home/kabenamualus/Research/task-dataset-metric-nli-extraction/data/pwc_ibm_full_5_50_800/50Neg800unk/twofoldwithunk/fold1/torch/XLNet/ -maxl 2700

sbatch EBigBird.sh python train_tdm.py -m BigBird -bs 2 -ne 10 -ptrain /nfs/home/kabenamualus/Research/task-dataset-metric-nli-extraction/data/pwc_ibm_full_5_10_800/twofoldwithunk/fold1/train.tsv -pvalid /nfs/home/kabenamualus/Research/task-dataset-metric-nli-extraction/data/pwc_ibm_full_5_10_800/twofoldwithunk/fold1/dev.tsv -o /nfs/home/kabenamualus/Research/task-dataset-metric-nli-extraction/data/pwc_ibm_full_5_10_800/twofoldwithunk/fold1/torch/EBigBird/ 

sbatch EBigBird.sh python train_tdm.py -m BigBird -bs 2 -ne 10 -ptrain /nfs/home/kabenamualus/Research/task-dataset-metric-nli-extraction/data/pwc_ibm_full_5_10_800/twofoldwithunk/fold2/train.tsv -pvalid /nfs/home/kabenamualus/Research/task-dataset-metric-nli-extraction/data/pwc_ibm_full_5_10_800/twofoldwithunk/fold2/dev.tsv -o /nfs/home/kabenamualus/Research/task-dataset-metric-nli-extraction/data/pwc_ibm_full_5_10_800/twofoldwithunk/fold2/torch/EBigBird/ 




sbatch XLNet.sh python train_tdm.py -m XLNet -bs 3 -ne 10 -ptrain /nfs/home/kabenamualus/Research/task-dataset-metric-nli-extraction/data/pwc_ibm_150_5_50_800/50Neg800unk/twofoldwithunk/fold1/train.tsv -pvalid /nfs/home/kabenamualus/Research/task-dataset-metric-nli-extraction/data/pwc_ibm_150_5_50_800/50Neg800unk/twofoldwithunk/fold1/dev.tsv -o /nfs/home/kabenamualus/Research/task-dataset-metric-nli-extraction/data/pwc_ibm_150_5_50_800/50Neg800unk/twofoldwithunk/fold1/torch/XLNet/

sbatch SciBertTrain.sh python train_tdm.py -m SciBert -bs 24 -ne 10 -ptrain /nfs/home/kabenamualus/Research/task-dataset-metric-nli-extraction/data/pwc_ibm_150_5_50_800/50Neg800unk/twofoldwithunk/fold1/train.tsv -pvalid /nfs/home/kabenamualus/Research/task-dataset-metric-nli-extraction/data/pwc_ibm_150_5_50_800/50Neg800unk/twofoldwithunk/fold1/dev.tsv -o /nfs/home/kabenamualus/Research/task-dataset-metric-nli-extraction/data/pwc_ibm_150_5_50_800/50Neg800unk/twofoldwithunk/fold1/torch/SciBert/

### Extended context 
sbatch EXLNet.sh python train_tdm.py -m XLNet -bs 6 -ne 10 -ptrain /nfs/home/kabenamualus/Research/task-dataset-metric-nli-extraction/data/ibm/exp/few-shot-setup/NLP-TDMS/paperVersion/train_full.tsv -pvalid /nfs/home/kabenamualus/Research/task-dataset-metric-nli-extraction/data/ibm/exp/few-shot-setup/NLP-TDMS/paperVersion/test_full.tsv -o /nfs/home/kabenamualus/Research/task-dataset-metric-nli-extraction/data/ibm/exp/few-shot-setup/NLP-TDMS/paperVersion/torch/EXLNet/ -maxl 1250

sbatch EBigBird.sh python train_tdm.py -m BigBird -bs 6 -ne 10 -ptrain /nfs/home/kabenamualus/Research/task-dataset-metric-nli-extraction/data/ibm/exp/few-shot-setup/NLP-TDMS/paperVersion/train_full.tsv -pvalid /nfs/home/kabenamualus/Research/task-dataset-metric-nli-extraction/data/ibm/exp/few-shot-setup/NLP-TDMS/paperVersion/test_full.tsv -o /nfs/home/kabenamualus/Research/task-dataset-metric-nli-extraction/data/ibm/exp/few-shot-setup/NLP-TDMS/paperVersion/torch/EBigBird/ 

sbatch ELongformer.sh python train_tdm.py -m Longformer -bs 16 -ne 15 -ptrain /nfs/home/kabenamualus/Research/task-dataset-metric-nli-extraction/data/ibm/exp/few-shot-setup/NLP-TDMS/paperVersion/train_full.tsv -pvalid /nfs/home/kabenamualus/Research/task-dataset-metric-nli-extraction/data/ibm/exp/few-shot-setup/NLP-TDMS/paperVersion/test_full.tsv -o /nfs/home/kabenamualus/Research/task-dataset-metric-nli-extraction/data/ibm/exp/few-shot-setup/NLP-TDMS/paperVersion/torch/ELongformer/ 


## Zero-shot 
sbatch SciBertTrain.sh python train_tdm.py -m SciBert -bs 24 -ne 15 -ptrain /nfs/home/kabenamualus/Research/task-dataset-metric-nli-extraction/data/ibm/exp/zero-shot-setup/NLP-TDMS/train.tsv -pvalid /nfs/home/kabenamualus/Research/task-dataset-metric-nli-extraction/data/ibm/exp/zero-shot-setup/NLP-TDMS/test.tsv -o /nfs/home/kabenamualus/Research/task-dataset-metric-nli-extraction/data/ibm/exp/zero-shot-setup/NLP-TDMS/torch/SciBert/

sbatch XLNet.sh python train_tdm.py -m XLNet -bs 24 -ne 15 -ptrain /nfs/home/kabenamualus/Research/task-dataset-metric-nli-extraction/data/ibm/exp/zero-shot-setup/NLP-TDMS/train.tsv -pvalid /nfs/home/kabenamualus/Research/task-dataset-metric-nli-extraction/data/ibm/exp/zero-shot-setup/NLP-TDMS/test.tsv -o /nfs/home/kabenamualus/Research/task-dataset-metric-nli-extraction/data/ibm/exp/zero-shot-setup/NLP-TDMS/torch/XLNet/

sbatch SciBertTrain.sh python predict_tdm.py -m SciBert -bs 24 -ptest /nfs/home/kabenamualus/Research/task-dataset-metric-nli-extraction/data/ibm/exp/zero-shot-setup/NLP-TDMS/test.tsv -pt /nfs/home/kabenamualus/Research/task-dataset-metric-nli-extraction/data/ibm/exp/zero-shot-setup/NLP-TDMS/torch/SciBert/Model_SciBert_Epoch_0_avg_metric_0.9512.pt -n 5 -o /nfs/home/kabenamualus/Research/task-dataset-metric-nli-extraction/data/ibm/exp/zero-shot-setup/NLP-TDMS/torch/SciBert/Epoch_0_avg_metric_0.9512.pt/

sbatch XLNet.sh python predict_tdm.py -m XLNet -bs 24 -ptest /nfs/home/kabenamualus/Research/task-dataset-metric-nli-extraction/data/ibm/exp/zero-shot-setup/NLP-TDMS/test.tsv -pt /nfs/home/kabenamualus/Research/task-dataset-metric-nli-extraction/data/ibm/exp/zero-shot-setup/NLP-TDMS/torch/XLNet/Model_XLNet_Epoch_11_avg_metric_0.9565.pt -n 5 -o /nfs/home/kabenamualus/Research/task-dataset-metric-nli-extraction/data/ibm/exp/zero-shot-setup/NLP-TDMS/torch/XLNet/torch/XLNet/Epoch_11_avg_metric_0.9565.pt/

sbatch XLNet.sh python predict_tdm.py -m XLNet -bs 24 -ptest /nfs/home/kabenamualus/Research/task-dataset-metric-nli-extraction/data/ibm/exp/zero-shot-setup/NLP-TDMS/test.tsv -pt /nfs/home/kabenamualus/Research/task-dataset-metric-nli-extraction/data/ibm/exp/zero-shot-setup/NLP-TDMS/torch/XLNet/Model_XLNet_Epoch_0_avg_metric_0.936.pt -n 5 -o /nfs/home/kabenamualus/Research/task-dataset-metric-nli-extraction/data/ibm/exp/zero-shot-setup/NLP-TDMS/torch/XLNet/torch/XLNet/Epoch_0_avg_metric_0.936.p/

# Predict
sbatch tdm.sh python predict_tdm.py -m SciBert -bs 32 -ptest /nfs/home/kabenamualus/Research/task-dataset-metric-nli-extraction/data/ibm/exp/zero-shot-setup/NLP-TDMS/test.tsv -pt /nfs/home/kabenamualus/Research/task-dataset-metric-nli-extraction/data/pwc_ibm_150_10_100_800/100Neg800unk/torch/SciBert/Model_SciBert_Epoch_0_avg_metric_0.875.pt -n 5 -o /nfs/home/kabenamualus/Research/task-dataset-metric-nli-extraction/data/pwc_ibm_150_10_100_800/100Neg800unk/torch/SciBert/Epoch_0_avg_metric_0.875.pt/

sbatch tdm.sh python predict_tdm.py -m Bert -bs 32 -ptest /nfs/home/kabenamualus/Research/task-dataset-metric-nli-extraction/data/ibm/exp/few-shot-setup/NLP-TDMS/paperVersion/test.tsv -pt /nfs/home/kabenamualus/Research/task-dataset-metric-nli-extraction/data/ibm/exp/few-shot-setup/NLP-TDMS/paperVersion/torch/Bert/Model_Bert_Epoch_5_avg_metric_0.9483.pt -n 5 -o /nfs/home/kabenamualus/Research/task-dataset-metric-nli-extraction/data/ibm/exp/few-shot-setup/NLP-TDMS/paperVersion/torch/Bert/Bert_Epoch_5_avg_metric_0.9483.pt/

sbatch XLNet.sh python predict_tdm.py -m XLNet -bs 24 -ptest /nfs/home/kabenamualus/Research/task-dataset-metric-nli-extraction/data/ibm/exp/few-shot-setup/NLP-TDMS/paperVersion/test.tsv -pt /nfs/home/kabenamualus/Research/task-dataset-metric-nli-extraction/data/ibm/exp/few-shot-setup/NLP-TDMS/paperVersion/torch/XLNet/Model_XLNet_Epoch_8_avg_metric_0.955.pt -n 5 -o /nfs/home/kabenamualus/Research/task-dataset-metric-nli-extraction/data/ibm/exp/few-shot-setup/NLP-TDMS/paperVersion/torch/XLNet/XLNet_Epoch_8_avg_metric_0.955.p/


sbatch tdm.sh python predict_tdm.py -m BigBird -bs 6 -ptest /nfs/home/kabenamualus/Research/task-dataset-metric-nli-extraction/data/ibm/exp/few-shot-setup/NLP-TDMS/paperVersion/test_full.tsv -pt /nfs/home/kabenamualus/Research/task-dataset-metric-nli-extraction/data/ibm/exp/few-shot-setup/NLP-TDMS/paperVersion/torch/EBigBird/Model_BigBird_Epoch_0_avg_metric_0.9604.pt -o /nfs/home/kabenamualus/Research/task-dataset-metric-nli-extraction/data/ibm/exp/few-shot-setup/NLP-TDMS/paperVersion/torch/EBigBird/Epoch_0_avg_metric_0.9604.pt/

=====

sbatch SciBertTest.sh python predict_tdm.py -m SciBert -bs 24 -ptest /nfs/home/kabenamualus/Research/task-dataset-metric-nli-extraction/data/ibm/exp/few-shot-setup/NLP-TDMS/paperVersion/test.tsv -pt /nfs/home/kabenamualus/Research/task-dataset-metric-nli-extraction/data/pwc_ibm_150_10_100_800/100Neg800unk/torch/SciBert/Model_SciBert_Epoch_12_avg_metric_0.9108.pt -o /nfs/home/kabenamualus/Research/task-dataset-metric-nli-extraction/data/pwc_ibm_150_10_100_800/100Neg800unk/torch/SciBert/Epoch_12_avg_metric_0.9108.pt/

sbatch XLNet.sh python predict_tdm.py -m XLNet -bs 24 -ptest /nfs/home/kabenamualus/Research/task-dataset-metric-nli-extraction/data/ibm/exp/few-shot-setup/NLP-TDMS/paperVersion/test.tsv -pt /nfs/home/kabenamualus/Research/task-dataset-metric-nli-extraction/data/pwc_ibm_150_10_80_800/80Neg800unk/torch/XLNet/Model_XLNet_Epoch_1_avg_metric_0.9156.pt -n 5 -o /nfs/home/kabenamualus/Research/task-dataset-metric-nli-extraction/data/pwc_ibm_150_10_80_800/80Neg800unk/torch/XLNet/Epoch_1_avg_metric_0.9156.pt/

# Valid 
sbatch eval.sh python evaluate_tdm.py -m SciBert -bs 16 -pvalid /nfs/home/kabenamualus/Research/task-dataset-metric-nli-extraction/data/ibm/exp/few-shot-setup/NLP-TDMS/paperVersion/test.tsv -pt /nfs/home/kabenamualus/Research/task-dataset-metric-extraction/data/ibm/exp/few-shot-setup/NLP-TDMS/paperVersion/torch/SciBert/Model_SciBert_avg_metric_0.8946.pt -o /nfs/home/kabenamualus/Research/task-dataset-metric-nli-extraction/data/ibm/exp/few-shot-setup/NLP-TDMS/paperVersion/torch/SciBert/


# Java
java -jar build/libs/task-dataset-metric-extraction-150.jar 'train' "/nfs/home/kabenamualus/Research/task-dataset-metric-extraction/data/pdf_IBM/" '5' '10' '600'  "/nfs/home/kabenamualus/Research/task-dataset-metric-nli-extraction/data/pwc_ibm_150_5_10_600/"

java -jar build/libs/task-dataset-metric-extraction-150.jar 'train' "/nfs/home/kabenamualus/Research/task-dataset-metric-extraction/data/pdf_IBM/" '5' '50' '600'  "/nfs/home/kabenamualus/Research/task-dataset-metric-nli-extraction/data/pwc_ibm_150_5_50_600/"

java -jar build/libs/task-dataset-metric-extraction-150.jar 'train' "/nfs/home/kabenamualus/Research/task-dataset-metric-extraction/data/pdf_IBM/" '5' '100' '600'  "/nfs/home/kabenamualus/Research/task-dataset-metric-nli-extraction/data/pwc_ibm_150_5_100_600/"

==============================================

java -jar build/libs/task-dataset-metric-extraction-full.jar 'train' "/nfs/home/kabenamualus/Research/task-dataset-metric-extraction/data/pdf_IBM/" '5' '100' '800'  "/nfs/home/kabenamualus/Research/task-dataset-metric-nli-extraction/data/pwc_ibm_full_5_100_800/"

java -jar build/libs/task-dataset-metric-extraction-150.jar 'train' "/nfs/home/kabenamualus/Research/task-dataset-metric-extraction/data/pdf_IBM/" '5' '100' '800'  "/nfs/home/kabenamualus/Research/task-dataset-metric-nli-extraction/data/pwc_ibm_150_5_100_800/"

java -jar build/libs/task-dataset-metric-extraction-full.jar 'train' "/nfs/home/kabenamualus/Research/task-dataset-metric-extraction/data/pdf_IBM/" '5' '50' '600'  "/nfs/home/kabenamualus/Research/task-dataset-metric-nli-extraction/data/pwc_ibm_full_5_50_600/"

java -jar build/libs/task-dataset-metric-extraction-full.jar 'train' "/nfs/home/kabenamualus/Research/task-dataset-metric-extraction/data/pdf_IBM/" '5' '100' '600'  "/nfs/home/kabenamualus/Research/task-dataset-metric-nli-extraction/data/pwc_ibm_full_5_100_600/"


============
java -jar build/libs/task-dataset-metric-extraction-full.jar 'train' "/nfs/home/kabenamualus/Research/task-dataset-metric-extraction/data/pdf_IBM/" '5' '100' '800'  "/nfs/home/kabenamualus/Research/task-dataset-metric-nli-extraction/data/pwc_ibm_full_5_100_800/"

java -jar build/libs/task-dataset-metric-extraction-150.jar 'train' "/nfs/home/kabenamualus/Research/task-dataset-metric-extraction/data/pdf_IBM/" '5' '100' '800'  "/nfs/home/kabenamualus/Research/task-dataset-metric-nli-extraction/data/pwc_ibm_short_150_5_100_800/"



### Transform IBM Data
java -jar build/libs/task-dataset-metric-extraction-full.jar 'train' "/nfs/home/kabenamualus/Research/task-dataset-metric-extraction/data/pdf_IBM_Original_Clean/" '5' '0' '1000'  "/nfs/home/kabenamualus/Research/task-dataset-metric-nli-extraction/data/ibm/exp/few-shot-setup/NLP-TDMS/paperVersion/IBM_jar_full/"

java -jar build/libs/task-dataset-metric-extraction-150_.jar 'train' "/nfs/home/kabenamualus/Research/task-dataset-metric-extraction/data/pdf_IBM_Original_Clean/" '5' '0' '1000'  "/nfs/home/kabenamualus/Research/task-dataset-metric-nli-extraction/data/ibm/exp/few-shot-setup/NLP-TDMS/paperVersion/IBM_jar_150/"

==============
Keep the same format of a folder in an other context, 
the file `tdm_style_one_to_two.py` allows to make sure that we have the same train, dev file but only with a different context. 

python tdm_style_one_to_two.py -psource /nfs/home/kabenamualus/Research/task-dataset-metric-nli-extraction/data/pwc_ibm_full_5_10_800/ -ptgtclone /nfs/home/kabenamualus/Research/task-dataset-metric-nli-extraction/data/pwc_ibm_short_5_10_800/

python tdm_style_one_to_two.py -psource /nfs/home/kabenamualus/Research/task-dataset-metric-nli-extraction/data/pwc_ibm_full_5_50_800/50Neg800unk/ -ptgtclone /nfs/home/kabenamualus/Research/task-dataset-metric-nli-extraction/data/pwc_ibm_150_5_50_800/50Neg800unk/

python tdm_style_one_to_two.py -psource /nfs/home/kabenamualus/Research/task-dataset-metric-nli-extraction/data/pwc_ibm_full_5_100_800/100Neg800unk/ -ptgtclone /nfs/home/kabenamualus/Research/task-dataset-metric-nli-extraction/data/pwc_ibm_150_5_100_800/100Neg800unk/