

## This is the official repository of the paper:
#### ElasticFace: Elastic Margin Loss for Deep Face Recognition

![evaluation](https://raw.githubusercontent.com/fdbtrs/ElasticFace/main/images/margins.png)




| Model  | Log file| Pretrained model|
| ------------- | ------------- |------------- |
| ElasticFace-Arc      |[log file](https://drive.google.com/file/d/1sUiCqBaKsWYj4ul5VbDJnWjqoLJmX-Yv/view?usp=sharing) |[pretrained-mode](https://drive.google.com/drive/folders/1KLuqtHSIQNMk5S7aH2fpnNnsQyMjuE66?usp=sharing) |
| ElasticFace-Cos  |[log file](https://drive.google.com/file/d/16t8ea684TZIe1Z4__d09qr83RxZodh4k/view?usp=sharingg) |[pretrained-mode](https://drive.google.com/drive/folders/1TRmN9FXBPJlHWt5BxnhxP8DIOQ74VuF2?usp=sharing) |
| ElasticFace-Arc+  |[log file](https://drive.google.com/file/d/1s9pyve_bPxlywBQF1Ad7jBpElnrEocsi/view?usp=sharing) |[pretrained-mode](https://drive.google.com/drive/folders/1nKySFrEPR-fxZa8f3kkLryvDNucEUvgX?usp=sharing) |
| ElasticFace-Cos+  |[log file](https://drive.google.com/file/d/1Om2tqcd1DKHKl_4vYQ-1DxPlvW9_08cZ/view?usp=sharing) |[pretrained-mode](https://drive.google.com/drive/folders/1h9e8VVQAp9kpSK6k94zyL7uGTUqXGuHK?usp=sharing) |




### Face recognition  model training 
Model training:
In the paper, we employ MS1MV2 as the training dataset which can be downloaded from InsightFace (MS1M-ArcFace in DataZoo)

Download MS1MV2 dataset from [insightface](https://github.com/deepinsight/insightface/tree/master/recognition/_datasets_) on strictly follow the licence distribution

Unzip the dataset and place it in the data folder

Set the config.output and config.loss in the config/config.py 



All code has been trained and tested using  Pytorch 1.7.1

## Face recognition evaluation
##### Evaluation on LFW, AgeDb-30, CPLFW, CALFW and CFP-FP: 
1. download the data from their offical webpages.
2. alternative: The evaluation datasets are available in the training dataset package as bin file
3. set the config.rec to dataset folder e.g. data/faces_emore
4. set the config.val_targets for list of the evaluation dataset
5. download the pretrained model from link the previous table
6. set the config.output to path to pretrained model weights
7. run eval/evaluation.py
8. the output is test.log contains the evaluation results over all epochs

##### Evaluation on IJB-B and IJB-C: 

1. Please apply for permissions from NIST before your usage [NIST_Request](https://nigos.nist.gov/datasets/ijbc/request)
2. run eval/IJB/runIJBEval.sh
### To-do 
- [x] Add evaluation script 
