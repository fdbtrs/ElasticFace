

## This is the official repository of the paper:
#### ElasticFace: Elastic Margin Loss for Deep Face Recognition
Paper on arxiv: [arxiv](https://arxiv.org/pdf/2109.09416.pdf)
#### *** Accepted CVPR workshops 2022 ***
![evaluation](https://raw.githubusercontent.com/fdbtrs/ElasticFace/main/images/margins.png)




| Model  | Log file| Pretrained model| checkpoint |
| ------------- | ------------- |------------- | ------------- |
| ElasticFace-Arc      |[log file](https://drive.google.com/file/d/1jGm6rHh-jJ40c34u5eXBgAhR3u4KHblH/view?usp=sharing) |[pretrained-mode](https://drive.google.com/drive/folders/1q3ws_BQLmgXyiy2msvHummXq4pRqc1rx?usp=sharing) | 295672backbone.pth |
| ElasticFace-Cos  |[log file](https://drive.google.com/file/d/1XgfEQgEabinH--VhIusWQ8Js43vz1vK0/view?usp=sharing) |[pretrained-mode](https://drive.google.com/drive/folders/1ZiLLZXQ1jMzFwMGhYjtMwcdHmuedQb-2?usp=sharing) | 295672backbone.pth |
| ElasticFace-Arc+  |[log file](https://drive.google.com/file/d/1cWphaOqgCtmJ8zgVfnMXh0mVl6EqQZNd/view?usp=sharing) |[pretrained-mode](https://drive.google.com/drive/folders/1sf-fNV5CeSpWuFj6Hkwp7Js8SBXjbPo_?usp=sharing) | 295672backbone.pth |
| ElasticFace-Cos+  |[log file](https://drive.google.com/file/d/1aqCN5yfzgGijJLg2hcrsW3fvwHeHNu6W/view?usp=sharing) |[pretrained-mode](https://drive.google.com/drive/folders/19LXrjVNt60JBZP7JqsvOSWMwGLGrcJl5?usp=sharing) | 295672backbone.pth |

Evaluation result:
See: [Paper with code](https://paperswithcode.com/paper/elasticface-elastic-margin-loss-for-deep-face)



### Face recognition  model training 
Model training:
In the paper, we employ MS1MV2 as the training dataset which can be downloaded from InsightFace (MS1M-ArcFace in DataZoo)
Download MS1MV2 dataset from [insightface](https://github.com/deepinsight/insightface/tree/master/recognition/_datasets_) on strictly follow the licence distribution

Unzip the dataset and place it in the data folder
Set the config.output and config.loss in the config/config.py 



All code has been trained and tested using  Pytorch 1.7.1

## Face recognition evaluation
##### evaluation on LFW, AgeDb-30, CPLFW, CALFW and CFP-FP: 
1. download the data from their offical webpages.
2. alternative: The evaluation datasets are available in the training dataset package as bin file
3. set the config.rec to dataset folder e.g. data/faces_emore
4. set the config.val_targets for list of the evaluation dataset
5. download the pretrained model from link the previous table
6. set the config.output to path to pretrained model weights
7. run eval/evaluation.py
8. the output is test.log contains the evaluation results over all epochs

### To-do 
- [x] Add evaluation script 


If you use any of the code provided in this repository, please cite the following paper:
## Citation
```
@InProceedings{Boutros_2022_CVPR,
    author    = {Boutros, Fadi and Damer, Naser and Kirchbuchner, Florian and Kuijper, Arjan},
    title     = {ElasticFace: Elastic Margin Loss for Deep Face Recognition},
    booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR) Workshops},
    month     = {June},
    year      = {2022},
    pages     = {1578-1587}
}


```


## License

```
This project is licensed under the terms of the Attribution-NonCommercial-ShareAlike 4.0 
International (CC BY-NC-SA 4.0) license. 
Copyright (c) 2021 Fraunhofer Institute for Computer Graphics Research IGD Darmstadt
```
