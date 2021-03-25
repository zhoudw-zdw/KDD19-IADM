# Code of KDD2019 Paper Adaptive Deep Models for Incremental Learning: Considering Capacity Scalability and Sustainability

Official implementation for KDD 2019 paper [Adaptive Deep Models for Incremental Learning: Considering Capacity Scalability and Sustainability](https://dl.acm.org/doi/10.1145/3292500.3330865).

## Requirement
- python3.6.5
- Pytorch0.4.1
- numpy
- scipy
- sklearn

## Files 
    --Incremental MNIST: Incremental MNIST dataset,including a training file and 4 testing files, which named as ldx_t.mat, lux_t.mat, rdx_t.mat and rux_t.mat.
    --Draw.py: Draw the accuracy performance curve.
    --Dataset.py: Prepare Incremental MNIST dataset.
    --Test.py: Test performance of some model.
    --IADM.py: A demo of our method.

## Basic Usage  
There are various parameters in the input structure paras:

    --alp : Percentage of Fisher Information accumulated during Backpropagation.
    --lr : Learning rate of our model
    --drawstep: The frequence of Test duiring training (number of instance,default:12000)
    --lamda : The regularization parameter in our paper.(In Eq(5).)

## Quick Start
```
python IADM.py 
```
Parameters/options can be tuned to get better results.


## Citation 
Please cite our work if you feel the paper or the code are helpful.

```
@inproceedings{yang2019adaptive,
author = {Yang, Yang and Zhou, Da-Wei and Zhan, De-Chuan and Xiong, Hui and Jiang, Yuan},
title = {Adaptive Deep Models for Incremental Learning: Considering Capacity Scalability and Sustainability},
booktitle = {KDD},
address = {Anchorage, AK},
pages = {74--82},
year = {2019}
}
```

## Contact 
If there are any questions, please feel free to contact with the authors:  Da-Wei Zhou (zhoudw@lamda.nju.edu.cn) and Yang Yang (yangy@lamda.nju.edu.cn). Enjoy the code.
