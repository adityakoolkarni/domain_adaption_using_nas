# Domain Adaption Using Neural Architecture Search and Small Group Learning

## Authors: Dongxia Wu, Aditya Kulkarni, Balaji Shankar Balachandran

## Description
This project aims to solve the problem of Domain Adaptation with minimum possible hand crafted architecture search. 

## Table of contents

## Requirements <a name="requirements"></a>
Use the requirements.txt file to create a virtual environment. That is , conda create --name <env> --file requirements.txt.

## Running the Code
sh script.sh
Update: GPU number if you have multiple GPU's 

## Dataset <a name="dataset"></a>
  CIFAR-10 for source domain, STL-10 for target domain

Code accompanying the paper  
***Small Group Learning, with Application to Neural Architecture Search*** [paper]()  
<!-- Xiangning Chen, Ruochen Wang, Minhao Cheng, Xiaocheng Tang, Cho-Jui Hsieh -->

This code is based on the implementation of [P-DARTS](https://github.com/chenxin061/pdarts), [DARTS](https://github.com/quark0/darts) and [PC-DARTS](https://github.com/yuhuixu1993/PC-DARTS).

## Architecture Search

<!-- **Search on NAS-Bench-201 Space: (3 datasets to choose from)**

* Data preparation: Please first download the 201 benchmark file and prepare the api follow [this repository](https://github.com/D-X-Y/NAS-Bench-201).

* ```cd 201-space && python train_search_progressive.py``` -->

**Composing SGL with DARTS:**

```
CIFAR-10/100: cd SGL-darts && python train_search_coop_pretrain.py --weight_lambda 1 \\
--is_cifar100 0/1 --gpu 0 --batch_size 50 --save xxx
```



## Ablation Study (Search)

<!-- **Composing LCT with DARTS:**

```
CIFAR-10/100: cd darts-LCT && python train_search_ts_ab1/train_search_ts_ab4.py \\
--teacher_arch 18 \\
--weight_lambda 1 --weight_gamma 1 --unrolled\\
--is_cifar100 0/1 --gpu 0 --save xxx
```

