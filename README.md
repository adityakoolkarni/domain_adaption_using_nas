# AdaptSGL: Domain Adaptation using Neural Architecture Search and Small Group Learning

## Authors: Dongxia Wu, Aditya Kulkarni, Balaji Shankar Balachandran

## Description
This project aims to solve the problem of Domain Adaptation with minimum possible hand crafted architecture search. 

## Table of contents

- [Requirements](#requirements)
- [Running the Code](#run)
- [Dataset](#dataset)
- [Reference](#reference)

## Requirements <a name="requirements"></a>
Use the requirements.txt file to create a virtual environment. That is , conda create --name <env> --file requirements.txt.

## Running the Code  <a name="run"></a>
Run the following to do the experiment:
```
cd SGL-darts \\
python baseline_train_search_coop_pretrain_with_test.py \\
python pseudo_train_search_coop_pretrain_with_test.py \\
python mmd_train_search_coop_pretrain_with_test.py \\
python AdaptSGL.py \\
```

Update: GPU number if you have multiple GPU's 

## Dataset <a name="dataset"></a>
  [CIFAR-10](https://www.cs.toronto.edu/~kriz/cifar.html) for source domain, [STL-10](https://cs.stanford.edu/~acoates/stl10/) for target domain

## Reference <a name="reference"></a>
Code accompanying the paper [Small-Group Learning, with Application to Neural Architecture Search](https://arxiv.org/abs/2012.12502)  
This code is based on the implementation of [DARTS](https://github.com/quark0/darts).

