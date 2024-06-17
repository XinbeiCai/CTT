# DIF-SR
The source code for our SIGIR 2022 Paper [**"Decoupled Side Information Fusion for Sequential Recommendation"**](https://arxiv.org/abs/2204.11046)

## Overview
We propose DIF-SR to effectively fuse side information for SR via
moving side information from input to the attention layer, motivated
by the observation that early integration of side information and
item id in the input stage limits the representation power of attention
matrices and flexibility of learning gradient. Specifically, we present
a novel decoupled side information fusion attention mechanism,
which allows higher rank attention matrices and adaptive gradient
and thus enhances the learning of item representation. Auxiliary
attribute predictors are also utilized upon the final representation
in a multi-task training scheme to promote the interaction of side
information and item representation.

## Preparation

Our code is based on PyTorch 1.8.1 and runnable for both windows and ubuntu server. Required python packages:

> + numpy==1.20.3
> + scipy==1.6.3
> + torch==1.8.1
> + tensorboard==2.7.0


## Usage

Download the dataset from

链接：https://pan.baidu.com/s/1FJsQaO6ITwmNmCf9Htdv3w 
提取码：6666 

And put the files in `./dataset/` like the following.

```
$ tree
.
├── Books
│   ├── Books.inter
│   └── Books.item
├── Movies_and_TV
│   ├── Movies_and_TV.inter
│   └── Movies_and_TV.item
└── CDs_and_Vinyl
    ├── CDs_and_Vinyl.inter
    └── CDs_and_Vinyl.item

```

Run `train/Books.sh`、`train/Movie.sh`、`train/CD.sh`. 


## Reproduction
See _benchmarks_ folder to reproduce the results.
For example, we show the detailed reproduce steps for the results of DIF-SR on the Amazon Beauty dataset in DIF_Amazon_Beauty.md file.

Due to some stochastic factors, slightly tuning the hyper-parameters using grid search is necessary if you want to reproduce the performance. If you have any question, please issue the project or email us and we will reply you soon.

## Cite

If you find this repo useful, please cite
```
@inproceedings{Xie2022DIF,
  author    = {Yueqi Xie and
               Peilin Zhou and
               Sunghun Kim},
  title     = {Decoupled Side Information Fusion for Sequential Recommendation},
  booktitle= {International ACM SIGIR Conference on Research and Development in Information Retrieval (SIGIR)},
  year      = {2022}
}
```

## Credit
This repo is based on [RecBole](https://github.com/RUCAIBox/RecBole).

## Contact
Feel free to contact us if there is any question. (YueqiXIE, yxieay@connect.ust.hk; Peilin Zhou, zhoupalin@gmail.com; Russell KIM, russellkim@upstage.ai)
