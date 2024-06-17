# MSSR
The source code for our WSDM '24 paper [**"Multi-Sequence Attentive User Representation Learning for Side-information Integrated Sequential Recommendation"**]



## Preparation
We train and evaluate our MSSR using a Tesla V100 PCIe GPU with 32 GB memory, where the CUDA version is 11.2. <br>
Our code is based on PyTorch, and requires the following python packages:

> + numpy==1.21.6
> + scipy==1.7.3 
> + torch==1.13.1+cu116
> + tensorboard==2.11.2

## Usage

Download the dataset from

link：https://pan.baidu.com/s/1FJsQaO6ITwmNmCf9Htdv3w 
code：6666 

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

Run `train/Books.sh`、`train/Movie.sh`、`train/CD.sh`. After training and evaluation, check out the final metrics in the "result.txt".

## Credit
This repo is based on [RecBole](https://github.com/RUCAIBox/RecBole).

