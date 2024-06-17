# MGCL

The source code for our Recsys 2023 Paper [**"A Multi-view Graph Contrastive Learning Framework for Cross-Domain Sequential Recommendation"**](https://dl.acm.org/doi/abs/10.1145/3604915.3608785).


## Environment

Our code is based on the following packages:

- Requirments： 
   - Python = 3.8.13
   - PyTorch 1.7.0
   - pandas 1.3.4
   - numpy 1.21.3


## Usage

1. Download the datasets and put the files in `cross_data/amazon/`.

2. Run the data preprocessing scripts to generate the data. 
``` 
cd cross_data
python process.py 
```
More details on data processing can be found in `cross_data/README.md`.
3. To run the program, try the script given in '/train/.
``` 
bash train/Book.sh
bash train/Movie.sh
bash train/CD.sh
```
More descriptions of the command arguments are as follws:  
```
arg_name            | type      | description  
--target_domain       str         Name of the target domain (e.g. Movies_and_TV).  
--source_domain       str         Name of the first source domain (e.g. Books).  
--num_epochs          int         Number of epochs.  
--batch_size          int         Batch size.  
--lr                  float       Learning rate.  
--device              str         Cpu or Cuda.  
--maxlen              int         Maximum length of sequences.  
--hidden_units        int         Latent vector dimensionality.  
--train_dir           str         Model to restore.  
--alpha               float       The weight of contrastive learning task.  
--beta                float       The weight of contrastive learning task. 
--gamma               float       The weight of contrastive learning task. 
--num_blocks          int         Number of attention blocks.  
--num_heads           int         Number of heads for attention.  
--dropout_rate        float       Dropout rate.  
--l2_emb              float       Regularization hyperparameter.  
```

## Cite

If you find this repo useful, please cite
```
@inproceedings{RecSys2023-MGCL,
  title={A Multi-view Graph Contrastive Learning Framework for Cross-Domain Sequential Recommendation},
  author={Zitao Xu and Weike Pan and Zhong Ming},
  booktitle={Proceedings of the 17th ACM Conference on Recommender Systems},
  pages = {},
  series = {RecSys '23},
  year={2023},
}
```