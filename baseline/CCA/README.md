# CCA

tensorflow implementation for CCA "Cascaded Cross Attention for Review-based Sequential Recommendation" 

## Packages 

- pandas==1.0.3
- tensorflow==1.14.0
- matplotlib==3.1.3
- numpy==1.18.1
- six==1.14.0
- scikit_learn==0.23.1

## Run

1) Create a folder named "review_datasets' and download the raw data from the [Amamzon dataset](https://cseweb.ucsd.edu/~jmcauley/datasets.html#amazon_reviews) into the subfloder
2) Run data_proccessing.py 
3) Run review_embedding.py
4) Run review_mean.py
5) Run CCA.py