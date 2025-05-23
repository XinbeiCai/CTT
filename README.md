# CTT

This repository contains the official implementation of our paper:
**"Contrastive Text-enhanced Transformer for Cross-Domain Sequential Recommendation"**

## 📋 Requirements

Experiments were conducted on a Tesla V100 PCIe GPU with 32GB of memory. The following Python packages are required:

```bash
Python==3.8.13  
torch==2.2.1  
pandas==2.0.3  
numpy==1.24.4  
transformers==4.38.2  
tqdm==4.67.1
```

You can install the dependencies using:

```bash
pip install -r requirements.txt
```


---

## 📁 Dataset Preparation

1. **Download Raw Review Data：**
   Download the raw datasets (`Books`, `Movies and TV`, `CDs_and_Vinyl`) from the [Amazon Review Data repository](http://jmcauley.ucsd.edu/data/amazon/).
   Place the downloaded files in the following directory:

   ```
   ./review_datasets/
   ```

2. **Download Pretrained BERT：**
   Download the `bert-base-cased` model files from Hugging Face:
   [https://huggingface.co/google-bert/bert-base-cased](https://huggingface.co/google-bert/bert-base-cased/tree/main)
   and place them in:

   ```
   ./bert-base-cased/
   ```
   ⭐ You may also use other text encoders as alternatives—please refer to the README in `data_processing/` for details.

3. **Generate Processed Data：**
   Run the following scripts to preprocess the data:

   ```bash
   python data_processing/data_processing.py
   python data_processing/review_embedding.py
   python data_processing/meta_embedding.py
   ```

4. *(Optional)* Download the preprocessed dataset directly:
   [Baidu Netdisk Link](https://pan.baidu.com/s/1hIDES34dw5G_6-L-oimzaA)
   Extraction code: `6666`
   Place the files in `CTT_proccessed_data` folder into:

   ```
   ./review_datasets/
   ```

---

## 🗂 Project Structure

```
CTT
│
├── README.md
├── baseline/                 # Baseline models 
├── bert-base-cased/          # Pretrained BERT model
├── data_processing/          # Data preprocessing scripts
├── layer.py                  # Transformer layers and modules
├── log/                      # Training logs and checkpoints
├── main.py                   # Entry point for training/evaluation
├── model.py                  # CTT model architecture
├── review_datasets/          # Raw and processed datasets
│   ├── Books/
│   ├── CDs_and_Vinyl/
│   └── Movies_and_TV/
├── train/                    # Training scripts
└── utils.py                  # Utility functions
```

---

## 🚀 Quick Start

To train the model on different domains, run the corresponding shell script located in `./train/`.

* **Books**

  ```bash
  bash train/Books.sh
  ```

* **Movies and TV**

  ```bash
  bash train/Movie.sh
  ```

* **CDs and Vinyl**

  ```bash
  bash train/CD.sh
  ```



