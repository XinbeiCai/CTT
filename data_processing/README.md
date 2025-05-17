

# Text Encoder Integration Guide

This project supports multiple text encoders for processing textual data. You can easily switch between different encoders by using the corresponding embedding scripts and updating the file paths in `CTT/utils.py`.

## Supported Text Encoders

### 1. BERT (Default: `bert-base-cased`)

Download the model from Hugging Face:
🔗 [https://huggingface.co/google-bert/bert-base-cased](https://huggingface.co/google-bert/bert-base-cased)

**Scripts:**

* `review_embedding.py`
* `meta_embedding.py`

### 2. E5 (`multilingual-e5-large`)

Download the model from Hugging Face:
🔗 [https://huggingface.co/intfloat/multilingual-e5-base](https://huggingface.co/intfloat/multilingual-e5-base)

**Scripts:**

* `review_embedding_e5.py`
* `meta_emb_e5.py`

### 3. T5 (`sentence-t5-base`)

Download the model from Hugging Face:
🔗 [https://huggingface.co/sentence-transformers/sentence-t5-base](https://huggingface.co/sentence-transformers/sentence-t5-base)

**Scripts:**

* `review_emb_t5.py`
* `meta_emb_sentencet5.py`

### 4. Modern BERT (`modern bert`)

Download the model from Hugging Face:
🔗 [https://huggingface.co/answerdotai/ModernBERT-base](https://huggingface.co/answerdotai/ModernBERT-base)

**Scripts:**

* `review_embedding_e5.py`
* `meta_emb_modernbert.py`

## How to Switch Encoders

To switch between encoders, simply replace the model paths in:

```
CTT/utils.py
```

Make sure the embedding scripts you choose are consistent with the encoder model you've downloaded.

---
