# Big homework 1. BLMs (Boutique LMs)

Language Model for writing stories trained on TinyStories dataset.

[Wandb report](https://wandb.ai/kkorolev/lm_project/reports/BHW1--Vmlldzo2MTU2MzY4) describes the pipeline in details.

[Model checkpoint](https://disk.yandex.ru/d/CoYoDi9t0atX3A)

## Installation guide

```shell
conda env create -f env.yaml
```

## Dataset preparation
Download TinyStories dataset from HF.
```shell
wget --quiet --show-progress "https://huggingface.co/datasets/roneneldan/TinyStories/resolve/main/TinyStories_all_data.tar.gz"
```
Then run prepara_data.py for creating a tokenized dataset. DATA_DIR is a directory of TinyStories JSON files.
```shell
python prepare_data.py -d DATA_DIR -t TXT_DIR [-v VOCAB_SIZE] -m TOKENIZER_PATH -o OUT_DIR
```
Afterwards a tokenized dataset will be saved in OUT_DIR.


Put paths to the dataset directory and the tokenizer .model file in a config. Configs for training and testing can be found in hw_lm/configs folder.

## Training
```shell
python train.py -c CONFIG -k WANDB_KEY
```
