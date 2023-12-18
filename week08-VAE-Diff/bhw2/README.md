# BHW 2

## Installation guide

```shell
conda env create -f env.yaml
```

Configs for training and testing can be found in `hw_diff/configs` folder.

## Training
```shell
python train.py
```

## Testing
Calculates FID and SSIM metrics and generates images.
```shell
python test.py
```
