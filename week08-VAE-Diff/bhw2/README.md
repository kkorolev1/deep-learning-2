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

![cats](https://github.com/kkorolev1/dl-2/assets/72045472/c9031dbd-9e1e-4ec2-b859-09a8f4f052e7)

