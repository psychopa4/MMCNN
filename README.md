# Multi-Memory Convolutional Neural Network for Video Super-Resolution
This work is based on [Tao et al](https://github.com/jiangsutx/SPMC_VideoSR)

## Datasets
We have collected 522 video sequences for training and 20 sequences for evaluation (mainly from documentaries), and in consider of copyright, the datasets should only be used for study.

The datasets can be downloaded from Google Drive, [train](https://drive.google.com/open?id=1xPMYiA0JwtUe9GKiUa4m31XvDPnX7Juu) and [evaluation](https://drive.google.com/open?id=1SgP9lZVpi-xvNeBxcze5FrjRLXWAE5ar).

For reserchers who cannot get access to Google, you may visit Baiduyun, [train](https://pan.baidu.com/s/1MjysWqjiJ5RcaoGn67YpUg) and [evaluation](https://pan.baidu.com/s/1ZgcZA_ExUfSmaB5QwIzqQg).

Unzip the training dataset to ./data/train/ and evaluation dataset to ./data/eval/ .

We only provide the ground truth images and the corresponding 4x downsampled LR images by Bicubic, and you may use `PIL` or `Matlab` to generate 2x or 3x downsampled LR images.

## Environment
  - Python (Tested on 3.6)
  - Tensorflow >= 1.3.0

## Training
 - python main.py to train model MMCNN-M10.

## Testing
It should be easy to use 'testvideo()' or 'testvideos()' functions.

## Citation
If you find our code or datasets helpful, please consider citing our work.

## Contact
If you have questions or suggestions, please send email to yipeng@whu.edu.cn.

## Visual Results
We show the visual results under 4x upscaling.
This frame is from [Videoset4 dataset](http://openaccess.thecvf.com/content_cvpr_2017/papers/Caballero_Real-Time_Video_Super-Resolution_CVPR_2017_paper.pdf).

![Image text](https://github.com/psychopa4/MMCNN/blob/master/pictures/000.jpg)

This frame is from [Myanmar test dataset](https://ieeexplore.ieee.org/document/7444187).

![Image text](https://github.com/psychopa4/MMCNN/blob/master/pictures/001.jpg)