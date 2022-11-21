# Remote Sensing Image Change Captioning With Dual-Branch Transformers: A New Method and a Large Scale Dataset
![visitors](https://visitor-badge.glitch.me/badge?page_id=Chen-Yang-Liu.RSICC.visitor)
![GitHub stars](https://badgen.net/github/stars/Chen-Yang-Liu/RSICC)
[![](https://img.shields.io/badge/code--license-MIT-brightgreen)](#License)

## Share us a :star: if this repo does help

Here, we provide the pytorch implementation of the paper: "Remote Sensing Image Change Captioning With Dual-Branch Transformers: A New Method and a Large Scale Dataset". 

For more ore information, please see our published paper in [[IEEE](https://ieeexplore.ieee.org/document/9934924) | [Lab Server](http://levir.buaa.edu.cn/publications/ChangeCaptioning.pdf)]  ***(Accepted by TGRS 2022)***

![RSICCformer_structure](Example/RSICCformer_structure.png)


## LEVIR-CC Dataset Download Source
- All of the Dataset: [[Google Drive](https://drive.google.com/file/d/1YppN3c2WQtuwI0cwQL2otLgW7ewFW_Ut/view?usp=sharing)  &#124; [Baidu Pan](https://pan.baidu.com/s/1YrWcz090kdqOZ0lrbqXJJA) (code:nq9y)]


The path list in the downloaded folder is as follows:
```python
path to LEVIR_CC_dataset:
                ├─LevirCCcaptions.json
                ├─images
                  ├─train
                  │  ├─A
                  │  ├─B
                  ├─val
                  │  ├─A
                  │  ├─B
                  ├─test
                  │  ├─A
                  │  ├─B
```
where A contains images of pre-phase, B contains images of post-phase.

The LEVIR-CC dataset contains 10077 pairs of bi-temporal remote sensing images and 50385 sentences describing the differences between images.
Some examples of our dataset are as follows:
![dataset_example](Example/dataset_example.png)

## Installation
Clone this repo:
```python
git clone https://github.com/Chen-Yang-Liu/RSICC
cd RSICC
```

## Quick Start
Firstly, you can download our RSICCformer pretrained model——by [[Google Drive](https://drive.google.com/file/d/1YppN3c2WQtuwI0cwQL2otLgW7ewFW_Ut/view?usp=sharing)  &#124; [Baidu Pan](https://pan.baidu.com/s/1YrWcz090kdqOZ0lrbqXJJA) (code:nq9y)]
After downloaded the pretrained model, you can put it in ./models_checkpoint/.

Then, run a demo to get started as follows:
```python
python demo.py
```

## Train
Prepare data for training:
```python
python create_input_files.py
```
Then, start training:
```python
python train.py
```
## Test and Compute captioning metrics
```python
python eval.py
```

## Please cite: 
```
@ARTICLE{9934924,
  author={Liu, Chenyang and Zhao, Rui and Chen, Hao and Zou, Zhengxia and Shi, Zhenwei},
  journal={IEEE Transactions on Geoscience and Remote Sensing}, 
  title={Remote Sensing Image Change Captioning With Dual-Branch Transformers: A New Method and a Large Scale Dataset}, 
  year={2022},
  volume={60},
  number={},
  pages={1-20},
  doi={10.1109/TGRS.2022.3218921}}
```
## Reference:
https://github.com/sgrvinod/a-PyTorch-Tutorial-to-Image-Captioning.git


