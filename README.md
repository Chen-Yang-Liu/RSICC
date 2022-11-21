# Remote Sensing Image Change Captioning With Dual-Branch Transformers: A New Method and a Large Scale Dataset
Here, we provide the pytorch implementation of the paper: "Remote Sensing Image Change Captioning With Dual-Branch Transformers: A New Method and a Large Scale Dataset". 

For more ore information, please see our published paper at https://ieeexplore.ieee.org/document/9934924

## LEVIR-CC Dataset
To explore the Remote Sensing Image Change Captioning (RSICC) task, we build a large-scale dataset named LEVIR-CC, which contains 10077 pairs of bi-temporal RS images and 50385 sentences describing the differences between images. The novel dataset provides an opportunity to explore models that align visual changes and language. We believe the dataset will promote the research of RSICC. 

Some examples of our dataset are as follows:
![Image text](Example/RSICCformer_structure.png)

## Installation
Clone this repo:
```python
git clone https://github.com/Chen-Yang-Liu/RSICC
cd RSICC
```

## Quick Start
You can run a demo to get started.
```python
python demo.py
```
## Prepare Dataset
### download the LEVIR-CC dataset
You could download the LEVIR-CD at

The path list in the downloaded folder is as follows:
```python
path to LEVIR-CD:
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

## Train
Prepare data for training:
```python
python create_input_files.py
```
start training:
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


