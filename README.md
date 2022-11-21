# Remote Sensing Image Change Captioning With Dual-Branch Transformers: A New Method and a Large Scale Dataset
Here, we provide the pytorch implementation of the paper: "Remote Sensing Image Change Captioning With Dual-Branch Transformers: A New Method and a Large Scale Dataset". 
For more ore information, please see our published paper at https://ieeexplore.ieee.org/document/9934924
We provide the pytorch implementation of RSICCformer for remote sensing image change captioning.

## LEVIR-CC Dataset
To explore the Remote Sensing Image Change Captioning (RSICC) task, we build a large-scale dataset named LEVIR-CC, which contains 10077 pairs of bi-temporal RS images and 50385 sentences describing the differences between images. The novel dataset provides an opportunity to explore models that align visual changes and language. We believe the dataset will promote the research of RSICC. 

Some examples of our dataset are as follows:
![Image text](Example/Example.png)

## Quick Start
You can run a demo to get started.

## Train
```python
python create_input_files.py
python train.py
```
## Test
```python
python eval.py
```

## Please cite: 
```
@ARTICLE{9709791,
  author={Liu, Chenyang and Zhao, Rui and Shi, Zhenwei},
  journal={IEEE Geoscience and Remote Sensing Letters}, 
  title={Remote-Sensing Image Captioning Based on Multilayer Aggregated Transformer}, 
  year={2022},
  volume={19},
  number={},
  pages={1-5},
  doi={10.1109/LGRS.2022.3150957}}
```
## Reference:
https://github.com/sgrvinod/a-PyTorch-Tutorial-to-Image-Captioning.git


