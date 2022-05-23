# Remote Sensing Image Change Captioning (RSICC)


To explore the Remote Sensing Image Change Captioning (RSICC) task, we build a large-scale dataset named LEVIR-CC, which contains 10077 pairs of bi-temporal RS images and 50385 sentences describing the differences between images. The novel dataset provides an opportunity to explore models that align visual changes and language. We believe the dataset will promote the research of RSICC. 

The images of the LEVIR-CC dataset are mainly from the change detection dataset LEVIR-CD, where each image has a spatial size of 1024×1024 pixels with a high resolution of 0.5 m/pixel. These bi-temporal images are from 20 different regions in Texas, USA. Since each image pair in the LEVIR-CD dataset contains very dense ground objects and changes, it is difficult to describe the changes accurately and adequately in a few sentences. Therefore, we crop the bi-temporal images to 256 × 256 pixels in our LEVIR-CC dataset. Besides, the bi-temporal RS images of our dataset are well registered in a pixel-by-pixel manner, so there is no view change.

Some examples of our dataset are as follows:
![Image text](Example/Example.png)
