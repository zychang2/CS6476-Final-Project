# CS6476 team 42 Project: Using CNN-LSTM and Vision Transformers to perform video transition classification

## Description

The aim of this project is to investigate the performance different types of Deep Learning (DL) architectures on a video classification task. The classification task is inspired by [[1]], where the authors mentioned classifying transition of intentionality ("Oops!") from videos. The dataset is an excerpt of the original dataset posted by [[1]], and we perform all training tasks on this smaller dataset. The dataset is accessible [here] (Georgia Tech Sharepoint).

### Sample data

#### Example of a video without "Oops":

![Video 1](https://youtu.be/zkm6EhrSDso)

#### Selected frames from the video:



| 0s | 1s | 2s | 3s | 4s | 5s |
|---------|---------|---------|---------|---------|---------|
| ![Image 1](resources\25 Best Trampoline Fail Nominees - FailArmy Hall of Fame (July 2017)18_00.jpg) | ![Image 1](resources\25 Best Trampoline Fail Nominees - FailArmy Hall of Fame (July 2017)18_05.jpg) | ![Image 1](resources\25 Best Trampoline Fail Nominees - FailArmy Hall of Fame (July 2017)18_10.jpg) | ![Image 1](resources\25 Best Trampoline Fail Nominees - FailArmy Hall of Fame (July 2017)18_14.jpg) | ![Image 1](resources\25 Best Trampoline Fail Nominees - FailArmy Hall of Fame (July 2017)18_19.jpg) | ![Image 1](resources\25 Best Trampoline Fail Nominees - FailArmy Hall of Fame (July 2017)18_24.jpg) |


#### Example of a video with "Oops":

![Video 2](https://youtu.be/CBYLn15tSCA)

## Reference
[1]: https://arxiv.org/pdf/1911.11206.pdf
[\[1\] D. Epstein, B. Chen, and C. Vondrick, “Oops! Predicting Unintentional Action in Video,” Jun. 2020.](https://arxiv.org/pdf/1911.11206.pdf)

[here]: https://gtvault.sharepoint.com/:f:/s/CVTeam/EkSa6fRmLYFEtbkHedlofBgB1Pm76-SfRxSppaaKjWZCmw?e=sItsAG