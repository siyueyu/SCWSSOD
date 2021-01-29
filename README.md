# SCWSSOD
This is the implementation of `Structure-Consistent Weakly Supervised Salient Object Detection with Local Saliency Coherence (AAAI2021)`.
# Training
## Requirements
1. Clone this project and install required pytorch first.
2. pip install -r requirements.txt
## Training data
The training data can be downloaed from [Scribble_Saliency](https://github.com/JingZhang617/Scribble_Saliency).
## Pretrained weights for backbone
The pretrianed weight for backbone can be downloaded from [Res50](https://drive.google.com/file/d/1arzcXccUPW1QpvBrAaaBv1CapviBQAJL/view?usp=sharing).
## Traing procedure
1. Download training data and put them into 'data' folder.
2. run train.py
# Testing
## Test Model
The test model can be downloaded from [model](https://drive.google.com/file/d/1X8Y7NcnzRY8we2tgDS6KRVOde5ij7yWE/view?usp=sharing).
## Testing procedure
1. Modify test path
2. run test.py
## Predicted Maps
The predicted Saliency Maps can be downloaded from [prediction](https://drive.google.com/file/d/1a_Hrl0YhMNdNsskKLrZ7JJhZwJtxwyqs/view?usp=sharing).
## Evaluation
The evaluation code is from [GCPANet](https://github.com/JosephChenHub/GCPANet).
# Others
The code is based on [GCPANet](https://github.com/JosephChenHub/GCPANet) and [GatedCRFLoss](https://github.com/LEONOB2014/GatedCRFLoss).
