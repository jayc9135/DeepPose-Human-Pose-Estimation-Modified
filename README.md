# DeepPose: Human Pose Estimation
# DeepPose: Human Pose Estimation
Project adopted from: 'https://github.com/HosseinPAI/DeepPose-Human-Pose-Estimation'

Date used from: 'https://github.com/axelcarlier/lsp'

We have carried out different experiments on the original model for our CSCI 635 Project.

This project is a PyTorch code based on [DeepPose: Human Pose Estimation via Deep Neural Networks](https://arxiv.org/abs/1312.4659). 
<p align="justify">
This model is consisted of different stages to estimate the coordination of each joint in the human body. It uses CNN layers to embed images and use regression to estimate points.
In this project, we only implement the initial stage, so the estimation accuracy is acceptable but different from the result on the original paper.
</p>
In the below picture, the structure of the model in the initial stage is shown.
<p align="center">
<img src="https://github.com/HosseinPAI/DeepPose-Human-Pose-Estimation/blob/master/.idea/pics/initial_stage.png" alt="Initial Stage" width="800" height='300'/>
</p>

<p align="justify">
We implement this model in two different modes on a simple <a href="https://sam.johnson.io/research/lsp_dataset.zip">LSP dataset</a>. One of them is simple training, and another one is the augmented mode. Some outputs predictions are shown in the below pictures. The right column is related to the true estimation, and the left is our prediction.
</p>
<p align="center">
<img src="https://github.com/HosseinPAI/DeepPose-Human-Pose-Estimation/blob/master/.idea/pics/output.png" alt="Initial Stage"/>
</p>

The necessary libraries have been written on the requirment.txt file. 
