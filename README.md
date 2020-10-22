# Anomaly Detection Framework for Industrial Vision

## Table of contents
* [Overview](#overview)
* [Dependencies](#dependencies)
* [Key Components of the Implementation](#Key-components)

## Overview 

This repository contains the code related to our anomaly detection framework that uses an autoencoder trained on images corrupted with our Stain-shaped noise. The full paper is available on [ArXiv](https://arxiv.org/abs/2008.12977) and will be soon presented at ICPR2020. 

Illustration belows presents an overview of anomaly dectection results obtained with our method (AESc + Stain) on some samples of the MVTec AD [dataset](https://www.mvtec.com/company/research/datasets/mvtec-ad/). 

<p align="center">
  <img width="300" src="https://github.com/anncollin/AnomalyDetection-Keras/blob/master/ReadmeImgs/resultOverview.png"> <br>
</p> 

## Dependencies

Our implementation is built on Keras (version 2.3.1) and tensorflow-gpu (version 1.15.0). <br>
The GUI in based on PySimpleGUI (version 4.16.0). 

More detailed informations are available in the `requirements.txt`.

## Key Components of the Implementation 

### Interresting files : Definition of the Stain noise corruption

We realize that our implementation is not the most intuitive due the interaction with the GUI to create the JSON files necessary to launch the training and evaluation procedures. However, our network architecture is a classical autoencoder and the training pipeline is standard. The novelty of the approach lies in the synthetic noise model used to corrupt our training images. This corruption is depicted below. 

<p align="center">
  <img width="500" src="https://github.com/anncollin/AnomalyDetection-Keras/blob/master/ReadmeImgs/StainNoiseModel.png"> <br>
</p> 

*The Stain noise model is a cubic interpolation between 20 points (orange dots), arranged in ascending order of polar coordinates, located around the border of an ellipse of variable size (blue line). The axes of the ellipse are comprised between 1 and 12% of the smallest image dimension and its eccentricity is randomly initialized.*

The definition of this Stain Corruption is contained in the `add_stain` function defined in the `datasets/add_corruption.py` file.

### The Graphical User Interface 

The experiments are launched through a GUI. This interface is composed of four tabs namely:
1. **Import_DB** Thanks to the "browse" buttons, select the folder path to the clean images and the abnormal test images. 
2. **Corrupt_DB** Define the specification of the synthetic noise model used to corrupt training images of a specific Database imported before. Default values are the one used in the paper. Also, a folder with test images corrupted with synthetic noise is also created at this time. 
3. **Train_Net** Define the network specifications and the training parameters in `hjson` files. 
4. **Eval_net** Run evaluation on a trained network. This can include performance evaluation with a residual- and/or an uncertainty-based approach.

<p float="left">
  <img src="https://github.com/anncollin/AnomalyDetection-Keras/blob/master/ReadmeImgs/Import_DB.png" width="600" />
  <img src="https://github.com/anncollin/AnomalyDetection-Keras/blob/master/ReadmeImgs/Corrupt_DB.png" width="600" />
  <img src="https://github.com/anncollin/AnomalyDetection-Keras/blob/master/ReadmeImgs/Train_net.png" width="600" />
  <img src="https://github.com/anncollin/AnomalyDetection-Keras/blob/master/ReadmeImgs/Eval_Net.png" width="600" />
</p>
