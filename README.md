# Anomaly Detection Framework for Industrial Vision

## Table of contents
- [Context](#Context)
- [Overview of the Method](#Overview-of-the-Method)
- [Dependencies](#Dependencies)
- [Key Components of the Implementation](#Key-Components-of-the-Implementation)
  - [Definition of the Stain noise corruption](#Definition-of-the-Stain-noise-corruption)
  - [The Graphical User Interface](#The-Graphical-User-Interface)
  
## Context 

This repository contains the code related to our anomaly detection framework that uses an autoencoder trained on images corrupted with our Stain-shaped noise. The full paper is available on [ArXiv](https://arxiv.org/abs/2008.12977) and will be soon presented at ICPR2020. 

Illustration belows presents an overview of anomaly dectection results obtained with our method (AESc + Stain) on some samples of the MVTec AD [dataset](https://www.mvtec.com/company/research/datasets/mvtec-ad/). 

<p align="center">
  <img width="300" src="https://github.com/anncollin/AnomalyDetection-Keras/blob/master/ReadmeImgs/resultOverview.png"> <br>
</p> 

## Overview of the Method

In this work, we adress the problem of anomaly detection in images for an industrial application. Our method is based on an autoencoder trained to map an
arbitrary image, i.e. with or without any defect, to a clean image, i.e. without any defect. In this approach, the defects can be dected through: 
- (commonly) a **residual-based approach** that evaluates the abnormality by measuring the absolute difference between the input image and its reconstructed clean version.
- (alternatively) an **uncertainty-based approach** relies on the intuition that structures that are not seen during training, i.e. the anomalies, will correlate with higher uncertainties, as estimated by the variance between 30 output images inferred with the MCDropout technique.

<p align="center">
  <img width="500" src="https://github.com/anncollin/AnomalyDetection-Keras/blob/master/ReadmeImgs/Method.png"> <br>
</p> 

To improve the sharpness of the recontructed clean image, we consider an autoencoder architecture with skip connections. In the common scenario where only clean images are available for training, we propose to corrupt them with a synthetic noise model to prevent the convergence of the network towards the identity mapping, and introduce an original Stain noise model for that purpose. We show that this model favors the reconstruction of clean images from arbitrary real-world images, regardless of the actual defects appearance.

## Dependencies

Our implementation is built on tensorflow (version 2.4.1). <br>
The GUI in based on PySimpleGUI (version 4.38.0). 

More detailed informations are available in the `requirements.txt`.

## Key Components of the Implementation 

### Definition of the Stain noise corruption

Our implementation relies on a GUI to create the JSON files necessary to launch the training and evaluation procedures. Handling this tool can look rather laborious. However, our network architecture is a classical autoencoder and the training pipeline is standard. For the readers that are interrested of identifying wether the method can improve theirs, they can focus on the understanding of the corruption model. The novelty of the approach lies in this synthetic noise model used to corrupt our training images. This corruption is depicted below. 

<p align="center">
  <img width="500" src="https://github.com/anncollin/AnomalyDetection-Keras/blob/master/ReadmeImgs/StainNoiseModel.png"> <br>
</p> 

*The Stain noise model is a cubic interpolation between 20 points (orange dots), arranged in ascending order of polar coordinates, located around the border of an ellipse of variable size (blue line). The axes of the ellipse are comprised between 1 and 12% of the smallest image dimension and its eccentricity is randomly initialized.*

The definition of this Stain Corruption is contained in the `add_stain` function defined in the `datasets/add_corruption.py` file.

### The Graphical User Interface 

The experiments are launched through a GUI called via the instruction `python main.py -gui1`. This interface is composed of four tabs namely:
1. **Import_DB** Thanks to the "browse" buttons, select the folder path to the clean images and the abnormal test images. 
2. **Corrupt_DB** Define the specification of the synthetic noise model used to corrupt training images of a specific Database imported before. Default values are the one used in the paper. Also, a folder with test images corrupted with synthetic noise is also created at this time. 
3. **Train_Net** Define the network specifications and the training parameters in `hjson` files. 
4. **Eval_net** Run evaluation on a trained network. This can include performance evaluation with a residual- and/or an uncertainty-based approach.

<p align="center">
  <img width="700" src="https://github.com/anncollin/AnomalyDetection-Keras/blob/master/ReadmeImgs/GUI.gif"> <br>
</p> 
