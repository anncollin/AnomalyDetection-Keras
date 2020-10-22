# Anomaly Detection Framework for Industrial Vision

## Table of contents
- [Overview](#Overview)
- [Dependencies](#Dependencies)
- [Key Components of the Implementation](#Key-Components-of-the-Implementation)
  - [Definition of the Stain noise corruption](#Definition-of-the-Stain-noise-corruption)
  - [The Graphical User Interface](#The-Graphical-User-Interface)

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
