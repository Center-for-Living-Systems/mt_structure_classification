### Image Classification of Microtubule-GUV Structures

This repository contains a complete workflow for performing image classification on grayscale microscopy images of microtubule-GUV (Giant Unilamellar Vesicle) structures. The project includes model training, evaluation, visualization, and inference on new test datasets.

#### Project Overview

Input Data: Grayscale PNG images with corresponding labels provided via CSV

Models Supported: ResNet18, EfficientNet-B0, ConvNeXt-Tiny

Output: Predicted labels, classification statistics, and visualizations

#### Features

Custom Dataset Loader for grayscale images

Train/Val split with augmentation

Support for multiple deep models using torchvision

Model saving and loading with timestamped directories

Confusion matrices and accuracy/loss plots

GUV content area calculation for each prediction

Grouped bar plots by concentration and GUV diameter

<img src="https://github.com/user-attachments/assets/0ba464ca-b8cd-4d16-a89e-af54d19984f7" style="width:70%;"/>

#### Dependencies

Python 3.8+

PyTorch 1.10+

torchvision

pandas, matplotlib, seaborn

PIL, scikit-learn

#### Acknowledgments

This pipeline was developed to support the classification of cytoskeletal structures in microscopy images from microtubule experiments involving GUVs. Inspired by efforts at the Center for Living Systems, University of Chicago.

Feel free to fork, use, and contribute!

