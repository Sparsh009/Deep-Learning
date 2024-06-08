# Deep-Learning
Neural Network &amp; Deep Learning

# CIFAR-10 Neural Network Architecture

## Introduction
This repository contains the implementation of a custom neural network designed to classify images from the CIFAR-10 dataset. The implementation highlights the steps involved in data loading, pre-processing, architecture setup, and both training and testing phases.

## Architecture Overview

### Initial Setup
- **Dataset**: CIFAR-10, consisting of 60,000 32x32 color images across 10 classes.
- **Data Preprocessing**: Transformation of images into PyTorch tensors, normalization for training stability.

### Model Architecture
- **Intermediate Blocks (B1, B2, B3)**: Each block consists of multiple convolution layers that process input independently in parallel.
- **Output Block (O)**: Implements average pooling to compress image features into a vector, which is then fed into a fully connected layer to produce logits for classification.

### Custom Enhancements
- **Data Augmentation**: Random horizontal flip and random rotation up to 10 degrees.
- **Batch Normalization**: Normalizes the output of previous activation layers to stabilize learning.
- **Activation Function**: ReLU to mitigate the vanishing gradient problem.
- **Dropout**: Added with a rate of 0.5 to reduce overfitting.
- **Softmax Function**: Used in the intermediate block to apply a learned weighted summation to channel means.

### Training and Hyperparameters
- **Optimizer**: Adam with a learning rate of 0.001.
- **Loss Function**: Cross-entropy loss.
- **Epochs**: 40 with a batch size of 64.

## Results
After enhancements, the highest recorded accuracy on the testing dataset is 82.24%, marking a significant improvement from the initial 40% accuracy over 40 epochs.

## Usage
Instructions on how to run and train the model, including environment setup and execution commands.

## Requirements
List of libraries and tools required to run the model, provided in a `requirements.txt` file.

## Contributing
Guidelines for contributing to the repository, including code style, pull request process, etc.

## License
Specify the license under which the project is released.

## Acknowledgments
Credits to individuals, institutions, or resources that were instrumental in the development of this project.

