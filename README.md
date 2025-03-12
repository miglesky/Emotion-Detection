
# Emotion Detection Using Convolutional Neural Networks (CNNs)

This project implements an emotion detection system using a Convolutional Neural Network (CNN) trained on the **FER2013 dataset**. The model classifies facial images into one of seven emotion categories: **disgust, fear, happy, sad, angry, neutral**, and **surprise**.

## Dataset
The **FER2013 dataset** consists of grayscale images of faces, each labeled with one of seven emotion categories. 

- **Training Set**: 28,709 images
- **Test Set**: 7,178 images

The dataset is organized into directories corresponding to each emotion class.

## Preprocessing
The images are preprocessed as follows:
- **Resized to 48x48 pixels**
- **Converted to grayscale**
- **Data augmentation** and **rescaling** using `ImageDataGenerator`

## Model Architecture
The model follows a **Sequential** CNN structure, with the following layers:
- **Conv2D layers**: Multiple convolutional layers using ReLU activation, progressively increasing the number of filters (32, 64, 128, 256).
- **MaxPooling layers**: Reducing the spatial dimensions after each convolutional block.
- **Dropout layers**: Added after pooling layers to reduce overfitting.
- **Flatten layer**: Converts the 3D output of convolution layers into a 1D vector.
- **Fully Connected (Dense) layers**: The final Dense layer uses a softmax activation function for multi-class classification.

## Compilation
The model is compiled with:
- **Optimizer**: Adam
- **Loss function**: Sparse categorical cross-entropy (since labels are integers)
- **Metrics**: Accuracy

## Model Summary
The model consists of **1.39 million parameters**.

## Usage
1. Train the model on the FER2013 dataset.
2. Use the trained model for real-time emotion detection from facial images.

Feel free to explore and experiment with the model for emotion detection in video frames or images.



