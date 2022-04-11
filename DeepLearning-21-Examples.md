The example codes of this book is as follows: https://github.com/dailai/DeepLearning-21-Examples. The examples are very representative,
 however, the content might be little out-of-date, publised in 2018 and applied based on the python 2. 
 
Thus, this tutorial will not focus on much on the technicial implementation part, but on the structure of the Deep Learning field and the possible connections 
between various proposed models.

This book will cover the basic models and datasets in the Computer Vision, Natural Language Processing and Reinforcement Learning areas.

the environment of this book was Ubuntu 14.04，Python 2.7、TensorFlow >= 1.4.0

# Patten Recognize in MNIST dataset
## MNIST
Could be seen as the 'hello world' dataset of machine learning field. 1~10 number images in black-white color

## Softmax
The softmax is the linear regression model for multi-classification tasks, while the Logistic regression is for binary-classification tasks.

## Convolution Layer
Could greatly help to the simple CV tasks to improve the acc to a very high level, near 99% here.

# Patten Recognize in CIFAR-10/ImageNet dataset
## CIFAR-10
Contain the real-world object images, which is in RGB colors, like airplane, bird, cat, horse, ship, truck, etc.
Due to the real-world objects complexity in the captured images, the simple linear softmax regression model would perform very bad here.

## Data Augmentation
Here, the data augmentation method was introduced to improve the model performance. Like rotate, shift, zoom, color-change. 
So the the training sample could be bigger and the model could best understand the difference.

## ImageNet
Proposed in 2016, more pictures, and in higher resolutions, and also containing more irrelevant noise and changes. 

The famous models like **AlexNet, VGGNet, GoogLeNet, ResNet**, performs great. Especially for Deep Residual Network, ResNet, the error rate is decrease to 3.57%, 
first time the machine learning models outperform the human.

# Fine-tune the Models 
For the pre-trained models like VGG16, fine-tune the model that has already trained on the dataset, like ImageNet.

The fine-tuning process could happen for training:
- only the last fully connected layer, treat the VGG16 as one feature extraction model
- all the parameters, slower but might achieve better performance
- only the parameters from deep layers like conv3, conv4, conv5, fc6, fc7, fc8, but do not train the conv1 and conv2

# Deep Dream to Understand Convolution Layers
The tech is set one specific output label probility, like banana, as the optimization objective, to make the model adjust the pixel values of the input 
image, so that the output image could make the output probility for banana reach the maximum. For the fixed dataset and model. The dataset here is ImageNet and model is Inception model. The input can contain only random noise. 

Considering the possible limitation of GPU RAM when the optimization is for a very high resolution picture, the simple way can be segment the original picture into small pieces and optimize one piece for one time. 

To further imporve the image quality, i.g. aiming at making the generated picture become soft where baseline picture pixel values change sharply in some small parts. The core idea is that reducing the high-frequency parts, where the pixel values change sharply. The mainstream methods are:
- increase the gradient of low-frequency parts
- **Laplacian Pyramid Gradient Normalization**:

# Object Detection in Deep Learning
## R-CNN
Full name is Region-CNN, first alogrithm that applied DL into object detection.





