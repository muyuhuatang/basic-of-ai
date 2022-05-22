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
Trianing process of R-CNN is;
- train AlexNet model on the ImageNet dataset
- fine-tune the model on the object detection dataset
- Apply Slective Search method to search the target region, and use the fine-tuned model to extract features in those regions, then store the extracted features
- Use the stored features to train SVM classifier

Due to the strong feature extraction capability of CNN networks, the R-CNN outperform traditional methods in the VOC 2007 dataset.

However, the shortcoming is that it is relatively computational expensive, because the feature extraction process is for every extracted region.

## Improved R-CNN
### SPPNet
Spatial Pyramid Pooling Convolutional Networks, the contribution is that it changed the CNN input size from fixed-size to adjustable-size through ROI Pooling layer, while keeping the output size of CNN remain the same. SPPNet still use the SVM as the classifier. 

Intuitively understanding, in R-CNN, the model extract the feature after selecting the target regions, the convolution happens for each single region separately; in SPPNet, the model execute the convolution computation at first, so it will only need process the convolution computation for once, which saved a lot of time.

### Fast R-CNN / Faster R-CNN
Different from SPPNet, the Fast R-CNN use the neural network to conduct classification, obtained better acc than SPPNet.

Still, the Faster R-CNN still applied the Slective Search method to located regions, which will cost the quite a few time. So in the Improved version of Faster R-CNN model, it used the RPN (Region Proposal Network) to located the regions, obtained faster speed and better acc.

### RPN (Region Proposal Network)
Applied the convolution layer to convolute the raw input data features.

## Further readings 
- R-FCN: object detection via region-based fully convolution networks
- YOLO: You Only Look Once - Unified, Realy-Time Object Detection
- SSD: Single Shot MultiBox Detector
- YOLO9000: Better, Faster, Stronger

# Human Face Detection and recognizetion
## MTCNN model
Step: detect all the human faces in the picture, then conduct Face Alignment base on the Landmark point of human faces like eyes, noses, mouthes.
For the MTCNN, MT means Multi-Task, it contains P-Net, R-Net, O-Net:
- P-Net, R-Net, O-Net: face classification, bounding box regression, facial landmark localization
- Difference is that P-Net check the region size of 12x12x3, R-Net check 24x24x3, O-Net check 48x48x3.
- The prior networks could be perceived as filters, passing the filtered information to the later network, and use the biggest but slower network to give out the final classifications. THus, the overall processing time is reduced

## Loss function design for the human faces (ensure the feature extraction quality)
Requirement:
- the vectors in the same kinds as close as possible
- the vectors in the differencet kinds as far as possible 

### Triplet Loss 
Intuitively, the loss is defined as the distance parameters between both same human face vectors and different human face vectors. In the definition formula, the distance of same human face vectors adding a constant alpha should be no bigger than the distance of different human face vectors (compared with same baseline, so there is three vectors in each calculation), and the loss function records the value of former minus latter when the former is actually bigger (in the perfect setting, it should be smaller).

The triplet loss directly optimize on the distant to solve the feature representation problem of human face recognization. However, the selection of triplet instances could be very triky: if randomly selected, then it would be hard to achieve the best performance after converge; if selected the most hard instances, it would be hard to converge. Thus, one sampling approach is to select the Semi-hard instances to train the model, to help the model obtain good performance while be able to converge. Besides, applying the Triplet Loss would require very large human face datasets to train to get a good result.

### Center Loss
The center loss do not directly optimize on the distance. It keep the original classification model, but create a class center for each class (in the human face recognization problem, one class present a human). Here, the images in one class should be as close as possible to their class center, and the centers of different classes should be as far as possible. 

The center loss approach do not need the specific sampling method like the triplet loss, and requires less image the well train the models. However, the problem is that if calculate the center in each epoch, it would be too computational expensive, thus one approach is the randomly select the center and learn/update the center location parameter during iterations. Applying the cener loss would help the model to obtain a better discrimitive among different classes.

## Application Scenarios
- Face Identification: identify if the gicen A, B should be the same person
- Face Recognition: (most popluar) given one image, detect the best suitable (similar) faces in the database.
- Face Clustering: clustering the human faces in the database, e.g. through K-Means.

### Famous dataset
LFW (labeled Faces in the Wild) dataset, obtained from the Internet, containing 13000 human face images labelled with name, more than 1680 people have multiple images collected (open sourced dataset)

### Pipeline
1. Use the MTCNN model to regonize and uniform the human faces into suitable format.
2. Use the CNN model, e.g., Iception ResNet v1 to train and test on the LFW dataset (pre-trained by a very large human face dataset, named MS-Celeb-1M) (can also pre-train on the dataset of CASIA-WebFace dataset, containing 494414 images from 10575 people).

# Image Style Transfer
## Image Restore - Image Style Transfer
Use the Gram metrix to represent the style of the input image, similar to the simple restoring image content

Referring the image content loss, we can define a 'style loss', through the different between the Gram metrix of initial image and restored image, which is to some extent restoring the image by considering primarily the style but not the content.

- image style transfer = image content restore + image style restore

# GAN & DCGAN
## GAN
GAN is consists of a Generator and Discriminator, the loss function part is interesting, for Dsicrimiator, it hopes the loss becomes larger, for Generator, it hopes the loss becomes lesser. Thus the trianing process is one step for optimizing D, next step for optimizing G, and then the D, and then the G, ...

## DCGAN
DC is for Deep Convolutional, where the DCGAN is specifically designed for generating the image samples. 

In the GAN paper, actually they did not made constrains on the structure of D and G. In DCGAN paper, for the image specific purpose, the D and G are both the Convolutional Neural Network. 

Moreover, in G and D:
- do not use the pooling layer, use the stride method instead
- apply the Batch Normalization to help the model converge
- in G, the activate function for last layer is tanh, the rest are all ReLU; in D, all activate functions are Leaky ReLU

# 






