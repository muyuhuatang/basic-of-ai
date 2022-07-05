The example codes of the book, <DeepLearning-21-Examples>, is as follows: https://github.com/dailai/DeepLearning-21-Examples. 

The examples are very representative,
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

# pix2pix Model and Auto Color Technique
## cGAN
We can easliy use the GAN to conduct unsupervised learning, and then generate the new samples. Still, it is unable to exactly control the type of the generated new samples. For example, the vanilla GAN can not generate specific MNIST numbers like 2, the generation results are random.

Thus, the cGAN (Conditional Generative Adversarial Networks), it add the additional condition y to both Generator and Discriminator, which is the aimed the specific target (or to say the condition). The Generator G should then generate the new samples that match the condition y, and the Discriminator should discriminate both whether the generated image is really and whether the generate is matching the condition y.

## pix2pix
Just like the machine translation that translate one natural language to another natural language in the NLP field, there is also the 'image translate' tasks like 'day to night', 'BW to color', 'aerial to map', 'edges to photo'. 

When experimenting the different loss functions like L1, Loss of cGAN, L1 + loss of cGAN, find out that if only us ethe L1, the generated images would be very vague, loss of cGAN is much better, and L1 + loss of cGAN is only slightly better than only loss of cGAN.

Besides, the pix2pix model also proposed the concept PatchGAN during improving the Discriminator, which is calculating and averaging the probilities for small NxN patches but not whole image. The PatchGAN could help to accelerate the calculation speed and converge process. As for the result, when choosing an adequate patch size, like 70x70 to 256x256, the generated images is not far from when using the whole image to calculate the probility.

In short, the pix2pix model is still a cGAN, but modified specifically for the image translation tasks.

# Super-resolution
Based on the pix2pix model that has just been introduced.

## Dataset
COCO dataset 2014

# CycleGAN & Unpaired Image Translation
The motivation is that in the really world, the paired image dataset is hard to collect, and sometime do not need that fine-grained dataset, like for the photo to oil painting translation. 

The CycleGAN is capable of using unpaired data to obtain a reflection F from X space to Y space compared to pix2pix model. 

The difficulty here is that the reflection F may reflect all the x in X space into one y in the Y space. Thus the author introduce the 'cycle consistency loss' to define another reflection G that from Y space to X space and make the F(G(y)) ~ y and G(F(x)) ~ x, which means that the x to y can be reversed back to original x again. 

## NOTE
Since the training process of GAN model is initially randomed, thus if find some picture output during training is opposite to the aim (like aim at light color but generating the dark color), should stop trianing and re-start the training from the begining.

# RNN and CharRNN (text generation)
The structure is specially designed for the sequence data, like natural language text, audio and time series (like stock).

The parameters used in each recurrent steps are shared (same matrix). 

In the vanilla structure of RNN model, the lengh of the input and output sequence should be equal. 

## N vs. 1 RNN
Here, only need to make the output after the last point of the vanilla sequence output structure.

## 1 vs. N RNN
Here, only input one datapoint, or use the same input data point in every recurrent steps.

Application scenarios: 
- image to text (image caption)
- use the type (keywords) to generate auto or music

## LSTM (Long Short-Term Memory)
Considering the gradient vanishing or exploding problem, the LSTM use the add function to replace the recurrenting matrix multiple process to diminish the gradient vanishing problem, and make the neural network to learn the long term features.

## Char RNN (N vs. N)
Use the model to predict the next Char based on the inputted Char, e.g., {h,e,l,l,o} -> {e,l,l,o,!}

# Sequence Classification Problem
N vs. 1 RNN structure

# Word Vectors and Word Embeddings
Word embedding normally refers to transfer a word into a vector representation, sometines called word2vec, traidtional methods like CBOW and Skip-Gram. 

## Why word embedding?
For the one-hot is completed equally perceive the words in the dictionary, but neglecting the correlations between specific words. And for the word2vec, it means that to learn a reflection f to represent the word in the vector space. Normally, the dimension of word vectors is much less than the vocabulary. 
 
When using the word2vec represented words, the model could obtain much more abundent word information, and the dimension of input is significantly reduced.

## Foundation Theory of the Word Embedding
How to learn a good reflection F?
One way is to count the probability that one word is appearing with other words, and put the often co-occur word in the close place of the vector space; another way is to predict, from one word or several words to predict their possible neighborring words, then the reflection is naturally obtained when doing the orediction. The later method is more welcomed in the traditional approaches, like CBOW and Skip-Gram.
 
## CBOW (Continuous Bag of Words)
Use the content of one word to predict the word.

However, considering the large amount in dictionary, which could cost a lot time to train the prediction model. So the researcher simplified the model into a bi-classification task between target word and randomly selected noise words.
 
## Skip-Gram
Opposite to the CBOW method, the Skip-Gram is use the missing word to predict the content words, so it can be percieved as prediction one word based on given one word.
 
## Visualization
Traditional method is to apply the t-SNE method to convert a high dimensions vector, like 128, into a 2 dimensions vector.
 
# AR models
The AR (Autoregressive model) is one basic approach to handle the time-serise tasks, for example, LSTM.
 
# Neural Network Machine Translation
Unlike the traditional methods, the Neural Network Machine Translation Apporach is to first transfer the source input into vectors and then use the model to generate vectors which can then be transfered into the target translation results.
 
## Encoder-Decoder Architecture
It is also called Seq2Seq model, whcih could effectively solve the in-equal problem between input and output sequence in the N vs. N RNN models.
 
## Attention Mechanism
In the encoder-decoder architecture, the encoder encode all the input sequence to a uniformed semantic feature c for further decoding, Thus, the lengh of c is the bottleneck of the model's performance. And for the machien translation task, when the input sequence is too long, then the c would be not sufficient to carry all required information.
 
Then the attnetion mechanism solve the problem via providing different c in different time, which means that every c will automatically select the most suiable context for the y. The attnetion scores a(ij) are also 'learnt' by model, it is related to the (i-1)th hidden state of decoder and (j)th hidden state of encoder.
 
# Convert Image to Text
AKA, the Image Caption task, means that automatically generate a descriptive text for the given image:
- detection of the objects in the image
- understanding the relations between detected objects
- expression by reasonable and meaningful descriptions

## Encoder-Decoder Architecture with CNN-encoder and RNN-decoder
Use the CNN to extract the information and feature with the given image.

### Adding Attention Mechanism
Paper 'Show, Attent and Tell: Neural Image Caption Generation with Visual Attention'
 
### Adding high-level Semantic
Here, the high-level semantic can be seen as one multi-label classification task, which is representing what objects exist in the given image.

# Q learning
For Reinforcement Learning, studying given environment conditions how to maximize the rewards. Agent, Environment, Action, Reward.
During the updating process of Q learning, every step the instruction is to select the next step action based on the current state and Q function values. The Q(s,a) is the expected reward value that execute the a action in the s state.

# SARSA Algorithm
SARSA (State-Action-Reward-State-Action) algorithm is one basic reinforcement learning algorithm. Similar to Q learning, it is also recurrently learning the Q function during the Agent conducting Action in the Environment.
The difference is that, in Q learning using epsillon-greedy to select the next step action, the agent might not execute the max_action; while for SARSA algorithm, the next action will definitely be the selected new_action. (off-policy vs. on-policy)
 
Intuitively understanding, the SARSA alogrithm is less aggressive than Q learning. Q Learning will use the Q[new_state, :].max() to update the Q value, which means that it is considering the maximize reward without much the risks behind the new states. However, the SARSA algorithm is only using Q[new_state, new_action] to update Q value, means that it will consider more about the possible negative rewards condition, thus the agent will hold still in the maze problem, and consequently harder to find the ultimite 'treasure in the maze'.
 
# Deep Q Learning
The advanced version of vanilla Q Learning, it use the deep neural networks to represent the Q function. 
The aim is to automatically learn how to play the video game from the initial scene of video games. The algorithm can only get the video game screen and the rewards of the game, while the more detailed information of the video game play is not provided.
 
Here the DQN (Deep Q Network) use the deep convolutional neural network to represent the Q function. The input is the the state s, the output is the Q values of every action. The input state is not one frame of the video game, but multiple frames of the video game play. 
 
## Experience Replay Mechanism
To generate the training samples for the designed DQN. The agent will firstly try to play the game and accumulate the experience as a 'pool', i.g., D={e(1),e(2),...,e(N)}. The ei = (s(t),a(t),r(t),s(t+1)), where the s(t) is the state in the time t, a(t) is the action in time t, r(t) is the award vaule here, and the s(t+1) is the next time state. 
 
# Policy Gradient Algorithm
For the reinforcement learning algorithm, the aim is to learn a value function Q(state, action). In the environment, the first is to confirm the current state and then choose one action with relatively high reward based on the Q(state, action) fucntion. The policy gradient method here is different, it will not learn the value function Q, instead, it will use the model (e.g., neural networks) to output the required actions.
 
Unlike the DQN, the Policy Network get the input of current state, but output the best next action (the action itself or the probility of the certain action).
 
## A (Advantage)
Set this to indicate whether the selected action a is the 'correct action' or 'wrong action' (mostly based on whether the game will fail).












