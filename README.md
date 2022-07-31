# Project 3: Image Classification to Classify Concretes With or Without Cracks

## 1.0 Introduction
This project has been carried out to identify cracks on concretes by using a convolutional neural network, CNN with high accuracy. The dataset is obtained from [here](https://data.mendeley.com/datasets/5y9wdsg2zt/2). The dataset contains concrete images with cracks. The dataset is further divided into two as negative and positive crack images for image classification. Each class has 20000 images with a total of 40000 images with 227 x 227 pixels with RGB channels. 

## 2.0 Framework
Spyder is used in the process of completing this project along with necessary libraries such as Numpy, Pandas, Matplotlib and Tensorflow Keras.


## 3.0 Methodology
### 3.1 Data Pipeline
The images are imported and splitted into train and validation dataset (70:30). The validation data is then further split into two portion to obtain some test data, with a ratio of 80:20. The number of validation batches and test batches are 750 and 150 respectively. 

### 3.2 Model Pipeline 
Transfer learning is adapted in building the deep learning model of this project. Firstly, a preprocessing layer is created to accept raw images as input so the model can handle feature normalization. As for feature extractor, a pretrained model, MobileNet V2 is used in creating the base model. For the classification layers, a global average pooling and dense layer are used to output softmax signals that identifies the predicted class. 


## 4.0 Results



