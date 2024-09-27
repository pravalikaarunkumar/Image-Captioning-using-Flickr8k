Image Captioning Model using LSTM and DenseNet

Overview
This repository contains an implementation of an image captioning model that utilizes DenseNet for image feature extraction and LSTM for generating captions. The model was trained using the Flickr 8k dataset on an L4 GPU.

The goal of this project is to generate descriptive captions for images by combining convolutional neural networks (CNNs) for image analysis with recurrent neural networks (RNNs) for sequential data processing.

Features
Image feature extraction using DenseNet.
Caption generation using LSTM.
Trained on the Flickr 8k dataset.
Utilized NVIDIA L4 GPU for efficient training.
Dataset
Flickr 8k Dataset
The Flickr 8k dataset consists of 8,000 images, each associated with five different captions. The dataset is a commonly used benchmark for image captioning tasks.

Download the dataset: Flickr 8k Dataset
Preprocessing
Each image was resized to 224x224 pixels before being passed through DenseNet.
Captions were tokenized and converted to sequences of integer tokens using the Keras Tokenizer.
A vocabulary of the most frequent words in the dataset was created.
Model Architecture
DenseNet

The DenseNet network is used to extract rich image features. The pre-trained DenseNet model from ImageNet is used, and the output from the last pooling layer is taken as the image feature vector.

LSTM

The LSTM network processes the sequence of tokens in the captions and generates the next word in the sequence at each time step. This recurrent model is essential for handling the sequential nature of language data.

Training
The training process involved the following key steps:

Feature Extraction: DenseNet was used to extract image features, which were then input to the LSTM network.
Caption Generation: The LSTM model predicts the next word in the caption sequence, given the image feature and the previous words.
Loss Function: Categorical Cross-Entropy was used for optimizing the model during training.
Optimization: The Adam optimizer was used with a learning rate of 0.001.
Training Hardware
The model was trained on a NVIDIA L4 GPU provided by [insert platform or service used].

Results
After training for X epochs, the model achieved the following results on the validation set:

BLEU Score: X
CIDEr Score: X
Here are some example outputs from the model:

Generated Caption: "A dog running through the grass."

Generated Caption: "A man riding a bike down the street."
