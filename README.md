# Image Caption Generator using BiLSTM and Flickr8k DataSet

## Introduction

This project focuses on creating an Image Caption Generator using the Flickr8k DataSet. The generator combines Convolutional Neural Network (CNN) for image feature extraction and Long Short-Term Memory (LSTM) for Natural Language Processing (NLP). We will be using Keras with TensorFlow as the backend for this project.

## Setup

Before running the code, ensure you have the required dependencies installed. The code is designed to be run on Google Colab.

1. **Install Required Packages:**
   - TensorFlow
   - Keras
   - tqdm
   - numpy
   - nltk
   - PIL (Pillow)

2. **Upload Kaggle API Key:**
   - Upload the `kaggle.json` file to your Google Colab environment.

3. **Download Dataset:**
   - Run the provided code to download and extract the Flickr8k dataset from Kaggle.

4. **Mount Google Drive:**
   - Mount your Google Drive to save and load files using the provided code.

## Data Preprocessing

The code involves several preprocessing steps:

- **Extracting Image Features:**
  - Utilizes VGG16 for image feature extraction.
  - Image features are stored using pickle.

- **Loading Captions Data:**
  - Captions are loaded from the provided `captions.txt` file.
  - Captions are mapped to their respective image IDs.

- **Text Data Preprocessing:**
  - Text data is cleaned by converting to lowercase, removing special characters, and adding start/end tags.

- **Tokenization:**
  - Tokenizes the captions to create a vocabulary using Keras Tokenizer.

- **Train-Test Split:**
  - Splits the data into training and testing sets.

## Model Creation

The model consists of an encoder-decoder architecture:

- **Encoder:**
  - Combines image features with sequence features from the LSTM layer.

- **Decoder:**
  - Consists of a Bidirectional LSTM layer followed by Dense layers.
  - Uses softmax activation for the output layer.

- **Compilation:**
  - The model is compiled with categorical cross-entropy loss and the Adam optimizer.
 
![image](https://github.com/the-madman-28/Image-caption-generator-using-BiLSTM/assets/68281837/79785aa5-1d3b-45c0-b3f7-d61e4bd54b89)

## Training

The model is trained using a data generator to handle large datasets efficiently. The training process involves iterating through the dataset for multiple epochs.

## Evaluation

The model is evaluated using BLEU (BiLingual Evaluation Understudy) scores, which measure the similarity between the generated captions and the reference captions.

## Testing

A function is provided to generate captions for new images. You can use the `generate_caption` function by providing the path to the image.

## Note

- Ensure you have the required datasets and paths correctly specified.
- Adjust hyperparameters based on your system's capabilities.
- The code assumes that the Kaggle API key (`kaggle.json`) is uploaded to your Colab environment.
- Save the trained model for later use.

