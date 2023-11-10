# Speech-Emotion-Recognition-using-RNN
Speech Emotion Recognition with Bidirectional LSTM  Efficient SER using Bidirectional LSTM. Features: MFCCs, chroma, mel spectrogram. Dataset prep, training, and prediction function included. Easy adaptation for diverse SER tasks.
This repository implements a Speech Emotion Recognition (SER) system using a Bidirectional Long Short-Term Memory (LSTM) neural network. The code features audio file feature extraction, dataset preparation, and training of the deep learning model. The Bidirectional LSTM architecture excels in capturing temporal dependencies, making it effective for SER.

Key Features:

Utilizes Mel-frequency cepstral coefficients (MFCCs), chroma, and mel spectrogram features for audio file feature extraction.
Implements a Bidirectional LSTM model with multiple layers and dropout for regularization.
Handles data preparation with label encoding and one-hot encoding.
Incorporates early stopping to prevent overfitting during model training.
Provides a function for making emotion predictions on new audio files.
Usage:

Mount Google Drive to access the dataset.
Extract features from audio files and organize them into training and testing sets.
Build and train the Bidirectional LSTM model.
Make predictions on new audio files using the provided prediction function.
Explore and adapt the code to suit your specific SER tasks. The repository is designed to be concise and user-friendly, facilitating the implementation of SER systems using deep learning techniques.
