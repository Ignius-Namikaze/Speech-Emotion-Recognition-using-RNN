import librosa
import os
import sklearn
import numpy as np
from sklearn.preprocessing import LabelEncoder

def extract_features(file_path, mfcc=True, chroma=True, mel=True):
    try:
        # Load audio file
        y, sr = librosa.load(file_path, sr=None)
        # Extract features
        features = []
        if mfcc:
            mfccs = np.mean(librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13).T, axis=0)
            features.extend(mfccs)
        if chroma:
            chroma = np.mean(librosa.feature.chroma_stft(y=y, sr=sr).T, axis=0)
            features.extend(chroma)
        if mel:
            mel = np.mean(librosa.feature.melspectrogram(y=y, sr=sr).T, axis=0)
            features.extend(mel)
        return features
    except Exception as e:
        print("Error encountered while processing {}: {}".format(file_path, e))
