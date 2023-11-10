from PyQt5.QtWidgets import *
from PyQt5.QtGui import *
from PyQt5.QtCore import *
import sys
from PyQt5 import uic
import os
import numpy as np
import cv2
import pickle
import backend as bk
import sounddevice as sd
from scipy.io.wavfile import write
import wavio as wv
from tensorflow.keras.models import load_model

class mainwindow(QMainWindow):
    def __init__(self):
        super().__init__()
        uic.loadUi("Emtional.ui", self)
        self.setWindowTitle("Emotion Predicter")

        #variables
        self.filename = None
        self.model = load_model('model.h5')

        #widgets
        self.aud_name = self.findChild(QLabel,"aud_name")
        self.rec = self.findChild(QPushButton,"record")
        self.sel_file = self.findChild(QPushButton,"select_file")
        self.pred_emo = self.findChild(QPushButton,"predict_emo")
        self.emo = self.findChild(QLabel,"Emotion")

        #connections
        self.sel_file.clicked.connect(self.seling)
        self.rec.clicked.connect(self.recording)
        self.pred_emo.clicked.connect(self.preddinf)

        self.show()

    def seling(self):
        if not self.filename:
            self.filename, _ = QFileDialog.getOpenFileName(self, 'Select Audio File', QDir.currentPath(),
                                                      'Audio *.wav')
            nme1 = self.filename.split("/")
            nme2 = nme1[len(nme1)-1]
            self.aud_name.setText(nme2)
            if not self.filename:
                return

    def recording(self):
        freq = 14400
        duration = 3
        recording = sd.rec(int(duration * freq), samplerate=freq, channels=1)
        sd.wait()
        os.chdir(f"{os.getcwd()}//temp")
        write("recording0.wav", freq, recording)
        self.filename = f"{os.getcwd()}\\recording0.wav"
        self.aud_name.setText("recording0.wav")

    def preddinf(self):
        if self.filename != None:
            pred_feature = []
            pred_features = bk.extract_features(self.filename)
            pred_feature.append(pred_features)
            Pred_X = np.array(pred_feature)
            X_pred_lstm = Pred_X.reshape(Pred_X.shape[0], 1, Pred_X.shape[1])
            pred = self.model.predict(X_pred_lstm)
            predicted_label_index = np.argmax(pred)
            moods = ['Angry','Disgust','Fear','Happy','Neutral','Pleasant/nSurprise','Sad']
            predicted_label = moods[predicted_label_index]
            self.emo.setText(predicted_label)
        else:
            pass


app = QApplication(sys.argv)
UIWindow = mainwindow()
app.exec_()

