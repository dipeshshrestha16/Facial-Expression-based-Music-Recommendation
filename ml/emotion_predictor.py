"""Emotion prediction module for analyzing facial expressions."""
import keras
import numpy as np
import cv2

model = keras.models.load_model(
    r"C:\Users\user\python-class\emotion_music_flask\models\emotion_model.h5"
)

EMOTIONS = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']

def preprocess_face(face):
    face = cv2.resize(face, (48, 48))
    face = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
    face = face / 255.0
    face = np.reshape(face, (1, 48, 48, 1))
    return face

def predict_emotion_from_face(face):
    processed = preprocess_face(face)
    prediction = model.predict(processed, verbose=0)
    emotion = EMOTIONS[np.argmax(prediction)]
    return emotion
