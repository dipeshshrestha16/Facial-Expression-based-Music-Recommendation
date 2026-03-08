"""Emotion prediction module for analyzing facial expressions."""
import keras
import numpy as np
import cv2

model = keras.models.load_model(
    r"C:\Users\user\python-class\emotion_music_flask\models\emotion_model.h5"
)

EMOTIONS = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

def preprocess_face(face):
    face = cv2.resize(face, (48, 48))
    face = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
    face = face / 255.0
    face = np.reshape(face, (1, 48, 48, 1))
    return face

def predict_emotion_from_face(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)

    if len(faces) == 0:
        return "neutral"

    x, y, w, h = faces[0]
    face_roi = frame[y:y+h, x:x+w]

    processed = preprocess_face(face_roi)
    prediction = model.predict(processed, verbose=0)
    emotion = EMOTIONS[np.argmax(prediction)]
    return emotion