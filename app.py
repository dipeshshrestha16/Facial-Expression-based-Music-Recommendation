from flask import Flask, render_template, Response
import cv2
from ml.emotion_predictor import predict_emotion_from_face
from ml.music_recommender import recommend_music

app = Flask(__name__)

camera = cv2.VideoCapture(0)
last_emotion = "neutral"

def gen_frames():
    global last_emotion

    while True:
        success, frame = camera.read()
        if not success:
            break

        # Detect emotion
        emotion = predict_emotion_from_face(frame)
        last_emotion = emotion

        cv2.putText(
            frame, f'Emotion: {emotion}',
            (20, 40), cv2.FONT_HERSHEY_SIMPLEX,
            1, (0, 255, 0), 2
        )

        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(gen_frames(),
        mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/result')
def result():
    songs = recommend_music(last_emotion)
    return render_template(
        'results.html',
        emotion=last_emotion,
        songs=songs
    )

if __name__ == '__main__':
    app.run(debug=True)
