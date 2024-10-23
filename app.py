from flask import Flask, render_template, Response, jsonify
from utils.camera import VideoCamera


app = Flask(__name__)

current_emotion = "Detecting..."

@app.route('/')
def index():
    """Render the main page."""
    return render_template('index.html')

def gen(camera):
    """Generate frames for the video feed."""
    global current_emotion
    while True:
        frame, emotion = camera.get_frame()
        if frame is not None:
            current_emotion = emotion
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/video_feed')
def video_feed():
    """Video streaming route."""
    return Response(gen(VideoCamera()), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/get_emotion')
def get_emotion():
    return jsonify({'emotion': current_emotion})  # Serve the current emotion as JSON


if __name__ == '__main__':
    app.run(debug=True)
