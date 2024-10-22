from flask import Flask, render_template, Response
from utils.camera import VideoCamera
import threading

app = Flask(__name__)

# Video camera object
video_camera = VideoCamera()

@app.route('/')
def index():
    """Render the main page."""
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    """Provide the video feed to the frontend."""
    return Response(video_camera.get_frame(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    app.run(debug=True)
