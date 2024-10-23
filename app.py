from flask import Flask, render_template, Response
from utils.camera import VideoCamera


app = Flask(__name__)

@app.route('/')
def index():
    """Render the main page."""
    return render_template('index.html')

def gen(camera):
    """Generate frames for the video feed."""
    while True:
        frame = camera.get_frame()
        if frame is not None:
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/video_feed')
def video_feed():
    """Video streaming route."""
    return Response(gen(VideoCamera()), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    app.run(debug=True)
