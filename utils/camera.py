import cv2
from utils.face_detection import detect_face_and_landmarks
from utils.emotion_detection import predict_emotion

class VideoCamera:
    def __init__(self):
        """Initialize the video camera."""
        self.video = cv2.VideoCapture(0)

    def __del__(self):
        """Release the video capture when done."""
        self.video.release()

    def get_frame(self):
        """Capture the video frame and return it along with emotion prediction."""
        while True:
            success, frame = self.video.read()
            if not success:
                continue

            # Convert frame to grayscale for face detection
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            
            # Detect faces and landmarks
            faces, landmarks_list = detect_faces_and_landmarks(gray)

            # Loop over detected faces and predict emotions
            for landmarks in landmarks_list:
                emotion = predict_emotion(landmarks)

                # Draw face bounding box and emotion label on the frame
                x, y, w, h = faces[0]
                cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
                cv2.putText(frame, emotion, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)

            # Encode the frame for streaming
            ret, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()

            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

