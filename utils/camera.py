import cv2
import dlib
import numpy as np
import joblib

# Load the pre-trained SVM emotion classifier
print("Loading the pre-trained emotion classifier...")
emotion_classifier = joblib.load('models/emotion_classifier.pkl')
print("Emotion classifier loaded successfully.")

# Load dlib's pre-trained shape predictor
landmark_predictor = dlib.shape_predictor('models/shape_predictor_68_face_landmarks.dat')
face_detector = dlib.get_frontal_face_detector()

# Function to extract geometric features from landmarks
def extract_geometric_features(landmarks):
    """Extract geometric features from facial landmarks."""
    features = []
    
    # Example: distance between eyes, width of mouth, etc.
    eye_distance = np.linalg.norm(np.array(landmarks[36]) - np.array(landmarks[45]))
    mouth_width = np.linalg.norm(np.array(landmarks[48]) - np.array(landmarks[54]))
    
    features.append(eye_distance)
    features.append(mouth_width)
    
    return np.array(features)

# Video stream capture class
class VideoCamera:
    def __init__(self):
        # Initialize video capture (use the webcam as the video source)
        self.video = cv2.VideoCapture(0)

    def __del__(self):
        # Release the video capture when done
        self.video.release()

    def get_frame(self):
        """Capture a frame from the video stream, process it, and return the result with emotion predictions."""
        ret, frame = self.video.read()

        if not ret:
            print("Failed to capture frame from video stream.")
            return None

        # Convert frame to grayscale for face detection
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Detect faces in the frame
        detected_faces = face_detector(gray, 1)

        for face in detected_faces:
            # Extract landmarks
            landmarks = landmark_predictor(gray, face)
            landmark_points = [(landmarks.part(i).x, landmarks.part(i).y) for i in range(68)]

            # Extract geometric features
            features = extract_geometric_features(landmark_points)

            # Predict emotion using the pre-trained model
            predicted_emotion = emotion_classifier.predict([features])[0]
            print(f"Predicted Emotion: {predicted_emotion}")

            # Draw a rectangle around the face and display the emotion on the frame
            cv2.rectangle(frame, (face.left(), face.top()), (face.right(), face.bottom()), (255, 0, 0), 2)
            cv2.putText(frame, predicted_emotion, (face.left(), face.top() - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)

        # Encode the frame to be displayed
        ret, jpeg = cv2.imencode('.jpg', frame)
        return jpeg.tobytes()

