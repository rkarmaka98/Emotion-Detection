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
    
    # Example: distance between key facial landmarks
    eye_distance = np.linalg.norm(np.array(landmarks[36]) - np.array(landmarks[45]))  # Distance between eyes
    mouth_width = np.linalg.norm(np.array(landmarks[48]) - np.array(landmarks[54]))   # Width of the mouth
    nose_to_mouth = np.linalg.norm(np.array(landmarks[33]) - np.array(landmarks[51]))  # Nose to mouth
    
    # Example: distances between eyebrows and eyes
    left_eyebrow_to_eye = np.linalg.norm(np.array(landmarks[21]) - np.array(landmarks[39]))  # Left eyebrow to left eye
    right_eyebrow_to_eye = np.linalg.norm(np.array(landmarks[22]) - np.array(landmarks[42]))  # Right eyebrow to right eye

    # Example: other facial distances
    chin_to_left_eye = np.linalg.norm(np.array(landmarks[8]) - np.array(landmarks[36]))  # Chin to left eye
    chin_to_right_eye = np.linalg.norm(np.array(landmarks[8]) - np.array(landmarks[45]))  # Chin to right eye
    mouth_to_left_eye = np.linalg.norm(np.array(landmarks[48]) - np.array(landmarks[36]))  # Mouth to left eye
    mouth_to_right_eye = np.linalg.norm(np.array(landmarks[54]) - np.array(landmarks[45]))  # Mouth to right eye
    nose_to_left_eye = np.linalg.norm(np.array(landmarks[33]) - np.array(landmarks[36]))  # Nose to left eye

    # Append the calculated distances to the feature vector
    features.extend([eye_distance, mouth_width, nose_to_mouth,
                     left_eyebrow_to_eye, right_eyebrow_to_eye,
                     chin_to_left_eye, chin_to_right_eye,
                     mouth_to_left_eye, mouth_to_right_eye, nose_to_left_eye])
    
    return np.array(features)



# Video stream capture class
class VideoCamera:
    def __init__(self):
        # Initialize video capture (use the webcam as the video source)
        self.video = cv2.VideoCapture(0)

    def __del__(self):
        self.video.release()

    def get_frame(self):
        ret, frame = self.video.read()

        if not ret:
            print("Failed to capture frame from video stream.")
            return None

        # Convert frame to grayscale for face detection
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Detect faces in the frame
        detected_faces = face_detector(gray, 1)

        emotion = "No Face Detected"
        for face in detected_faces:
            # Detect facial landmarks
            landmarks = landmark_predictor(gray, face)
            landmark_points = [(landmarks.part(i).x, landmarks.part(i).y) for i in range(68)]

            # Extract geometric features (10 features)
            geometric_features = extract_geometric_features(landmark_points)

            # Crop the face region and ensure it's valid
            face_image = gray[face.top():face.bottom(), face.left():face.right()]
            if face_image.size == 0:
                print("Warning: Face region is empty or invalid.")
                continue


            # Predict emotion using the pre-trained SVM classifier
            predicted_emotion = emotion_classifier.predict([geometric_features])[0]
            print(f"Predicted Emotion: {predicted_emotion}")

            emotion = predicted_emotion

            # Draw a rectangle around the face and display the emotion on the frame
            cv2.rectangle(frame, (face.left(), face.top()), (face.right(), face.bottom()), (0, 255, 0), 1)
            cv2.putText(frame, predicted_emotion, (face.left(), face.top() - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)

        # Encode the frame to be displayed
        ret, jpeg = cv2.imencode('.jpg', frame)
        return jpeg.tobytes(), emotion
