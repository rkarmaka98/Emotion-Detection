import os
import cv2
import dlib
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
import joblib

# Load dlib's pre-trained shape predictor (make sure the file is in the right path)
landmark_predictor = dlib.shape_predictor('models/shape_predictor_68_face_landmarks.dat')
face_detector = dlib.get_frontal_face_detector()

def extract_geometric_features(landmarks):
    """Extract geometric features from facial landmarks."""
    features = []
    
    # Example: distance between eyes, width of mouth, etc.
    eye_distance = np.linalg.norm(np.array(landmarks[36]) - np.array(landmarks[45]))
    mouth_width = np.linalg.norm(np.array(landmarks[48]) - np.array(landmarks[54]))
    features.append(eye_distance)
    features.append(mouth_width)
    
    return np.array(features)

# Define the path to your dataset
dataset_dir = 'data/FER2013/train/'  # Assuming you have organized your dataset into 'train' and 'test'

# Prepare dataset: extract image data and labels
X = []
y = []

# Loop through each emotion directory (angry, happy, etc.)
for emotion in os.listdir(dataset_dir):
    emotion_dir = os.path.join(dataset_dir, emotion)
    
    if os.path.isdir(emotion_dir):
        # Process each image in the emotion directory
        for img_file in os.listdir(emotion_dir):
            img_path = os.path.join(emotion_dir, img_file)
            image = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)  # Load as grayscale
            
            # Detect faces and landmarks in the image
            detected_faces = face_detector(image, 1)
            
            for face in detected_faces:
                landmarks = landmark_predictor(image, face)
                landmark_points = [(landmarks.part(i).x, landmarks.part(i).y) for i in range(68)]
                
                # Extract geometric features (e.g., distance between landmarks)
                features = extract_geometric_features(landmark_points)
                X.append(features)
                y.append(emotion)  # Emotion is the directory name (e.g., 'happy', 'angry')

# Convert to NumPy arrays
X = np.array(X)
y = np.array(y)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train an SVM classifier
emotion_classifier = SVC(kernel='linear', probability=True)
emotion_classifier.fit(X_train, y_train)

# Save the trained model
joblib.dump(emotion_classifier, 'emotion_classifier.pkl')
print("Model saved as 'emotion_classifier.pkl'")

# Optional: Evaluate the model
accuracy = emotion_classifier.score(X_test, y_test)
print(f'Model accuracy: {accuracy * 100:.2f}%')


