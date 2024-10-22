import os
import cv2
import dlib
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
import joblib

# Load dlib's pre-trained shape predictor
print("Loading dlib's shape predictor and face detector...")
landmark_predictor = dlib.shape_predictor('models/shape_predictor_68_face_landmarks.dat')
face_detector = dlib.get_frontal_face_detector()
print("Shape predictor and face detector loaded successfully.")

# Function to extract geometric features
def extract_geometric_features(landmarks):
    """Extract geometric features from facial landmarks."""
    print("Extracting geometric features from landmarks...")
    features = []
    
    # Example: distance between eyes, width of mouth, etc.
    eye_distance = np.linalg.norm(np.array(landmarks[36]) - np.array(landmarks[45]))
    mouth_width = np.linalg.norm(np.array(landmarks[48]) - np.array(landmarks[54]))
    
    features.append(eye_distance)
    features.append(mouth_width)
    
    print(f"Features extracted: eye_distance={eye_distance}, mouth_width={mouth_width}")
    return np.array(features)

# Define the path to your dataset
dataset_dir = 'data/FER2013/train/'  # Assuming you have organized your dataset into 'train' and 'test'

# Prepare dataset: extract image data and labels
X = []
y = []

print(f"Processing dataset at {dataset_dir}...")
# Loop through each emotion directory (angry, happy, etc.)
for emotion in os.listdir(dataset_dir):
    emotion_dir = os.path.join(dataset_dir, emotion)
    
    if os.path.isdir(emotion_dir):
        print(f"Processing emotion: {emotion}")
        # Process each image in the emotion directory
        for img_file in os.listdir(emotion_dir):
            img_path = os.path.join(emotion_dir, img_file)
            print(f"Processing image: {img_file}")
            
            # Load image as grayscale
            image = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            if image is None:
                print(f"Warning: Failed to load image {img_file}")
                continue
            
            # Detect faces and landmarks in the image
            detected_faces = face_detector(image, 1)
            
            if len(detected_faces) == 0:
                print(f"No face detected in {img_file}")
                continue
            
            for face in detected_faces:
                landmarks = landmark_predictor(image, face)
                landmark_points = [(landmarks.part(i).x, landmarks.part(i).y) for i in range(68)]
                
                # Extract geometric features (e.g., distance between landmarks)
                features = extract_geometric_features(landmark_points)
                X.append(features)
                y.append(emotion)  # Emotion is the directory name (e.g., 'happy', 'angry')

print(f"Finished processing dataset. Total images processed: {len(X)}")

# Convert to NumPy arrays
X = np.array(X)
y = np.array(y)

# Split the data into training and testing sets
print("Splitting data into training and testing sets...")
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print(f"Training set size: {len(X_train)}, Test set size: {len(X_test)}")

# Train an SVM classifier
print("Training the SVM classifier...")
emotion_classifier = SVC(kernel='linear', probability=True)
emotion_classifier.fit(X_train, y_train)
print("Model training completed.")

# Save the trained model
print("Saving the trained model to 'emotion_classifier.pkl'...")
joblib.dump(emotion_classifier, 'models/emotion_classifier.pkl')
print("Model saved successfully as 'emotion_classifier.pkl'.")

# Optional: Evaluate the model
print("Evaluating the model...")
accuracy = emotion_classifier.score(X_test, y_test)
print(f'Model accuracy: {accuracy * 100:.2f}%')
