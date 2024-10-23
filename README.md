# Real-Time Emotion Detection System

This project is a real-time emotion detection system that uses **Support Vector Classifier (SVC)** to classify human emotions based on facial landmarks. The system captures live video using a webcam, detects faces using `dlib`, extracts geometric features, and predicts the user's emotion. The results are displayed in real-time on a web interface.

## Features
- **Real-time emotion detection**: Detects emotions from live video feed.
- **SVC-based classifier**: Uses a Support Vector Classifier (SVC) to classify emotions.
- **Facial landmark detection**: Uses `dlib` for detecting key facial landmarks.
- **Geometric feature extraction**: Extracts geometric distances between facial landmarks to form feature vectors.
- **Web interface**: Displays the video feed with detected faces and predicted emotions using Flask.
- **Live emotion updates**: The system dynamically updates the detected emotion and displays it on the web interface with animations.

## Demo

https://github.com/user-attachments/assets/75b7386b-3335-4236-9e52-53ff4f92a3cb

## Installation

### Requirements
- Python 3.x
- OpenCV
- Flask
- Dlib
- Scikit-learn
- Joblib

### Steps to Run the Project Locally

1. **Clone the repository**
    ```
    git clone https://github.com/your-username/emotion-detection.git
    cd emotion-detection
    ```
2. **Create and activate a virtual environment**
    ```
    python -m venv env
    source env/bin/activate   # On Windows: env\Scripts\activate
    ```
3. **Install the required Python packages**
    ```
    pip install -r requirements.txt
    
    ```
