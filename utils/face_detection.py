import dlib

# Load dlib's pre-trained shape predictor
landmark_predictor = dlib.shape_predictor('models/shape_predictor_68_face_landmarks.dat')
face_detector = dlib.get_frontal_face_detector()

def detect_face_and_landmarks(image):
    """Detect faces and facial landmarks in the image."""
    detected_faces = face_detector(image, 1)
    landmarks_list = []

    for face in detected_faces:
        landmarks = landmark_predictor(image, face)
        landmark_points = [(landmarks.part(i).x, landmarks.part(i).y) for i in range(68)]
        landmarks_list.append(landmark_points)

    return detected_faces, landmarks_list
