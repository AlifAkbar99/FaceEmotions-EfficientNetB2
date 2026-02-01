import cv2
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.efficientnet import preprocess_input
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

# Load emotion model
model = load_model("EfficientNetB2_Best.h5")
emotion_labels = ['Anger', 'Disgust', 'Fear', 'Happiness', 'Neutral', 'Sadness', 'Surprise']

# Face Landmarker setup
base_options = python.BaseOptions(model_asset_path='face_landmarker.task')
options = vision.FaceLandmarkerOptions(
    base_options=base_options,
    output_face_blendshapes=False,
    output_facial_transformation_matrixes=False,
    num_faces=1
)
detector = vision.FaceLandmarker.create_from_options(options)

# Panel settings
PANEL_WIDTH = 250
TEXT_COLOR = (255, 255, 255)

# Face mesh connections (simplified - main contours)
FACE_OVAL = [10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288, 397, 365, 379, 378, 400, 377, 152, 148, 176, 149, 150, 136, 172, 58, 132, 93, 234, 127, 162, 21, 54, 103, 67, 109, 10]
LEFT_EYE = [362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385, 384, 398, 362]
RIGHT_EYE = [33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246, 33]
LIPS_OUTER = [61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291, 308, 324, 318, 402, 317, 14, 87, 178, 88, 95, 78, 61]
LIPS_INNER = [78, 95, 88, 178, 87, 14, 317, 402, 318, 324, 308, 291, 375, 321, 405, 314, 17, 84, 181, 91, 146, 61, 78]
LEFT_EYEBROW = [276, 283, 282, 295, 285, 300, 293, 334, 296, 336]
RIGHT_EYEBROW = [46, 53, 52, 65, 55, 70, 63, 105, 66, 107]
NOSE = [168, 6, 197, 195, 5, 4, 1, 19, 94, 2]

def draw_face_mesh(image, landmarks, w, h):
    """Draw face mesh manually"""
    
    def draw_line(indices, color=(200, 200, 200), thickness=1):
        points = []
        for idx in indices:
            if idx < len(landmarks):
                x = int(landmarks[idx].x * w)
                y = int(landmarks[idx].y * h)
                points.append((x, y))
        for i in range(len(points) - 1):
            cv2.line(image, points[i], points[i+1], color, thickness)
    
    def draw_points(indices, color=(0, 0, 255), radius=1):
        for idx in indices:
            if idx < len(landmarks):
                x = int(landmarks[idx].x * w)
                y = int(landmarks[idx].y * h)
                cv2.circle(image, (x, y), radius, color, -1)
    
    # Draw contours
    draw_line(FACE_OVAL, (180, 180, 180), 1)
    draw_line(LEFT_EYE, (0, 255, 255), 1)
    draw_line(RIGHT_EYE, (0, 255, 255), 1)
    draw_line(LIPS_OUTER, (180, 180, 180), 1)
    draw_line(LIPS_INNER, (100, 100, 100), 1)
    draw_line(LEFT_EYEBROW, (180, 180, 180), 1)
    draw_line(RIGHT_EYEBROW, (180, 180, 180), 1)
    draw_line(NOSE, (180, 180, 180), 1)
    
    # Draw key points (eyes, nose, mouth)
    eye_points = list(range(33, 42)) + list(range(133, 142)) + list(range(362, 374)) + list(range(263, 272))
    draw_points(eye_points, (0, 0, 255), 2)
    
    # Nose points
    nose_points = [1, 2, 4, 5, 6, 19, 94, 195, 197]
    draw_points(nose_points, (0, 0, 255), 2)
    
    # Mouth points  
    mouth_points = [61, 78, 95, 88, 178, 87, 14, 317, 402, 318, 324, 308, 291, 375, 321, 405, 314, 17, 84, 181, 91, 146]
    draw_points(mouth_points, (0, 0, 255), 2)
    
    return image

def get_face_bbox(landmarks, w, h, padding=20):
    """Get bounding box from landmarks"""
    x_coords = [lm.x * w for lm in landmarks]
    y_coords = [lm.y * h for lm in landmarks]
    
    x_min = max(0, int(min(x_coords)) - padding)
    y_min = max(0, int(min(y_coords)) - padding)
    x_max = min(w, int(max(x_coords)) + padding)
    y_max = min(h, int(max(y_coords)) + padding)
    
    return x_min, y_min, x_max, y_max

def create_panel(emotions_pred, face_status, panel_height):
    """Create info panel"""
    panel = np.zeros((panel_height, PANEL_WIDTH, 3), dtype=np.uint8)
    
    y_pos = 40
    
    # Emotion percentages
    for i, label in enumerate(emotion_labels):
        percentage = int(emotions_pred[i] * 100)
        text = f"{label}"
        percent_text = f"{percentage} %"
        
        if i == np.argmax(emotions_pred) and percentage > 0:
            color = (0, 255, 255)
        else:
            color = TEXT_COLOR
        
        cv2.putText(panel, text, (20, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        cv2.putText(panel, percent_text, (180, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        y_pos += 35
    
    # Status section
    y_pos += 20
    cv2.putText(panel, "Status:", (20, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.6, TEXT_COLOR, 2)
    y_pos += 30
    cv2.putText(panel, "* Source: Webcam", (20, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.5, TEXT_COLOR, 1)
    y_pos += 25
    cv2.putText(panel, "* Player: Playing", (20, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.5, TEXT_COLOR, 1)
    y_pos += 25
    cv2.putText(panel, f"* Face: {face_status}", (20, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.5, TEXT_COLOR, 1)
    y_pos += 25
    cv2.putText(panel, "* Markers: Scale to face", (20, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.5, TEXT_COLOR, 1)
    
    # Hint
    y_pos += 40
    cv2.putText(panel, "Hint:", (20, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.6, TEXT_COLOR, 2)
    y_pos += 25
    cv2.putText(panel, "Press 'q' to quit", (20, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.5, TEXT_COLOR, 1)
    
    return panel

cam = cv2.VideoCapture(0)

while True:
    ret, frame = cam.read()
    if not ret:
        break
    
    frame = cv2.flip(frame, 1)
    h, w, _ = frame.shape
    
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
    
    detection_result = detector.detect(mp_image)
    
    face_status = "Not Found"
    emotions_pred = np.zeros(len(emotion_labels))
    
    if detection_result.face_landmarks:
        face_status = "Found"
        landmarks = detection_result.face_landmarks[0]
        
        # Draw mesh
        frame = draw_face_mesh(frame, landmarks, w, h)
        
        # Get bbox and predict emotion
        x_min, y_min, x_max, y_max = get_face_bbox(landmarks, w, h)
        face_roi = frame[y_min:y_max, x_min:x_max]
        
        if face_roi.size > 0:
            face_resized = cv2.resize(face_roi, (224, 224))
            face_rgb = cv2.cvtColor(face_resized, cv2.COLOR_BGR2RGB)
            face_input = np.expand_dims(face_rgb, axis=0)
            face_input = preprocess_input(face_input)
            
            emotions_pred = model.predict(face_input, verbose=0)[0]
    
    # Create and combine panel
    panel = create_panel(emotions_pred, face_status, h)
    combined = np.hstack((frame, panel))
    
    cv2.imshow("Face Emotion Detection", combined)
    
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cam.release()
cv2.destroyAllWindows()