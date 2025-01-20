import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
import cv2
from keras.models import model_from_json
import numpy as np

# Load model from JSON file and weights
json_file = open("facialemotionmodel.json", "r")
model_json = json_file.read()
json_file.close()
model = model_from_json(model_json)
model.load_weights("facialemotionmodel.h5")

# Load Haar Cascade for face detection
haar_file = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
face_cascade = cv2.CascadeClassifier(haar_file)

# Function to preprocess input image
def extract_features(image):
    feature = np.array(image)
    feature = feature.reshape(1, 48, 48, 1)
    return feature / 255.0

# Initialize webcam
webcam = cv2.VideoCapture(0)
labels = {0: 'angry', 1: 'disgust', 2: 'fear', 3: 'happy', 4: 'neutral', 5: 'sad', 6: 'surprise'}

try:
    while True:
        ret, frame = webcam.read()
        if not ret:
            print("Failed to grab frame.")
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)

        for (x, y, w, h) in faces:
            face = gray[y:y+h, x:x+w]
            cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
            face_resized = cv2.resize(face, (48, 48))
            img = extract_features(face_resized)
            pred = model.predict(img)
            prediction_label = labels[pred.argmax()]
            cv2.putText(frame, prediction_label, (x, y-10), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (0, 0, 255), 2)

        cv2.imshow("Emotion Detection", frame)

        # Check for 'q' key press to exit
        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("Exiting...")
            break

finally:
    # Release resources
    webcam.release()
    cv2.destroyAllWindows()
