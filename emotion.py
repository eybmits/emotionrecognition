import argparse
import cv2
import torch
import numpy as np
from torchvision import transforms
from emonet import emoNet
import torch.nn.functional as nnf

# Initialize the argument parser
parser = argparse.ArgumentParser()
parser.add_argument("-i", "--video", type=str, required=True, help="path to the video file/webcam")
parser.add_argument("-m", "--model", type=str, required=True, help="path to the trained model")
parser.add_argument("-conf", "--confidence", type=float, default=0.5, help="minimum probability to filter out weak detections")
args = vars(parser.parse_args())

# Load the face detector model (Haar Cascade)
print("[INFO] loading face detector model...")
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Load the emotion detection model
print("[INFO] loading emotion detection model...")
device = "cuda" if torch.cuda.is_available() else "cpu"
emotion_dict = {0: "Angry", 1: "disgust", 2: "fear", 3: "happy", 4: "sad", 5: "surprise"}
model = emoNet(num_of_channels=1, num_of_classes=len(emotion_dict))
model.load_state_dict(torch.load(args["model"], map_location=device))
model.to(device)
model.eval()
# Preprocessing transformations
data_transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Grayscale(num_output_channels=1),
    transforms.Resize((48, 48)),
    transforms.ToTensor()
])

# Initialize the video stream
vs = cv2.VideoCapture(args['video'])

while True:
    # Read the next frame from the input stream
    grabbed, frame = vs.read()
    if not grabbed:
        break

    # Convert frame to grayscale and detect faces
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30), flags=cv2.CASCADE_SCALE_IMAGE)

    for (x, y, w, h) in faces:
        # Extract the face ROI, apply data transforms, and prepare for model input
        face_roi = gray[y:y+h, x:x+w]
        face = data_transform(face_roi)
        face = face.unsqueeze(0).to(device)

        # Perform emotion detection
        predictions = model(face)
        prob = nnf.softmax(predictions, dim=1)
        top_p, top_class = prob.topk(1, dim=1)
        emotion = emotion_dict[top_class.item()]

        # Draw the face box and label on the frame
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
        text = f"{emotion}: {top_p.item() * 100:.2f}%"
        cv2.putText(frame, text, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    # Display the output frame
    cv2.imshow("Frame", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Clean up
cv2.destroyAllWindows()
vs.release()
