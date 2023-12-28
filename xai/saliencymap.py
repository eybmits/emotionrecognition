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
parser.add_argument("-o", "--output", type=str, required=True, help="path to the output video file")
args = vars(parser.parse_args())

# Load the face detector model (Haar Cascade)
print("[INFO] loading face detector model...")
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Load the emotion detection model
print("[INFO] loading emotion detection model...")
device = "cuda" if torch.cuda.is_available() else "cpu"
emotion_dict = {0: "Angry", 1: "Fearful", 2: "Happy", 3: "Neutral", 4: "Sad", 5: "Surprised"}
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

# Initialize the video writer
frame_width = int(vs.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(vs.get(cv2.CAP_PROP_FRAME_HEIGHT))
frame_rate = int(vs.get(cv2.CAP_PROP_FPS))
fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Use 'mp4v' codec
out = cv2.VideoWriter(args['output'], fourcc, frame_rate, (frame_width, frame_height))

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
        face.requires_grad = True

        # Perform emotion detection
        predictions = model(face)
        prob = nnf.softmax(predictions, dim=1)
        top_p, top_class = prob.topk(1, dim=1)
        emotion = emotion_dict[top_class.item()]

        # Generate saliency map
        model.zero_grad()
        top_class_prob = predictions[0, top_class.item()]
        top_class_prob.backward()
        saliency, _ = torch.max(face.grad.data.abs(), dim=1)
        saliency = saliency.reshape(48, 48).cpu().numpy()
        saliency = cv2.resize(saliency, (w, h))
        saliency = np.clip(saliency * 255, 0, 255).astype(np.uint8)

        # Apply Gaussian blur to the saliency map
        saliency = cv2.GaussianBlur(saliency, (11, 11), 0)

        heatmap = cv2.applyColorMap(saliency, cv2.COLORMAP_JET)

        # Overlay the saliency map on the original color face
        color_face = cv2.cvtColor(face_roi, cv2.COLOR_GRAY2BGR)
        overlayed_image = cv2.addWeighted(color_face, 0.5, heatmap, 0.5, 0)

        # Insert the overlayed image back into the frame
        frame[y:y+h, x:x+w] = overlayed_image

        # Draw the face box and label on the frame
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
        text = f"{emotion}: {top_p.item() * 100:.2f}%"
        cv2.putText(frame, text, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    # Write the frame to the output video
    out.write(frame)

    # Display the output frame
    cv2.imshow("Frame with Saliency Map", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Clean up
cv2.destroyAllWindows()
vs.release()
out.release()  # Release the video writer
