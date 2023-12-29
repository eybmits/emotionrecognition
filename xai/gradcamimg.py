import argparse
import cv2
import torch
import numpy as np
from torchvision import transforms
from emonet import emoNet  # Make sure this matches the actual class name in your emonet module
import torch.nn.functional as nnf
from gradcam import GradCAM

# Initialize the argument parser
parser = argparse.ArgumentParser()
parser.add_argument("-i", "--image", type=str, required=True, help="path to the image file")
parser.add_argument("-m", "--model", type=str, required=True, help="path to the trained model")
parser.add_argument("-o", "--output", type=str, required=True, help="path to the output image file")
args = vars(parser.parse_args())

# Load the face detector model (Haar Cascade)
print("[INFO] loading face detector model...")
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Load the emotion detection model
print("[INFO] loading emotion detection model...")
device = "cuda" if torch.cuda.is_available() else "cpu"
model = emoNet(num_of_channels=1, num_of_classes=6)  # Initialize the model with 6 classes
model.load_state_dict(torch.load(args["model"], map_location=device))
model.to(device)
model.eval()

# Initialize Grad-CAM
target_layer = model.features[-1]  # Adjust the layer for more detailed heatmap
grad_cam = GradCAM(model, target_layer)

# Preprocessing transformations
data_transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Grayscale(num_output_channels=1),
    transforms.Resize((48, 48)),  # Resize to the size the model was trained on
    transforms.ToTensor()
])

# Emotion label mapping and its inverse
label_mapping = {
    'anger': 0, 'disgust': 1, 'fear': 2,
    'happiness': 3, 'sadness': 4, 'surprise': 5
}
inverse_label_mapping = {v: k for k, v in label_mapping.items()}

# Load the image
image = cv2.imread(args['image'])
rgb_img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# Convert image to grayscale and detect faces
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30), flags=cv2.CASCADE_SCALE_IMAGE)

for (x, y, w, h) in faces:
    # Extract the face ROI, apply data transforms, and prepare for model input
    face_roi = gray[y:y+h, x:x+w]
    face_roi_resized = cv2.resize(face_roi, (48, 48))  # Resize for the model
    face = data_transform(face_roi_resized)
    face = face.unsqueeze(0).to(device)

    # Perform emotion detection
    with torch.no_grad():
        output = model(face)
        probabilities = nnf.softmax(output, dim=1)
        predicted = torch.max(probabilities, 1)[1]
        predicted_emotion = inverse_label_mapping[predicted.item()]

    # Extract the original face region used for overlay
    original_face_region = rgb_img[y:y+h, x:x+w]

    # Generate Grad-CAM heatmap and apply it to the original face region
    cam_image = grad_cam(face, predicted.item(), original_face_region)  # Use the predicted emotion class

    # Insert the Grad-CAM result back into the original image
    image[y:y+h, x:x+w] = cam_image

    # Draw the face box and label on the image
    cv2.rectangle(image, (x, y), (x+w, y+h), (0, 255, 0), 2)
    cv2.putText(image, predicted_emotion, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

# Create a legend box on the right
legend_x = image.shape[1] + 10
legend_y = 20
for emotion, prob in zip(label_mapping.keys(), probabilities[0]):
    text = f"{emotion}: {prob * 100:.2f}%"
    cv2.putText(image, text, (legend_x, legend_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
    legend_y += 20

# Save the output image
cv2.imwrite(args['output'], image)

# Optionally, display the output image
cv2.imshow("Image with Enhanced Grad-CAM", image)
cv2.waitKey(0)
cv2.destroyAllWindows()
