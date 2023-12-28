import os
import torch
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
import argparse
import csv
from emonet import emoNet  # Importing the emoNet class from emonet.py

# Parse command line arguments
parser = argparse.ArgumentParser(description='Classify images and output scores to a CSV file.')
parser.add_argument('folder_path', type=str, help='Path to the folder containing images')
parser.add_argument('model_path', type=str, help='Path to the trained model file')
args = parser.parse_args()

# Define transformations for the image
data_transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.Resize((48, 48)),
    transforms.ToTensor()
])

# Load the trained model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = emoNet(num_of_channels=1, num_of_classes=6)
model.load_state_dict(torch.load(args.model_path, map_location=device))
model.to(device)
model.eval()

# Emotion label mapping and its inverse
label_mapping = {
    'anger': 0, 'disgust': 1, 'fear': 2, 
    'happiness': 3, 'sadness': 4, 'surprise': 5
}
inverse_label_mapping = {v: k for k, v in label_mapping.items()}

# List to hold all rows for the CSV
csv_rows = [['filepath', 'anger', 'disgust', 'fear', 'happiness', 'sadness', 'surprise', 'predicted_label']]

# Process images
for image_name in os.listdir(args.folder_path):
    if image_name.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff')):
        image_path = os.path.join(args.folder_path, image_name)
        image = Image.open(image_path).convert('L')
        image = data_transform(image)
        image = image.unsqueeze(0).to(device)

        with torch.no_grad():
            output = model(image)
            probabilities = F.softmax(output, dim=1)
            predicted = torch.max(probabilities, 1)[1]
            predicted_class = inverse_label_mapping[predicted.item()]

        rounded_probabilities = [round(prob, 2) for prob in probabilities[0].cpu().numpy()]
        csv_rows.append([image_path] + rounded_probabilities + [predicted_class])

csv_file_path = os.path.join(args.folder_path, 'classification_scores.csv')
with open(csv_file_path, 'w', newline='') as file:
    writer = csv.writer(file)
    writer.writerows(csv_rows)

print(f"Classification scores have been saved to {csv_file_path}")
