# /path/to/full_script.py
import os
import torch
from PIL import Image, ImageOps
from torchvision import transforms, datasets
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
from torch.optim import SGD
from torchvision.transforms.functional import to_tensor, resize
from emonet import config as cfg, EarlyStopping, LRScheduler, emoNet

# Function to normalize image saturation and brightness
def normalize_saturation_and_brightness(image):
    image_hsv = image.convert('HSV')
    h, s, v = image_hsv.split()
    s = ImageOps.autocontrast(s)
    v = ImageOps.autocontrast(v)
    image_normalized = Image.merge('HSV', (h, s, v)).convert('RGB')
    return image_normalized


# Custom Image Dataset
class CustomImageDataset(Dataset):
    def __init__(self, img_dir, transform=None, label_mapping=None):
        self.img_dir = img_dir
        self.transform = transform
        self.images = [file for file in sorted(os.listdir(img_dir)) if file.endswith(('.png', '.jpg', '.jpeg'))]
        self.label_mapping = label_mapping

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_name = os.path.join(self.img_dir, self.images[idx])
        image = Image.open(img_name)
        image = normalize_saturation_and_brightness(image)
        label_text = self.extract_label(self.images[idx])
        label = self.label_mapping.get(label_text, -1)

        if self.transform:
            image = self.transform(image)

        return image, label

    @staticmethod
    def extract_label(file_name):
        return file_name.split('_')[-1].split('.')[0]

# Label mapping and transform placeholders
label_mapping = {
    'anger': 0, 'disgust': 1, 'fear': 2,
    'happiness': 3, 'sadness': 4, 'surprise': 5
}

# Set directory for validation set
directory = '/Users/markusbaumann/emotionrecognition/data/validation_set'

# Calculate mean and std for normalization
transform_to_tensor = transforms.Compose([transforms.ToTensor()])

# Define transformations with the calculated mean and std
val_transforms = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.Resize((48, 48)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5071], std=[0.2370])
])

# Instantiate dataset and dataloader for validation set
val_dataset = CustomImageDataset(img_dir=directory, transform=val_transforms, label_mapping=label_mapping)
val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)

# Load the model and prepare for evaluation
model = emoNet(num_of_channels=1, num_of_classes=len(label_mapping))
model.load_state_dict(torch.load('/Users/markusbaumann/emotionrecognition/output/model.pth'))
model.eval()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Evaluate the model on the validation set
correct = 0
total = 0
with torch.no_grad():
    for images, labels in val_loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

# Calculate and print accuracy
accuracy = 100 * correct / total
print(f'Validation Accuracy: {accuracy:.2f}%')
