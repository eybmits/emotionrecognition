import os
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
from torchvision import transforms, datasets
from torch.utils.data import Dataset, DataLoader, random_split, WeightedRandomSampler
from sklearn.metrics import classification_report
from collections import Counter
from datetime import datetime
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.optim import SGD

# Import custom modules
from emonet import config as cfg, EarlyStopping, LRScheduler, emoNet

# Custom Image Dataset Definition
class CustomImageDataset(Dataset):
    def __init__(self, img_dir, transform=None, label_mapping=None):
        self.img_dir = img_dir
        self.transform = transform
        self.images = sorted(os.listdir(img_dir))  # Sort image filenames
        self.label_mapping = label_mapping

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_name = os.path.join(self.img_dir, self.images[idx])
        image = Image.open(img_name).convert('L')
        label_text = self.extract_label(self.images[idx])
        label = self.label_mapping.get(label_text, -1)

        if self.transform:
            image = self.transform(image)

        return image, label

    @staticmethod
    def extract_label(file_name):
        return file_name.split('_')[-1].split('.')[0]

# Label mapping for the validation set
label_mapping = {
    'anger': 0, 'disgust': 1, 'fear': 2, 
    'happiness': 3, 'sadness': 4, 'surprise': 5
}

# Transformations
train_transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.RandomHorizontalFlip(),
    transforms.RandomCrop((48, 48)),
    transforms.ToTensor()
])

test_transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.ToTensor()
])

val_transforms = transforms.Compose([
    transforms.Resize((48, 48)),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

# Datasets and Data Loaders
train_data = datasets.ImageFolder(cfg.TRAIN_DIRECTORY, transform=train_transform)
test_data = datasets.ImageFolder(cfg.TEST_DIRECTORY, transform=test_transform)
val_dataset = CustomImageDataset(img_dir='/Users/markusbaumann/Documents/CS/computervision/Project/emotions/validation_set', transform=val_transforms, label_mapping=label_mapping)

val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)

# Model loading and evaluation
model = emoNet(num_of_channels=1, num_of_classes=len(train_data.classes))
model.load_state_dict(torch.load('/Users/markusbaumann/emotionrecognition/output/model.pth'))
model.eval()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Model Evaluation on Validation Set
correct = 0
total = 0

with torch.no_grad():
    for images, labels in val_loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

accuracy = 100 * correct / total
print(f'Validation Accuracy: {accuracy:.2f}%')
