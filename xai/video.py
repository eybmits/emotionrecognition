from torchvision.transforms import RandomHorizontalFlip
from torch.utils.data import WeightedRandomSampler
from sklearn.metrics import classification_report
from torchvision.transforms import RandomCrop
from torchvision.transforms import Grayscale
from torchvision.transforms import ToTensor
from torch.utils.data import random_split
from torch.utils.data import DataLoader
from emonet import config as cfg
from emonet import EarlyStopping
from emonet import LRScheduler
from torchvision import transforms
from emonet import emoNet
from torchvision import datasets
import matplotlib.pyplot as plt
from collections import Counter
from datetime import datetime
from torch.optim import SGD
import torch.nn as nn
import pandas as pd
import argparse
import torch
import math
import cv2
import numpy as np
import torch.nn.functional as F

# Definieren Sie hier die Klasse Ihres SimpleCNN-Modells
# class SimpleCNN(nn.Module):
#     ...
train_transform = transforms.Compose([
    Grayscale(num_output_channels=1),
    RandomHorizontalFlip(),
    RandomCrop((48, 48)),
    ToTensor()
])
 
test_transform = transforms.Compose([
    Grayscale(num_output_channels=1),
    ToTensor()
])


train_data = datasets.ImageFolder(cfg.TRAIN_DIRECTORY, transform=train_transform)
test_data = datasets.ImageFolder(cfg.TEST_DIRECTORY, transform=test_transform)

classes = train_data.classes
# Laden Sie Ihr trainiertes Modell
num_of_classes = len(classes)
print(f"[INFO] Class labels: {classes}")
model = emoNet(num_of_channels=1, num_of_classes=num_of_classes)

num_of_classes = len(classes)
print(f"[INFO] Class labels: {classes}")


model.load_state_dict(torch.load('/Users/markusbaumann/emotionrecognition/output/model.pth'))
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
model.eval()

# Pfad zur Haar-Kaskade für die Gesichtserkennung
face_cascade_path = '/Users/markusbaumann/emotionrecognition/emoNet/haarcascade_frontalface_default.xml'
face_cascade = cv2.CascadeClassifier(face_cascade_path)

# Starten Sie die Webcam
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    raise IOError("Cannot open webcam")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.1, 4)

    for (x, y, w, h) in faces:
        # Zeichnen Sie ein grünes Rechteck um das erkannte Gesicht
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
        cv2.putText(frame, 'Face detected', (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        # Bereiten Sie das Gesicht für das Modell vor
        face_img = gray[y:y+h, x:x+w]
        face_img = cv2.resize(face_img, (48, 48))  # Passen Sie dies an die Eingabegröße Ihres Modells an
        face_img = np.expand_dims(face_img, axis=0)
        face_img = np.expand_dims(face_img, axis=0)
        face_img = face_img / 255.0  # Normalisierung
        face_tensor = torch.from_numpy(face_img).type(torch.FloatTensor)
        face_tensor = face_tensor.to(device)

        # Modellvorhersage
        with torch.no_grad():
            output = model(face_tensor)
            probabilities = F.softmax(output, dim=1)
            predicted = torch.max(probabilities, 1)[1]
            predicted_class = classes[predicted.item()]

        # Zeigen Sie die Vorhersage auf dem Frame an
        cv2.putText(frame, predicted_class, (x, y + h + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

    cv2.imshow('Face Emotion Recognition', frame)
    if cv2.waitKey(2) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()