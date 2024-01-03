import cv2
import numpy as np
from skimage.feature import local_binary_pattern
import torch
import torch.optim as optim
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import os
from skimage.feature import local_binary_pattern
from skimage.feature import hog
import numpy as np
import cv2
import torch
from skimage.filters import gabor


def extract_features(image_path):
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise ValueError(f"Image at path {image_path} could not be loaded.")

    # LBP Features
    lbp = local_binary_pattern(image, 24, 8, method="uniform")
    lbp_hist, _ = np.histogram(lbp.ravel(), bins=np.arange(0, 24 + 3), range=(0, 24 + 2))
    lbp_hist = lbp_hist.astype("float")
    lbp_hist /= (lbp_hist.sum() + 1e-7)

    # HOG Features
    hog_features, _ = hog(image, orientations=8, pixels_per_cell=(16, 16),
                          cells_per_block=(1, 1), visualize=True, feature_vector=True)

    # Gabor Features
    gabor_feats = []
    for theta in [0, np.pi / 4, np.pi / 2, 3 * np.pi / 4]:
        gabor_filter, _ = gabor(image, frequency=0.6, theta=theta)
        gabor_feats.extend(np.ravel(gabor_filter))
    
    # Combine all features into one vector
    combined_features = np.hstack((lbp_hist, hog_features, gabor_feats))

    # Normalize the feature vector
    normalized_features = (combined_features - np.mean(combined_features)) / (np.std(combined_features) + 1e-7)

    return torch.tensor(normalized_features, dtype=torch.float32)



from torch.utils.data import Dataset, DataLoader
import os

class LBPDataset(Dataset):
    def __init__(self, directory, classes):
        self.data = []
        self.labels = []
        self.class_to_idx = {cls: idx for idx, cls in enumerate(classes)}

        for cls in classes:
            path = os.path.join(directory, cls)
            for img_file in os.listdir(path):
                if not img_file.lower().endswith(('.png', '.jpg', '.jpeg')):
                    continue

                img_path = os.path.join(path, img_file)
                try:
                    print(f"Extracting features from {img_path}")  # Debug print
                    feature = extract_features(img_path)

                    # Read image and convert it to a tensor
                    image = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
                    image_tensor = torch.tensor(image, dtype=torch.float32)
                    image_tensor = image_tensor.unsqueeze(0)  # Add channel dimension

                    self.data.append((image_tensor, feature))
                    self.labels.append(self.class_to_idx[cls])
                except Exception as e:
                    print(f"Error processing {img_path}: {e}")

        print(f"Dataset initialized with {len(self.data)} samples")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx][0], self.data[idx][1], self.labels[idx]




class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3)  # 1 input channel (grayscale), 32 output channels, 3x3 kernel
        self.conv2 = nn.Conv2d(32, 64, 3)
        self.fc1 = nn.Linear(64 * 10 * 10, 128)  # Adjust the dimensions according to your CNN architecture

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2)
        x = x.view(-1, 64 * 10 * 10)  # Flatten the output for the fully connected layer
        x = F.relu(self.fc1(x))
        return x


class EmotionClassifier(nn.Module):
    def __init__(self, feature_input_size, num_classes):
        super(EmotionClassifier, self).__init__()
        self.cnn = SimpleCNN()
        self.fc1 = nn.Linear(feature_input_size + 128, 128)  # Concatenated vector size
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, num_classes)

    def forward(self, image, features):
        image_output = self.cnn(image)
        combined = torch.cat((image_output, features), 1)
        x = F.relu(self.fc1(combined))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x



import torch.optim as optim
import torch

# Assuming `data` folder with subfolders for each class
classes = ['angry', 'disgust', 'fear', 'happy', 'sad', 'surprise']
input_size = 9314  # Updated input size for 48x48 images
num_classes = len(classes)

# Initialize datasets and dataloaders
train_dataset = LBPDataset('/Users/markusbaumann/emotionrecognition/data/train', classes)
test_dataset = LBPDataset('/Users/markusbaumann/emotionrecognition/data/test', classes)
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=True)

# Initialize the model
model = EmotionClassifier(input_size, num_classes)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training Loop


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Training Loop
for epoch in range(100):  # number of epochs
    for raw_images, features, targets in train_loader:
        optimizer.zero_grad()

        # Move tensors to the same device as the model
        raw_images = raw_images.to(device)
        features = features.to(device)
        targets = targets.to(device)

        # Forward pass
        output = model(raw_images, features)
        loss = criterion(output, targets)

        # Backward and optimize
        loss.backward()
        optimizer.step()


# Testing Loop
# Testing Loop
correct = 0
total = 0
with torch.no_grad():
    for raw_images, features, targets in test_loader:
        raw_images = raw_images.to(device)
        features = features.to(device)
        targets = targets.to(device)

        outputs = model(raw_images, features)
        _, predicted = torch.max(outputs.data, 1)
        total += targets.size(0)
        correct += (predicted == targets).sum().item()

print(f'Accuracy of the network on test images: {100 * correct / total}%')
