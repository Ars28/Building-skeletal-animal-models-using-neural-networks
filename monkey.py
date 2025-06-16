import os
import json
import numpy as np
import cv2
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import torchvision.models as models
from sklearn.model_selection import train_test_split

json_path = 'keypoints.json'
images_root = 'animals'

with open(json_path, 'r') as f:
    data = json.load(f)

id_to_filename = {int(k): v for k, v in data['images'].items()}

samples = []
for annotation in data['annotations']:
    img_id = annotation['image_id']
    if img_id not in id_to_filename:
        continue
    filename = id_to_filename[img_id]
    keypoints = annotation['keypoints']
    
    xy_keypoints = []
    for kp in keypoints:
        x, y, v = kp
        xy_keypoints.extend([x, y])
    
    samples.append((os.path.join(images_root, filename), xy_keypoints))

train_samples, test_samples = train_test_split(samples, test_size=0.2, random_state=42)

class AnimalPoseDataset(Dataset):
    def __init__(self, samples, transform=None):
        self.samples = samples
        self.transform = transform

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, keypoints = self.samples[idx]
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, (224, 224))
        
        keypoints = np.array(keypoints).astype(np.float32)
        keypoints = keypoints.reshape(-1, 2)
        keypoints = keypoints.flatten()

        if self.transform:
            image = self.transform(image)

        return image, torch.tensor(keypoints)

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

train_dataset = AnimalPoseDataset(train_samples, transform=transform)
test_dataset = AnimalPoseDataset(test_samples, transform=transform)
train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=8, shuffle=False)

class PoseResNet(nn.Module):
    def __init__(self, num_keypoints):
        super(PoseResNet, self).__init__()
        self.backbone = models.resnet18(pretrained=True)
        self.backbone.fc = nn.Linear(self.backbone.fc.in_features, num_keypoints * 2)

    def forward(self, x):
        return self.backbone(x)

num_keypoints = len(train_samples[0][1]) // 2
model = PoseResNet(num_keypoints)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

epochs = 10
for epoch in range(epochs):
    model.train()
    running_loss = 0.0
    for images, keypoints in train_loader:
        images = images.to(device)
        keypoints = keypoints.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, keypoints)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()

    print(f"Epoch [{epoch+1}/{epochs}], Loss: {running_loss/len(train_loader):.4f}")

torch.save(model.state_dict(), "animal_pose_model.pth")