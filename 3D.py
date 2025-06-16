import os
import cv2
import json
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
from torch.utils.data import Dataset, DataLoader
import albumentations as A
from albumentations.pytorch import ToTensorV2
import matplotlib.pyplot as plt

# Пути к данным
images_folder = 'animals'
annotations_file = 'keypoints.json'

# Загружаем разметку
with open(annotations_file, 'r') as f:
    annotations = json.load(f)

# Dataset
class AnimalKeypointsDataset(Dataset):
    def __init__(self, annotations, images_folder, transform=None):
        self.samples = []
        self.transform = transform
        self.images_folder = images_folder

        for ann in annotations['annotations']:
            img_id = ann['image_id']
            filename = annotations['images'][str(img_id)]
            keypoints = ann['keypoints']
            keypoints = np.array([[x, y] for x, y, v in keypoints])
            self.samples.append((filename, keypoints))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        filename, keypoints = self.samples[idx]
        img_path = os.path.join(self.images_folder, filename)
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        h, w, _ = image.shape

        keypoints_norm = keypoints / np.array([w, h])

        if self.transform:
            augmented = self.transform(image=image, keypoints=keypoints_norm)
            image = augmented['image']
            keypoints_norm = np.array(augmented['keypoints'])

        return image, keypoints_norm.flatten()

# Аугментации и преобразования
transform = A.Compose([
    A.Resize(224, 224),
    A.HorizontalFlip(p=0.5),
    A.RandomBrightnessContrast(p=0.2),
    A.Normalize(mean=(0.485, 0.456, 0.406),
                std=(0.229, 0.224, 0.225)),
    ToTensorV2()
], keypoint_params=A.KeypointParams(format='xy', remove_invisible=False))

# Создаем dataset и dataloader
dataset = AnimalKeypointsDataset(annotations, images_folder, transform)
dataloader = DataLoader(dataset, batch_size=8, shuffle=True)

# Сеть для keypoints
class KeypointModel(nn.Module):
    def __init__(self, num_keypoints):
        super(KeypointModel, self).__init__()
        self.backbone = torchvision.models.resnet18(weights="IMAGENET1K_V1")
        self.backbone.fc = nn.Linear(self.backbone.fc.in_features, num_keypoints * 2)

    def forward(self, x):
        return self.backbone(x)

num_keypoints = dataset[0][1].shape[0] // 2
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = KeypointModel(num_keypoints).to(device)

# Обучение
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=1e-3)
epochs = 20

for epoch in range(epochs):
    model.train()
    running_loss = 0.0
    for images, keypoints in dataloader:
        images = images.to(device)
        keypoints = keypoints.to(device)

        outputs = model(images)
        loss = criterion(outputs, keypoints)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        running_loss += loss.item()

    print(f"Epoch [{epoch+1}/{epochs}], Loss: {running_loss/len(dataloader):.4f}")

# Проверка и визуализация результата
def visualize(img, keypoints):
    img = img.permute(1, 2, 0).cpu().numpy()
    img = (img * np.array([0.229, 0.224, 0.225])) + np.array([0.485, 0.456, 0.406])
    img = np.clip(img, 0, 1)
    h, w = 224, 224
    keypoints = keypoints.reshape(-1, 2) * np.array([w, h])

    plt.imshow(img)
    for (x, y) in keypoints:
        plt.scatter(x, y, c='r')
    plt.show()

model.eval()
sample_img, _ = dataset[0]
sample_input = sample_img.unsqueeze(0).to(device)
with torch.no_grad():
    pred = model(sample_input).cpu().numpy()[0]
visualize(sample_img, pred)