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
ANNOTATIONS = {
    'train': 'ap-10k/annotations/ap10k-train-split1.json',
    'val': 'ap-10k/annotations/ap10k-val-split1.json'
}
IMAGES_FOLDER = 'ap-10k\data'


# Dataset
class AP10KDataset(Dataset):
    def __init__(self, annotation_path, img_dir, transform=None):
        with open(annotation_path) as f:
            data = json.load(f)

        self.img_dir = img_dir
        self.transform = transform
        self.img_info = {img['id']: img for img in data['images']}
        self.annotations = data['annotations']

        # Фильтрация аннотаций: оставляем только те, где есть ключевые точки
        self.valid_indices = [
            idx for idx, ann in enumerate(self.annotations)
            if len(ann['keypoints']) > 0 and
               any(v > 0 for v in ann['keypoints'][2::3])  # Проверяем visibility > 0
        ]

        print(f"Loaded {len(self.valid_indices)} valid samples out of {len(self.annotations)}")

    def __len__(self):
        return len(self.valid_indices)

    def __getitem__(self, idx):
        real_idx = self.valid_indices[idx]
        ann = self.annotations[real_idx]
        img_info = self.img_info[ann['image_id']]

        # Загрузка изображения
        img_path = os.path.join(self.img_dir, img_info['file_name'])
        img = cv2.imread(img_path)
        if img is None:
            raise ValueError(f"Failed to load image: {img_path}")
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # Обработка ключевых точек
        kpts = np.array(ann['keypoints']).reshape(-1, 3)
        kpts_xy = kpts[:, :2]  # Координаты
        visibility = kpts[:, 2]  # Видимость

        # Фильтрация: оставляем только видимые точки (visibility > 0)
        visible_mask = visibility > 0
        kpts_xy = kpts_xy[visible_mask]

        # Если нет видимых точек, возвращаем нули (но такие случаи уже отфильтрованы в __init__)
        if len(kpts_xy) == 0:
            kpts_xy = np.zeros((17, 2))  # Резервный вариант

        # Нормализация
        h, w = img.shape[:2]
        kpts_norm = kpts_xy / np.array([w, h])

        if self.transform:
            transformed = self.transform(image=img, keypoints=kpts_norm)
            img = transformed['image']
            kpts_norm = np.array(transformed['keypoints'])

        # Всегда возвращаем 34 значения (17 точек * 2 координаты)
        output_kpts = np.zeros(34)  # По умолчанию нули
        output_kpts[:len(kpts_norm.flatten())] = kpts_norm.flatten()  # Заполняем доступные точки

        return img, output_kpts


# Аугментации
transform = A.Compose([
    A.Resize(256, 256),
    A.HorizontalFlip(p=0.5),
    A.Rotate(limit=20, p=0.3, border_mode=cv2.BORDER_CONSTANT),
    A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1, p=0.3),
    A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ToTensorV2()
], keypoint_params=A.KeypointParams(format='xy'))

# Создание датасетов
train_dataset = AP10KDataset(
    annotation_path=ANNOTATIONS['train'],
    img_dir=IMAGES_FOLDER,
    transform=transform
)

val_dataset = AP10KDataset(
    annotation_path=ANNOTATIONS['val'],
    img_dir=IMAGES_FOLDER,
    transform=transform
)


class KeypointModel(nn.Module):
    def __init__(self, num_keypoints):
        super().__init__()
        # Бэкбон
        self.backbone = torchvision.models.resnet34(weights="IMAGENET1K_V1")
        self.backbone.fc = nn.Identity()

        # Головы
        self.fc_shared = nn.Sequential(
            nn.Linear(512, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.3)
        )
        self.fc_x = nn.Linear(512, num_keypoints // 2)
        self.fc_y = nn.Linear(512, num_keypoints // 2)

        # Инициализация весов голов
        for m in [self.fc_x, self.fc_y]:
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')

    def forward(self, x):
        features = self.backbone(x)
        shared = self.fc_shared(features)
        x_coords = self.fc_x(shared)
        y_coords = self.fc_y(shared)
        return torch.stack([x_coords, y_coords], dim=2).flatten(1)


# DataLoader
BATCH_SIZE = 32
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE)

# Модель (17 ключевых точек * 2 координаты)
num_keypoints = 17 * 2
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = KeypointModel(num_keypoints).to(device)
criterion = nn.SmoothL1Loss()
optimizer = optim.AdamW(model.parameters(), lr=3e-4)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=3, factor=0.1)

for epoch in range(100):
    model.train()
    for images, keypoints in train_loader:
        images = images.to(device)
        keypoints = keypoints.to(device)

        outputs = model(images)
        loss = criterion(outputs, keypoints)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    print(f"Epoch {epoch + 1}, Loss: {loss.item():.4f}")


def visualize_ap10k(img_tensor, keypoints, title=''):
    """Визуализация одного примера из AP-10K"""
    img = img_tensor.permute(1, 2, 0).numpy()
    img = (img * [0.229, 0.224, 0.225]) + [0.485, 0.456, 0.406]  # Денормализация
    img = np.clip(img, 0, 1)

    kpts = keypoints.reshape(-1, 2) * 256  # Масштабируем обратно

    plt.figure(figsize=(8, 8))
    plt.imshow(img)
    plt.scatter(kpts[:, 0], kpts[:, 1], c='red', s=20, marker='o')
    plt.title(title)
    plt.axis('off')
    plt.show()

model.eval()
# Пример визуализации
sample_img, sample_kpts = train_dataset[42]
visualize_ap10k(sample_img, sample_kpts, 'AP-10K Sample')


# Сохраняем веса модели и параметры
torch.save({
    'model_state_dict': model.state_dict(),
    'num_keypoints': num_keypoints,
    'transform': transform,
    'model_class': 'KeypointModel'  # Добавляем информацию о классе
}, 'keypoint_model6.pth')

print("Модель сохранена в keypoint_model6.pth")
