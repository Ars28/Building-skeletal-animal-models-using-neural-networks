import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def generate_synthetic_data(num_samples=1000, num_keypoints=10):
    X_2d = np.random.rand(num_samples, num_keypoints * 2)
    Z_offset = np.random.uniform(0.5, 1.5, size=(num_samples, num_keypoints, 1))
    X_3d = np.concatenate([X_2d.reshape(num_samples, num_keypoints, 2), Z_offset], axis=2)
    X_3d = X_3d.reshape(num_samples, num_keypoints * 3)
    return X_2d.astype(np.float32), X_3d.astype(np.float32)

num_keypoints = 10
X_2d, X_3d = generate_synthetic_data(num_samples=2000, num_keypoints=num_keypoints)

class Keypoint3DDataset(Dataset):
    def __init__(self, X_2d, X_3d):
        self.X_2d = X_2d
        self.X_3d = X_3d

    def __len__(self):
        return len(self.X_2d)

    def __getitem__(self, idx):
        return self.X_2d[idx], self.X_3d[idx]

dataset = Keypoint3DDataset(X_2d, X_3d)
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

class PoseLiftingModel(nn.Module):
    def __init__(self, num_keypoints):
        super(PoseLiftingModel, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(num_keypoints * 2, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, num_keypoints * 3)
        )

    def forward(self, x):
        return self.fc(x)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = PoseLiftingModel(num_keypoints).to(device)

criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=1e-3)
epochs = 50

for epoch in range(epochs):
    model.train()
    running_loss = 0.0
    for X_batch, Y_batch in dataloader:
        X_batch = X_batch.to(device)
        Y_batch = Y_batch.to(device)

        optimizer.zero_grad()
        outputs = model(X_batch)
        loss = criterion(outputs, Y_batch)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    print(f"Epoch [{epoch+1}/{epochs}], Loss: {running_loss/len(dataloader):.4f}")

model.eval()
sample_2d = torch.tensor(X_2d[0:1]).to(device)
pred_3d = model(sample_2d).cpu().detach().numpy().reshape(num_keypoints, 3)
gt_3d = X_3d[0].reshape(num_keypoints, 3)

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

ax.scatter(gt_3d[:, 0], gt_3d[:, 1], gt_3d[:, 2], c='g', label='Ground Truth')
ax.scatter(pred_3d[:, 0], pred_3d[:, 1], pred_3d[:, 2], c='r', label='Predicted')
ax.legend()
plt.show()