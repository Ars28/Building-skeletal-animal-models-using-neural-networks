import torch
import matplotlib.pyplot as plt
import cv2
from new_class import KeypointModel  # Импорт класса модели
import numpy as np

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_model(model_path):
    # Загружаем с weights_only=False для совместимости
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    model = KeypointModel(20 * 2).to(device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    return model, checkpoint['transform']


model, transform = load_model('keypoint_model6.pth')


def predict_keypoints(model, transform, image_path):
    # Загрузка и преобразование изображения
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Не удалось загрузить изображение: {image_path}")

    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    h, w = image.shape[:2]

    # Применяем аугментации
    transformed = transform(image=image, keypoints=[[0, 0]])  # Фиктивные точки
    image_tensor = transformed['image'].unsqueeze(0).to(device)

    # Предсказание
    with torch.no_grad():
        keypoints_norm = model(image_tensor).cpu().numpy()[0]

    # Преобразование координат обратно в оригинальный размер
    keypoints = keypoints_norm.reshape(-1, 2) * np.array([w, h])

    return image, keypoints


def visualize_prediction(image, keypoints, title='Predicted Keypoints'):
    plt.figure(figsize=(10, 10))
    plt.imshow(image)

    # Отображаем точки
    for i, (x, y) in enumerate(keypoints):
        plt.scatter(x, y, c='red', s=10, marker='o')
        plt.text(x, y, str(i), color='white', fontsize=8,
                 bbox=dict(facecolor='red', alpha=0.5, pad=1))

    plt.title(title)
    plt.axis('off')
    plt.show()


test_image_path = "test3.jpg"
original_image, predicted_keypoints = predict_keypoints(model, transform, test_image_path)
visualize_prediction(original_image, predicted_keypoints)
