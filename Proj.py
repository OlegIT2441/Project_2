import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import models
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix, f1_score, accuracy_score
import os

class SyntheticMultispectralDataset(Dataset):
    def __init__(self, num_samples=600, img_size=64):
        self.num_samples = num_samples
        self.img_size = img_size
        
        # Генерация данных
        self.images = []
        self.labels = []
        

        
        for i in range(num_samples):

            img = np.random.rand(4, img_size, img_size).astype(np.float32) * 0.2
            
            label = i % 3
            
            if label == 0: # Healthy
                # R низкий, G высокий, B низкий, NIR очень высокий
                img[0, :, :] += 0.1 # R
                img[1, :, :] += np.random.uniform(0.6, 0.9) # G
                img[2, :, :] += 0.1 # B
                img[3, :, :] += np.random.uniform(0.8, 1.0) # NIR (высокий NDVI)
                
            elif label == 1: # Diseased
                # R средний, G средний, B средний, NIR средний
                img[0, :, :] += np.random.uniform(0.4, 0.6)
                img[1, :, :] += np.random.uniform(0.3, 0.5) # Меньше зеленого
                img[2, :, :] += 0.2
                img[3, :, :] += np.random.uniform(0.4, 0.6) # NIR падает
                
            else: # Drought Stress
                # R высокий, G низкий, B средний, NIR низкий
                img[0, :, :] += np.random.uniform(0.7, 0.9) # Коричневый
                img[1, :, :] += 0.2
                img[2, :, :] += 0.3
                img[3, :, :] += np.random.uniform(0.2, 0.4) # Низкий NIR
            
            # Клиппинг значений
            img = np.clip(img, 0, 1)
            self.images.append(img)
            self.labels.append(label)
            
        self.images = torch.tensor(np.array(self.images), dtype=torch.float32)
        self.labels = torch.tensor(self.labels, dtype=torch.long)

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        return self.images[idx], self.labels[idx]

# ---------------------------------------------------------
# 2. Архитектура модели (Modified EfficientNet)
# ---------------------------------------------------------

class MultispectralEfficientNet(nn.Module):
    def __init__(self, num_classes=3, in_channels=4, pretrained=True):
        super(MultispectralEfficientNet, self).__init__()
        
        # Загружаем предобученную модель (ImageNet)
        self.model = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.DEFAULT if pretrained else None)
        
        # MODIFICATION: Меняем первый сверточный слой для 4-х каналов
        # Оригинальный слой: Conv2d(3, 32, kernel_size=(3, 3), ...)
        original_conv = self.model.features[0][0]
        
        # Создаем новый слой для 4 каналов
        new_conv = nn.Conv2d(
            in_channels, 
            original_conv.out_channels, 
            kernel_size=original_conv.kernel_size, 
            stride=original_conv.stride, 
            padding=original_conv.padding, 
            bias=original_conv.bias
        )
        
        # Инициализация весов:
        # Копируем веса RGB каналов
        with torch.no_grad():
            new_conv.weight[:, :3, :, :] = original_conv.weight
            # 4-й канал (NIR) инициализируем средним от существующих (или случайным)
            new_conv.weight[:, 3, :, :] = original_conv.weight.mean(dim=1)
            
        self.model.features[0][0] = new_conv
        
        # Меняем классификатор под наше количество классов
        in_features = self.model.classifier[1].in_features
        self.model.classifier[1] = nn.Linear(in_features, num_classes)

    def forward(self, x):
        return self.model(x)

# ---------------------------------------------------------
# 3. Обучение и Валидация
# ---------------------------------------------------------

def train_model(model, train_loader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    running_corrects = 0
    
    for inputs, labels in train_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(inputs)
        _, preds = torch.max(outputs, 1)
        loss = criterion(outputs, labels)
        
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item() * inputs.size(0)
        running_corrects += torch.sum(preds == labels.data)
        
    epoch_loss = running_loss / len(train_loader.dataset)
    epoch_acc = running_corrects.double() / len(train_loader.dataset)
    return epoch_loss, epoch_acc

def evaluate_model(model, test_loader, criterion, device, phase='val'):
    model.eval()
    running_loss = 0.0
    running_corrects = 0
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            loss = criterion(outputs, labels)
            
            running_loss += loss.item() * inputs.size(0)
            running_corrects += torch.sum(preds == labels.data)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            
    epoch_loss = running_loss / len(test_loader.dataset)
    epoch_acc = running_corrects.double() / len(test_loader.dataset)
    
    return epoch_loss, epoch_acc, all_labels, all_preds

# ---------------------------------------------------------
# 4. Основной запуск
# ---------------------------------------------------------

if __name__ == "__main__":
    # Настройки
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    BATCH_SIZE = 16
    EPOCHS = 5
    NUM_CLASSES = 3
    
    print(f"Используется устройство: {DEVICE}")
    
    # 1. Данные
    print("Генерация данных...")
    full_dataset = SyntheticMultispectralDataset(num_samples=600, img_size=64)
    
    # Разделение Train / Val / Test (60% / 20% / 20%)
    train_size = int(0.6 * len(full_dataset))
    val_size = int(0.2 * len(full_dataset))
    test_size = len(full_dataset) - train_size - val_size
    
    train_dataset, val_dataset, test_dataset = random_split(
        full_dataset, [train_size, val_size, test_size], generator=torch.Generator().manual_seed(42)
    )
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)
    
    # 2. Модель
    model = MultispectralEfficientNet(num_classes=NUM_CLASSES, in_channels=4).to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    # 3. Цикл обучения
    print("Начинаем обучение...")
    history = {'train_loss': [], 'val_loss': [], 'val_acc': []}
    
    for epoch in range(EPOCHS):
        train_loss, train_acc = train_model(model, train_loader, criterion, optimizer, DEVICE)
        val_loss, val_acc, _, _ = evaluate_model(model, val_loader, criterion, DEVICE)
        
        print(f"Epoch {epoch+1}/{EPOCHS} | Train Loss: {train_loss:.4f} Acc: {train_acc:.4f} | Val Loss: {val_loss:.4f} Acc: {val_acc:.4f}")
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)

    # 4. Тестирование и Отчет
    print("\n--- Финальный отчет на Test Set ---")
    test_loss, test_acc, y_true, y_pred = evaluate_model(model, test_loader, criterion, DEVICE)
    
    # Метрики
    print(f"Accuracy: {accuracy_score(y_true, y_pred):.4f}")
    print(f"Macro F1-Score: {f1_score(y_true, y_pred, average='macro'):.4f}")
    print("\nClassification Report:")
    target_names = ['Healthy', 'Diseased', 'Drought Stress']
    print(classification_report(y_true, y_pred, target_names=target_names))
    
    # Confusion Matrix
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=target_names, yticklabels=target_names)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    plt.show()
