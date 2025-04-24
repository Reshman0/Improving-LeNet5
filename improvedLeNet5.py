import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt

# 1. Data preparation
transform = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

train_dataset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
val_dataset   = torchvision.datasets.CIFAR10(root='./data', train=True, download=False,
                                           transform=transforms.Compose([
                                               transforms.ToTensor(),
                                               transforms.Normalize((0.5,), (0.5,))
                                           ]))
test_dataset  = torchvision.datasets.CIFAR10(root='./data', train=False, download=True,
                                           transform=transforms.Compose([
                                               transforms.ToTensor(),
                                               transforms.Normalize((0.5,), (0.5,))
                                           ]))

# Split train into train/val
num_train = len(train_dataset)
indices = list(range(num_train))
split = int(0.8 * num_train)
train_idx, val_idx = indices[:split], indices[split:]
train_loader = DataLoader(torch.utils.data.Subset(train_dataset, train_idx), batch_size=128, shuffle=True, num_workers=2)
val_loader   = DataLoader(torch.utils.data.Subset(train_dataset, val_idx),   batch_size=128, shuffle=False, num_workers=2)
test_loader  = DataLoader(test_dataset, batch_size=128, shuffle=False, num_workers=2)

# 2. Improved LeNet5 model (CNN-only, no dense layers)
class ImprovedLeNet5(nn.Module):
    def __init__(self, num_classes=10):
        super(ImprovedLeNet5, self).__init__()
        # Convolutional backbone
        self.features = nn.Sequential(
            # Conv Block 1
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.1, inplace=True),
            nn.MaxPool2d(2),
            nn.Dropout(0.2),
            # Conv Block 2
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.1, inplace=True),
            nn.MaxPool2d(2),
            nn.Dropout(0.3),
            # Conv Block 3
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.1, inplace=True),
            nn.MaxPool2d(2),
            nn.Dropout(0.4)
        )
        # Classifier via 1x1 conv + global avg pool
        self.classifier = nn.Sequential(
            nn.Conv2d(128, num_classes, kernel_size=1),
            nn.AdaptiveAvgPool2d(1)  # output size (num_classes, 1, 1)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        x = x.view(x.size(0), -1)
        return x

# 3. Training and evaluation functions

def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    running_loss, correct, total = 0.0, 0, 0
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * images.size(0)
        _, preds = outputs.max(1)
        correct += preds.eq(labels).sum().item()
        total += labels.size(0)
    return running_loss / total, correct / total


def evaluate(model, loader, criterion, device):
    model.eval()
    running_loss, correct, total = 0.0, 0, 0
    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            running_loss += loss.item() * images.size(0)
            _, preds = outputs.max(1)
            correct += preds.eq(labels).sum().item()
            total += labels.size(0)
    return running_loss / total, correct / total

# 4. Main training loop with logging

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = ImprovedLeNet5(num_classes=10).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=3)

    num_epochs = 30
    history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}

    for epoch in range(1, num_epochs + 1):
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc     = evaluate(model, val_loader,   criterion, device)
        scheduler.step(val_loss)
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        print(f"Epoch {epoch}/{num_epochs} - "
              f"Train loss: {train_loss:.4f}, Acc: {train_acc:.4f} - "
              f"Val loss: {val_loss:.4f}, Acc: {val_acc:.4f}")
        # Early stopping condition can be added here

    # Test evaluation
    test_loss, test_acc = evaluate(model, test_loader, criterion, device)
    print(f"Test   loss: {test_loss:.4f}, Acc: {test_acc:.4f}")

    # 5. Plot results
    epochs = range(1, num_epochs + 1)
    plt.figure(figsize=(12,5))
    plt.subplot(1,2,1)
    plt.plot(epochs, history['train_loss'], label='Train Loss')
    plt.plot(epochs, history['val_loss'],   label='Val Loss')
    plt.legend(); plt.title('Loss')

    plt.subplot(1,2,2)
    plt.plot(epochs, history['train_acc'], label='Train Acc')
    plt.plot(epochs, history['val_acc'],   label='Val Acc')
    plt.legend(); plt.title('Accuracy')
    plt.show()

if __name__ == '__main__':
    main()
