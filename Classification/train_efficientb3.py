import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms
import pandas as pd
from PIL import Image
import numpy as np
import timm

class CustomDataset(Dataset):
    def __init__(self, csv_file, img_dir, transform=None):
        self.data = pd.read_csv(csv_file)
        self.img_dir = img_dir
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_name = os.path.join(self.img_dir, self.data.iloc[idx, 0] + '.jpg')
        image = Image.open(img_name).convert('RGB')
        label = self.data.iloc[idx, 1]

        if self.transform:
            image = self.transform(image)

        return image, label

def get_transform(train):
    if train:
        return transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(15),
            transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
    else:
        return transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

class CustomEfficientNet(nn.Module):
    def __init__(self, num_classes=3):
        super(CustomEfficientNet, self).__init__()
        self.efficientnet = timm.create_model('efficientnet_b3', pretrained=True)
        in_features = self.efficientnet.classifier.in_features
        self.efficientnet.classifier = nn.Sequential(
           # nn.Dropout(0.3),
            nn.Linear(in_features, num_classes)
        )

    def forward(self, x):
        return self.efficientnet(x)

def mixup_data(x, y, alpha=1.0):
    lam = np.random.beta(alpha, alpha)
    batch_size = x.size()[0]
    index = torch.randperm(batch_size).cuda()
    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam

def mixup_criterion(criterion, pred, y_a, y_b, lam):
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)

def train_model(model, dataloaders, criterion, optimizer, scheduler, num_epochs=100, device='cuda', patience=20):
    model.to(device)
    best_loss = float('inf')
    best_model_wts = model.state_dict()
    no_improve = 0

    for epoch in range(num_epochs):
        print(f'Epoch {epoch+1}/{num_epochs}')
        print('-' * 10)

        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()
            else:
                model.eval()

            running_loss = 0.0
            running_corrects = 0

            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)

                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == 'train'):
                    if phase == 'train':
                        inputs, labels_a, labels_b, lam = mixup_data(inputs, labels)
                        outputs = model(inputs)
                        loss = mixup_criterion(criterion, outputs, labels_a, labels_b, lam)
                    else:
                        outputs = model(inputs)
                        loss = criterion(outputs, labels)

                    _, preds = torch.max(outputs, 1)

                    if phase == 'train':
                        loss.backward()
                        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                        optimizer.step()

                running_loss += loss.item() * inputs.size(0)
                if phase == 'val':
                    running_corrects += torch.sum(preds == labels.data)

            if phase == 'train':
                scheduler.step()

            epoch_loss = running_loss / len(dataloaders[phase].dataset)
            epoch_acc = running_corrects.double() / len(dataloaders[phase].dataset) if phase == 'val' else 0.0

            print(f'{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')

            if phase == 'val':
                if epoch_loss < best_loss:
                    best_loss = epoch_loss
                    best_model_wts = model.state_dict()
                    no_improve = 0
                else:
                    no_improve += 1

        if no_improve >= patience:
            print(f'Early stopping triggered after {epoch+1} epochs')
            break

    print(f'Best val Loss: {best_loss:4f}')
    model.load_state_dict(best_model_wts)
    return model

def main():
    data_dir = '../data/images'
    train_csv = '../class_data/train_classification.csv'
    val_csv = '../class_data/val_classification.csv'
    test_csv = '../class_data/test_classification.csv'

    # Datasets
    image_datasets = {
        'train': CustomDataset(train_csv, data_dir, get_transform(train=True)),
        'val': CustomDataset(val_csv, data_dir, get_transform(train=False)),
        'test': CustomDataset(test_csv, data_dir, get_transform(train=False))
    }

    # DataLoaders
    dataloaders = {x: DataLoader(image_datasets[x], batch_size=32, shuffle=(x == 'train'), num_workers=4)
                   for x in ['train', 'val', 'test']}

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    model = CustomEfficientNet(num_classes=3)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    optimizer = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-2)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=100, eta_min=1e-6)

    model = train_model(model, dataloaders, criterion, optimizer, scheduler, num_epochs=100, device=device, patience=20)

    # Save the trained model
    torch.save(model.state_dict(), 'advanced_torch_classification_model.pth')

    # Evaluation on test set
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in dataloaders['test']:
            inputs = inputs.to(device)
            labels = labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    print(f'Accuracy on test set: {100 * correct / total:.2f}%')

if __name__ == '__main__':
    main()
