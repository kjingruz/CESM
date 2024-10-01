import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import pandas as pd
from PIL import Image
import timm
from torch.optim.lr_scheduler import CosineAnnealingLR

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

class ViTModel(nn.Module):
    def __init__(self, num_classes=3):
        super(ViTModel, self).__init__()
        self.vit = timm.create_model('vit_base_patch16_224', pretrained=True)
        num_ftrs = self.vit.head.in_features
        self.vit.head = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(num_ftrs, num_classes)
        )

    def forward(self, x):
        return self.vit(x)

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
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            if phase == 'train':
                scheduler.step()

            epoch_loss = running_loss / len(dataloaders[phase].dataset)
            epoch_acc = running_corrects.double() / len(dataloaders[phase].dataset)

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

    model = ViTModel(num_classes=3)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=0.01)
    scheduler = CosineAnnealingLR(optimizer, T_max=100, eta_min=1e-6)

    model = train_model(model, dataloaders, criterion, optimizer, scheduler, num_epochs=100, device=device, patience=20)

    # Save the trained model
    torch.save(model.state_dict(), 'vit_classification_model.pth')

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
