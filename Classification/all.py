import os
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from PIL import Image
from sklearn.metrics import confusion_matrix
import timm
from torch.optim.lr_scheduler import CosineAnnealingLR
import torch.optim as optim
import albumentations as A
from albumentations.pytorch import ToTensorV2

# Define the CustomDataset class for standard transforms
class CustomDataset(Dataset):
    def __init__(self, csv_file, img_dir, transform=None):
        self.data = pd.read_csv(csv_file)
        self.img_dir = img_dir
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_name = os.path.join(self.img_dir, str(self.data.iloc[idx, 0]) + '.jpg')
        image = Image.open(img_name).convert('RGB')
        label = int(self.data.iloc[idx, 1])

        if self.transform:
            image = self.transform(image)

        return image, label

# Define the CustomDataset class for Albumentations transforms
class CustomDatasetAlbumentations(Dataset):
    def __init__(self, csv_file, img_dir, transform=None):
        self.data = pd.read_csv(csv_file)
        self.img_dir = img_dir
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_name = os.path.join(self.img_dir, str(self.data.iloc[idx, 0]) + '.jpg')
        image = Image.open(img_name).convert('RGB')
        label = int(self.data.iloc[idx, 1])

        if self.transform:
            image = np.array(image)
            augmented = self.transform(image=image)
            image = augmented['image']
        else:
            image = transforms.ToTensor()(image)

        return image, label

# Define the transformation functions
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

# Albumentations transform for Model 1 (ResNet50 with Albumentations)
def get_train_transform_model1():
    return A.Compose([
        A.RandomResizedCrop(height=224, width=224),
        A.HorizontalFlip(p=0.5),
        A.RandomBrightnessContrast(p=0.2),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2(),
    ])

# Define the compute_sens_spec function
def compute_sens_spec(y_true, y_pred, pos_label):
    y_true_binary = np.array([1 if y == pos_label else 0 for y in y_true])
    y_pred_binary = np.array([1 if y == pos_label else 0 for y in y_pred])
    tn, fp, fn, tp = confusion_matrix(y_true_binary, y_pred_binary).ravel()
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    return sensitivity, specificity

# Training function
def train_model(model, dataloaders, criterion, optimizer, scheduler, num_epochs=100, device='cuda', patience=20):
    model.to(device)
    best_loss = float('inf')
    best_model_wts = model.state_dict()
    no_improve = 0

    for epoch in range(num_epochs):
        print(f'Epoch {epoch+1}/{num_epochs} for {model.model_name}')
        print('-' * 50)

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

            # Early stopping
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
    # Load best model weights
    model.load_state_dict(best_model_wts)
    return model

# Training function for models using Mixup
def train_model_mixup(model, dataloaders, criterion, optimizer, scheduler, num_epochs=100, device='cuda', patience=20):
    model.to(device)
    best_loss = float('inf')
    best_model_wts = model.state_dict()
    no_improve = 0

    for epoch in range(num_epochs):
        print(f'Epoch {epoch+1}/{num_epochs} for {model.model_name}')
        print('-' * 50)

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
                    optimizer.step()

                running_loss += loss.item() * inputs.size(0)
                if phase == 'val':
                    running_corrects += torch.sum(preds == labels.data)

            if phase == 'train':
                scheduler.step()

            epoch_loss = running_loss / len(dataloaders[phase].dataset)
            if phase == 'val':
                epoch_acc = running_corrects.double() / len(dataloaders[phase].dataset)
                print(f'{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')
            else:
                print(f'{phase} Loss: {epoch_loss:.4f}')

            # Early stopping
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
    # Load best model weights
    model.load_state_dict(best_model_wts)
    return model

# Mixup functions
def mixup_data(x, y, alpha=1.0):
    '''Returns mixed inputs, pairs of targets, and lambda'''
    lam = np.random.beta(alpha, alpha)
    batch_size = x.size()[0]
    index = torch.randperm(batch_size).to(x.device)
    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam

def mixup_criterion(criterion, pred, y_a, y_b, lam):
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)

# Helper function to get model by class name and dropout settings
def get_model_by_name(model_class_name, use_dropout=False, dropout_rates=None):
    if model_class_name == 'DenseNetModel':
        return DenseNetModel(num_classes=3, use_dropout=use_dropout, dropout_rate=dropout_rates)
    elif model_class_name == 'CustomEfficientNet_B4':
        return CustomEfficientNet_B4(num_classes=3, use_dropout=use_dropout, dropout_rate=dropout_rates)
    elif model_class_name == 'CustomEfficientNet_B3':
        return CustomEfficientNet_B3(num_classes=3, use_dropout=use_dropout, dropout_rate=dropout_rates)
    elif model_class_name == 'EfficientNetV2Model':
        return EfficientNetV2Model(num_classes=3, use_dropout=use_dropout, dropout_rate=dropout_rates)
    elif model_class_name == 'ViTModel':
        return ViTModel(num_classes=3, use_dropout=use_dropout, dropout_rate=dropout_rates)
    elif model_class_name == 'CustomResNet':
        return CustomResNet(num_classes=3, use_dropout=use_dropout, dropout_rates=dropout_rates)
    else:
        raise ValueError(f"Unknown model class: {model_class_name}")

# Model definitions with dropout options
class CustomResNet(nn.Module):
    def __init__(self, num_classes=3, use_dropout=False, dropout_rates=None):
        super(CustomResNet, self).__init__()
        self.model_name = 'ResNet50'
        self.resnet = models.resnet50(pretrained=True)
        num_ftrs = self.resnet.fc.in_features

        layers = []
        if use_dropout:
            if dropout_rates is None:
                dropout_rates = [0.5]
            for rate in dropout_rates:
                layers.append(nn.Dropout(rate))
        layers.append(nn.Linear(num_ftrs, 512))
        layers.append(nn.ReLU())
        if use_dropout:
            for rate in dropout_rates:
                layers.append(nn.Dropout(rate))
        layers.append(nn.Linear(512, num_classes))

        self.resnet.fc = nn.Sequential(*layers)

    def forward(self, x):
        return self.resnet(x)

class DenseNetModel(nn.Module):
    def __init__(self, num_classes=3, use_dropout=False, dropout_rate=0.5):
        super(DenseNetModel, self).__init__()
        self.model_name = 'DenseNet121'
        self.densenet = models.densenet121(pretrained=True)
        num_ftrs = self.densenet.classifier.in_features

        layers = []
        if use_dropout:
            layers.append(nn.Dropout(dropout_rate))
        layers.append(nn.Linear(num_ftrs, num_classes))

        self.densenet.classifier = nn.Sequential(*layers)

    def forward(self, x):
        return self.densenet(x)

class CustomEfficientNet_B4(nn.Module):
    def __init__(self, num_classes=3, use_dropout=False, dropout_rate=0.5):
        super(CustomEfficientNet_B4, self).__init__()
        self.model_name = 'EfficientNet-B4'
        self.efficientnet = timm.create_model('efficientnet_b4', pretrained=True)
        in_features = self.efficientnet.classifier.in_features

        layers = []
        if use_dropout:
            layers.append(nn.Dropout(dropout_rate))
        layers.append(nn.Linear(in_features, num_classes))

        self.efficientnet.classifier = nn.Sequential(*layers)

    def forward(self, x):
        return self.efficientnet(x)

class CustomEfficientNet_B3(nn.Module):
    def __init__(self, num_classes=3, use_dropout=False, dropout_rate=0.5):
        super(CustomEfficientNet_B3, self).__init__()
        self.model_name = 'EfficientNet-B3'
        self.efficientnet = timm.create_model('efficientnet_b3', pretrained=True)
        in_features = self.efficientnet.classifier.in_features

        layers = []
        if use_dropout:
            layers.append(nn.Dropout(dropout_rate))
        layers.append(nn.Linear(in_features, num_classes))

        self.efficientnet.classifier = nn.Sequential(*layers)

    def forward(self, x):
        return self.efficientnet(x)

class EfficientNetV2Model(nn.Module):
    def __init__(self, num_classes=3, use_dropout=False, dropout_rate=0.5):
        super(EfficientNetV2Model, self).__init__()
        self.model_name = 'EfficientNetV2-S'
        self.efficientnet = timm.create_model('tf_efficientnetv2_s', pretrained=True)
        num_ftrs = self.efficientnet.classifier.in_features

        layers = []
        if use_dropout:
            layers.append(nn.Dropout(dropout_rate))
        layers.append(nn.Linear(num_ftrs, num_classes))

        self.efficientnet.classifier = nn.Sequential(*layers)

    def forward(self, x):
        return self.efficientnet(x)

class ViTModel(nn.Module):
    def __init__(self, num_classes=3, use_dropout=False, dropout_rate=0.5):
        super(ViTModel, self).__init__()
        self.model_name = 'ViT Base Patch16 224'
        self.vit = timm.create_model('vit_base_patch16_224', pretrained=True)
        num_ftrs = self.vit.head.in_features

        layers = []
        if use_dropout:
            layers.append(nn.Dropout(dropout_rate))
        layers.append(nn.Linear(num_ftrs, num_classes))

        self.vit.head = nn.Sequential(*layers)

    def forward(self, x):
        return self.vit(x)

# Main function
def main():
    # Update these paths according to your directory structure
    data_dir = '../data/images'
    train_csv = '../class_data/train_classification.csv'
    val_csv = '../class_data/val_classification.csv'
    test_csv = '../class_data/test_classification.csv'

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # List to hold all model evaluations
    all_models_data = []

    # Define class names
    class_names = ['Normal', 'Benign', 'Malignant']

    # List of models to train and evaluate
    models_info = [
        # DenseNet121 with dropout (0.3)
        {
            'Model Name': 'DenseNet121 with Dropout 0.3',
            'Model Definition': 'DenseNet121',
            'Pretrained Dataset': 'ImageNet',
            'Augmentation Method': 'Standard Transforms',
            'Model Class': 'DenseNetModel',
            'Save Path': 'densenet121_dropout0.3.pth',
            'Use Albumentations': False,
            'Use Mixup': False,
            'Use Dropout': True,
            'Dropout Rate': 0.3
        },
        # DenseNet121 without dropout
        {
            'Model Name': 'DenseNet121 without Dropout',
            'Model Definition': 'DenseNet121',
            'Pretrained Dataset': 'ImageNet',
            'Augmentation Method': 'Standard Transforms',
            'Model Class': 'DenseNetModel',
            'Save Path': 'densenet121_no_dropout.pth',
            'Use Albumentations': False,
            'Use Mixup': False,
            'Use Dropout': False,
            'Dropout Rate': None
        },
        # EfficientNet-B3 with dropout (0.3)
        {
            'Model Name': 'EfficientNet-B3 with Dropout 0.3',
            'Model Definition': 'EfficientNet-B3',
            'Pretrained Dataset': 'ImageNet',
            'Augmentation Method': 'Standard Transforms',
            'Model Class': 'CustomEfficientNet_B3',
            'Save Path': 'efficientnet_b3_dropout0.3.pth',
            'Use Albumentations': False,
            'Use Mixup': False,
            'Use Dropout': True,
            'Dropout Rate': 0.3
        },
        # EfficientNet-B3 without dropout
        {
            'Model Name': 'EfficientNet-B3 without Dropout',
            'Model Definition': 'EfficientNet-B3',
            'Pretrained Dataset': 'ImageNet',
            'Augmentation Method': 'Standard Transforms',
            'Model Class': 'CustomEfficientNet_B3',
            'Save Path': 'efficientnet_b3_no_dropout.pth',
            'Use Albumentations': False,
            'Use Mixup': False,
            'Use Dropout': False,
            'Dropout Rate': None
        },
        # ResNet50 without dropout
        {
            'Model Name': 'ResNet50 without Dropout',
            'Model Definition': 'ResNet50',
            'Pretrained Dataset': 'ImageNet',
            'Augmentation Method': 'Albumentations',
            'Model Class': 'CustomResNet',
            'Save Path': 'resnet50_no_dropout.pth',
            'Use Albumentations': True,
            'Use Mixup': False,
            'Use Dropout': False,
            'Dropout Rate': None
        },
        # ResNet50 with dropout (0.3)
        {
            'Model Name': 'ResNet50 with Dropout 0.3',
            'Model Definition': 'ResNet50',
            'Pretrained Dataset': 'ImageNet',
            'Augmentation Method': 'Albumentations',
            'Model Class': 'CustomResNet',
            'Save Path': 'resnet50_dropout0.3.pth',
            'Use Albumentations': True,
            'Use Mixup': False,
            'Use Dropout': True,
            'Dropout Rate': [0.3]
        },
        # ResNet50 with dropout (0.3 and 0.5)
        {
            'Model Name': 'ResNet50 with Dropout 0.3 and 0.5',
            'Model Definition': 'ResNet50',
            'Pretrained Dataset': 'ImageNet',
            'Augmentation Method': 'Albumentations',
            'Model Class': 'CustomResNet',
            'Save Path': 'resnet50_dropout0.3_0.5.pth',
            'Use Albumentations': True,
            'Use Mixup': False,
            'Use Dropout': True,
            'Dropout Rate': [0.3, 0.5]
        },
        # Vision Transformer without dropout
        {
            'Model Name': 'ViT without Dropout',
            'Model Definition': 'ViT Base Patch16 224',
            'Pretrained Dataset': 'ImageNet',
            'Augmentation Method': 'Standard Transforms',
            'Model Class': 'ViTModel',
            'Save Path': 'vit_no_dropout.pth',
            'Use Albumentations': False,
            'Use Mixup': False,
            'Use Dropout': False,
            'Dropout Rate': None
        },
        # Vision Transformer with dropout (0.5)
        {
            'Model Name': 'ViT with Dropout 0.5',
            'Model Definition': 'ViT Base Patch16 224',
            'Pretrained Dataset': 'ImageNet',
            'Augmentation Method': 'Standard Transforms',
            'Model Class': 'ViTModel',
            'Save Path': 'vit_dropout0.5.pth',
            'Use Albumentations': False,
            'Use Mixup': False,
            'Use Dropout': True,
            'Dropout Rate': 0.5
        },
        # EfficientNetV2-S with dropout (0.3)
        {
            'Model Name': 'EfficientNetV2-S with Dropout 0.3',
            'Model Definition': 'EfficientNetV2-S',
            'Pretrained Dataset': 'ImageNet',
            'Augmentation Method': 'Standard Transforms',
            'Model Class': 'EfficientNetV2Model',
            'Save Path': 'efficientnetv2_s_dropout0.3.pth',
            'Use Albumentations': False,
            'Use Mixup': False,
            'Use Dropout': True,
            'Dropout Rate': 0.3
        },
        # EfficientNetV2-S without dropout
        {
            'Model Name': 'EfficientNetV2-S without Dropout',
            'Model Definition': 'EfficientNetV2-S',
            'Pretrained Dataset': 'ImageNet',
            'Augmentation Method': 'Standard Transforms',
            'Model Class': 'EfficientNetV2Model',
            'Save Path': 'efficientnetv2_s_no_dropout.pth',
            'Use Albumentations': False,
            'Use Mixup': False,
            'Use Dropout': False,
            'Dropout Rate': None
        },
        # EfficientNet-B4 without dropout
        {
            'Model Name': 'EfficientNet-B4 without Dropout',
            'Model Definition': 'EfficientNet-B4',
            'Pretrained Dataset': 'ImageNet',
            'Augmentation Method': 'Standard Transforms',
            'Model Class': 'CustomEfficientNet_B4',
            'Save Path': 'efficientnet_b4_no_dropout.pth',
            'Use Albumentations': False,
            'Use Mixup': True,  # Assuming Mixup as per previous code
            'Use Dropout': False,
            'Dropout Rate': None
        },
        # EfficientNet-B4 with dropout (0.3)
        {
            'Model Name': 'EfficientNet-B4 with Dropout 0.3',
            'Model Definition': 'EfficientNet-B4',
            'Pretrained Dataset': 'ImageNet',
            'Augmentation Method': 'Standard Transforms',
            'Model Class': 'CustomEfficientNet_B4',
            'Save Path': 'efficientnet_b4_dropout0.3.pth',
            'Use Albumentations': False,
            'Use Mixup': True,  # Assuming Mixup as per previous code
            'Use Dropout': True,
            'Dropout Rate': 0.3
        },
    ]

    for model_info in models_info:
        print(f"Training and Evaluating {model_info['Model Name']}...")

        # Initialize the model
        model = get_model_by_name(
            model_info['Model Class'],
            use_dropout=model_info['Use Dropout'],
            dropout_rates=model_info['Dropout Rate']
        )

        # Select the appropriate dataset class and transforms
        if model_info['Use Albumentations']:
            train_dataset = CustomDatasetAlbumentations(
                train_csv, data_dir, get_train_transform_model1())
        else:
            train_dataset = CustomDataset(
                train_csv, data_dir, get_transform(train=True))

        # Create DataLoaders
        dataloaders = {
            'train': DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=4),
            'val': DataLoader(
                CustomDataset(val_csv, data_dir, get_transform(train=False)),
                batch_size=32, shuffle=False, num_workers=4),
            'test': DataLoader(
                CustomDataset(test_csv, data_dir, get_transform(train=False)),
                batch_size=32, shuffle=False, num_workers=4)
        }

        # Define criterion, optimizer, scheduler
        criterion = nn.CrossEntropyLoss()

        # Define optimizer and scheduler
        optimizer = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=0.01)
        scheduler = CosineAnnealingLR(optimizer, T_max=100, eta_min=1e-6)

        # Modify training function for models using Mixup
        if model_info.get('Use Mixup', False):
            model = train_model_mixup(model, dataloaders, criterion, optimizer, scheduler,
                                      num_epochs=100, device=device, patience=20)
        else:
            model = train_model(model, dataloaders, criterion, optimizer, scheduler,
                                num_epochs=100, device=device, patience=20)

        # Save the trained model
        torch.save(model.state_dict(), model_info['Save Path'])

        # Evaluation on test set
        model.eval()
        all_labels = []
        all_preds = []
        with torch.no_grad():
            for inputs, labels in dataloaders['test']:
                inputs = inputs.to(device)
                labels = labels.to(device)
                outputs = model(inputs)
                _, predicted = torch.max(outputs.data, 1)
                all_labels.extend(labels.cpu().numpy())
                all_preds.extend(predicted.cpu().numpy())

        # Compute confusion matrix
        conf_matrix = confusion_matrix(all_labels, all_preds)

        # Compute sensitivity and specificity for each class
        sens_spec = {}
        for i, class_name in enumerate(class_names):
            sensitivity, specificity = compute_sens_spec(all_labels, all_preds, pos_label=i)
            sens_spec[class_name] = {
                'Sensitivity': sensitivity,
                'Specificity': specificity
            }

        # Collect data
        model_data = {
            'Model Name': model_info['Model Name'],
            'Augmentation Method': model_info['Augmentation Method'],
            'Pretrained Model': model_info['Model Definition'],
            'Pretrained Dataset': model_info['Pretrained Dataset'],
            'Confusion Matrix': conf_matrix.flatten().tolist(),
            'Sensitivity (Normal vs Others)': sens_spec['Normal']['Sensitivity'],
            'Specificity (Normal vs Others)': sens_spec['Normal']['Specificity'],
            'Sensitivity (Benign vs Others)': sens_spec['Benign']['Sensitivity'],
            'Specificity (Benign vs Others)': sens_spec['Benign']['Specificity'],
            'Sensitivity (Malignant vs Others)': sens_spec['Malignant']['Sensitivity'],
            'Specificity (Malignant vs Others)': sens_spec['Malignant']['Specificity'],
        }

        # Calculate overall accuracy
        total_correct = np.sum(np.array(all_labels) == np.array(all_preds))
        total_samples = len(all_labels)
        accuracy = total_correct / total_samples
        model_data['Accuracy'] = accuracy

        # Append to the list
        all_models_data.append(model_data)

    # Create DataFrame and save to CSV and Excel
    df = pd.DataFrame(all_models_data)
    df.to_csv('models_evaluation_dropout_comparison.csv', index=False)
    df.to_excel('models_evaluation_dropout_comparison.xlsx', index=False)
    print("Training and evaluation completed. Results saved to 'models_evaluation_dropout_comparison.csv' and 'models_evaluation_dropout_comparison.xlsx'.")

if __name__ == '__main__':
    main()
