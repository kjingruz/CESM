# train_maskrcnn.py
import os
import torch
from torch.utils.data import DataLoader
from dataset_builder import MaskRCNNDataset, detection_collate_fn
from model_factory import create_maskrcnn_model

def train_loop(model, train_loader, optimizer, device="cuda"):
    model.train()
    total_loss = 0
    for images, targets in train_loader:
        # images: list of [C,H,W] Tensors
        # targets: list of dicts
        images = [img.to(device) for img in images]
        targets = [{k: v.to(device) for k,v in t.items()} for t in targets]

        optimizer.zero_grad()
        loss_dict = model(images, targets)
        losses = sum(loss for loss in loss_dict.values())
        losses.backward()
        optimizer.step()
        total_loss += losses.item()
    return total_loss / len(train_loader)

def main():
    # 1. Paths
    data_dir = "./images"
    coco_train_json = "./coco_annotations/train_annotations.json"
    coco_val_json   = "./coco_annotations/val_annotations.json"

    # 2. Create dataset/dataloader for training
    train_dataset = MaskRCNNDataset(
        coco_json_path=coco_train_json,
        image_dir=data_dir,
        transforms=None,       # or define a MONAI Compose
        category_shift=0       # e.g. if your cat IDs are [0,1,2] and you want to keep them
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=2,
        shuffle=True,
        collate_fn=detection_collate_fn
    )

    # Optionally create a val dataset/loader for future validation
    # ...

    # 3. Create model
    num_classes = 3  # e.g. (background=0, benign=1, malignant=2)
    model = create_maskrcnn_model(num_classes=num_classes, pretrained=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)

    # 4. Create optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    # 5. Training loop
    num_epochs = 5
    for epoch in range(num_epochs):
        avg_loss = train_loop(model, train_loader, optimizer, device=device)
        print(f"Epoch {epoch+1}/{num_epochs}, Train Loss: {avg_loss:.4f}")

    # 6. Save model
    os.makedirs("./output_maskrcnn", exist_ok=True)
    torch.save(model.state_dict(), "./output_maskrcnn/model_final.pth")

if __name__ == "__main__":
    main()
