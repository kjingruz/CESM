# dataset_builder.py
import os
import json
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
from monai.transforms import (
    Compose,
    LoadImageD,
    EnsureChannelFirstD,
    ScaleIntensityD,
    EnsureTypeD,
)
from pycocotools.coco import COCO

def detection_collate_fn(batch):
    """
    Collate function for DataLoader when dealing with detection tasks.
    Each item in `batch` is (image, target).
    We want to return two lists: [image_1, image_2, ...] and [target_1, target_2, ...].
    """
    images, targets = [], []
    for item in batch:
        images.append(item[0])
        targets.append(item[1])
    return images, targets

class MaskRCNNDataset(Dataset):
    """
    A custom Dataset that loads images using MONAI transforms
    and returns instance segmentation annotations (boxes, masks, labels)
    in PyTorch detection format.
    """
    def __init__(
        self,
        coco_json_path,
        image_dir,
        transforms=None,
        category_shift=0,
    ):
        """
        Args:
            coco_json_path: Path to the COCO annotation file with polygons.
            image_dir: Directory containing images (flattened or sub-folders, but we assume the `file_name` matches).
            transforms: Optional MONAI Compose object. If None, we define a basic pipeline.
            category_shift: If your classes are [0,1,2], but the model expects [1,2,3], set category_shift=1.
        """
        super().__init__()
        self.coco = COCO(coco_json_path)
        self.image_dir = image_dir
        self.img_ids = list(self.coco.imgs.keys())
        self.category_shift = category_shift

        # Basic transform if none provided
        if transforms is None:
            self.transforms = Compose([
                LoadImageD(keys=["image"], ensure_channel_first=True),
                ScaleIntensityD(keys=["image"]),
                EnsureTypeD(keys=["image"])
            ])
        else:
            self.transforms = transforms

    def __len__(self):
        return len(self.img_ids)

    def __getitem__(self, idx):
        img_id = self.img_ids[idx]
        img_info = self.coco.loadImgs([img_id])[0]
        file_name = img_info["file_name"]
        image_path = os.path.join(self.image_dir, file_name)

        # 1) Apply MONAI transforms to load the image
        data_dict = {"image": image_path}
        data_out = self.transforms(data_dict)  # or self.transforms if you pass in data_dict
        image = data_out["image"]  # shape [C, H, W] as a torch.Tensor

        ann_ids = self.coco.getAnnIds(imgIds=[img_id])
        anns = self.coco.loadAnns(ann_ids)

        boxes = []
        labels = []
        masks_list = []

        _, h, w = image.shape  # from [C, H, W]

        for ann in anns:
            cat_id = ann["category_id"] + self.category_shift
            labels.append(cat_id)

            # bounding box => [x, y, w, h]
            x, y, bw, bh = ann["bbox"]
            boxes.append([x, y, x + bw, y + bh])

            # Polygon segmentation => fill in a mask
            seg = ann["segmentation"]
            mask = np.zeros((h, w), dtype=np.uint8)
            if isinstance(seg, list):
                for poly in seg:
                    poly_pts = np.array(poly, dtype=np.int32).reshape(-1,2)
                    cv2.fillPoly(mask, [poly_pts], 1)
            masks_list.append(mask)

        if len(anns) == 0:
            # no instances
            boxes_np = np.zeros((0,4), dtype=np.float32)
            labels_np = np.zeros((0,), dtype=np.int64)
            masks_np  = np.zeros((0,h,w), dtype=np.uint8)
        else:
            boxes_np = np.array(boxes, dtype=np.float32)
            labels_np = np.array(labels, dtype=np.int64)
            masks_np = np.stack(masks_list, axis=0)

        boxes_t  = torch.as_tensor(boxes_np, dtype=torch.float32)
        labels_t = torch.as_tensor(labels_np, dtype=torch.int64)
        masks_t  = torch.as_tensor(masks_np, dtype=torch.uint8)

        target = {
            "boxes": boxes_t,
            "labels": labels_t,
            "masks": masks_t
        }

        return image, target
