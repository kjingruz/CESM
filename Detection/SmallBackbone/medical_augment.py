# medical_augment.py

import albumentations as A
import cv2
import numpy as np
import torch

from detectron2.data import detection_utils as utils
from detectron2.structures import BoxMode

class MedicalAugMapper:
    """
    A custom mapper that applies medical-oriented augmentations using albumentations.
    Expects bounding boxes in Detectron2 XYWH format and converts them for Albumentations.
    Then converts them back to XYWH and returns a torch.Tensor for 'image'.
    """

    def __init__(self, cfg, is_train=True):
        self.is_train = is_train

        # Example albumentations pipeline: adjust or add transforms for your domain
        self.transform = A.Compose([
            # 1) Random flips
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.1),

            # 2) Random rotation
            A.Rotate(limit=10, border_mode=cv2.BORDER_CONSTANT, p=0.3),

            # 3) CLAHE to enhance contrast
            A.CLAHE(clip_limit=2.0, tile_grid_size=(8,8), p=0.3),

            # 4) Random brightness/contrast
            A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.3),

            # 5) Optional: Gaussian blur
            A.GaussianBlur(blur_limit=3, p=0.1),
        ],
        bbox_params=A.BboxParams(
            format='pascal_voc',  # We'll convert XYWH -> XYXY for Albumentations
            label_fields=['category_ids']
        ))

    def __call__(self, dataset_dict):
        dataset_dict = dataset_dict.copy()

        # 1) Load image
        image = utils.read_image(dataset_dict["file_name"], format="RGB")  # shape: (H, W, C)

        # 2) Convert XYWH -> XYXY for each annotation
        annos = dataset_dict.get("annotations", [])
        bboxes = []
        category_ids = []
        for anno in annos:
            bbox = anno["bbox"]  # [x, y, w, h]
            x_min = bbox[0]
            y_min = bbox[1]
            x_max = x_min + bbox[2]
            y_max = y_min + bbox[3]
            bboxes.append([x_min, y_min, x_max, y_max])
            category_ids.append(anno["category_id"])

        # 3) Apply Albumentations
        transformed = self.transform(
            image=image,
            bboxes=bboxes,
            category_ids=category_ids
        )
        aug_img = transformed["image"]    # (H, W, C)
        aug_bboxes = transformed["bboxes"]
        aug_category_ids = transformed["category_ids"]

        # 4) Convert bounding boxes back to XYWH
        new_annos = []
        for box, cls_id in zip(aug_bboxes, aug_category_ids):
            x_min, y_min, x_max, y_max = box
            w = x_max - x_min
            h = y_max - y_min
            # Filter out invalid boxes if they've collapsed to <1 pixel
            if w <= 1 or h <= 1:
                continue
            new_annos.append({
                "bbox": [x_min, y_min, w, h],
                "bbox_mode": BoxMode.XYWH_ABS,
                "category_id": cls_id
            })

        dataset_dict["annotations"] = new_annos

        # 5) Convert the augmented NumPy image to a torch.Tensor
        #    Detectron2 expects CHW format in float32
        aug_img_torch = torch.as_tensor(aug_img.transpose(2, 0, 1), dtype=torch.float32)
        dataset_dict["image"] = aug_img_torch

        return dataset_dict
