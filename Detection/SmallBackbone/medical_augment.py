# medical_augment.py
import albumentations as A
import cv2
import numpy as np

from detectron2.data import detection_utils as utils
from detectron2.structures import BoxMode

class MedicalAugMapper:
    """
    A custom mapper that applies medical-oriented augmentations using albumentations.
    Expects bounding boxes in Detectron2 XYWH format and converts them for Albumentations.
    """
    def __init__(self, cfg, is_train=True):
        self.is_train = is_train

        # Example albumentations pipeline
        # Adjust or add transforms as needed for your domain
        self.transform = A.Compose([
            # 1) Random flips
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.1),

            # 2) Random small rotation
            A.Rotate(limit=10, border_mode=cv2.BORDER_CONSTANT, p=0.3),

            # 3) CLAHE to enhance contrast
            A.CLAHE(clip_limit=2.0, tile_grid_size=(8,8), p=0.3),

            # 4) Random brightness/contrast
            A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.3),

            # 5) Optional: Gaussian blur
            A.GaussianBlur(blur_limit=3, p=0.1),

        ], bbox_params=A.BboxParams(
            format='pascal_voc',  # We'll convert XYWH -> XYXY for Albumentations
            label_fields=['category_ids']
        ))

    def __call__(self, dataset_dict):
        dataset_dict = dataset_dict.copy()

        # Load image using detectron2's utility
        image = utils.read_image(dataset_dict["file_name"], format="RGB")

        # Convert D2 XYWH -> Albumentations XYXY
        annos = dataset_dict.get("annotations", [])
        bboxes = []
        category_ids = []
        for anno in annos:
            bbox = anno["bbox"]  # XYWH
            x_min = bbox[0]
            y_min = bbox[1]
            x_max = bbox[0] + bbox[2]
            y_max = bbox[1] + bbox[3]
            bboxes.append([x_min, y_min, x_max, y_max])
            category_ids.append(anno["category_id"])

        # Apply Albumentations transforms
        transformed = self.transform(
            image=image,
            bboxes=bboxes,
            category_ids=category_ids
        )
        aug_img = transformed["image"]
        aug_bboxes = transformed["bboxes"]
        aug_category_ids = transformed["category_ids"]

        # Convert back to Detectron2 XYWH
        new_annos = []
        for box, cls in zip(aug_bboxes, aug_category_ids):
            x_min, y_min, x_max, y_max = box
            new_w = x_max - x_min
            new_h = y_max - y_min
            if new_w <= 1 or new_h <= 1:
                # Filter out invalid boxes (e.g., due to strong augmentations)
                continue
            new_annos.append({
                "bbox": [x_min, y_min, new_w, new_h],
                "bbox_mode": BoxMode.XYWH_ABS,
                "category_id": cls,
            })

        dataset_dict["annotations"] = new_annos

        # Albumentations returns HWC, detectron2 expects CHW in a torch.Tensor
        dataset_dict["image"] = np.ascontiguousarray(aug_img.transpose(2, 0, 1))
        return dataset_dict
