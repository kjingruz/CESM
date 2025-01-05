# medical_augment.py

import albumentations as A
import cv2
import numpy as np
import torch

from detectron2.data import detection_utils as utils
from detectron2.structures import BoxMode
from detectron2.data import transforms as T

def create_instances(annos, image_shape):
    """
    Convert a list of Detectron2-style annotations (with 'bbox', 'category_id', etc.)
    into a Detectron2 `Instances` object for training.
    image_shape: (height, width)
    """
    import detectron2.data.detection_utils as d2utils
    return d2utils.annotations_to_instances(annos, image_shape)

def clip_bbox_to_image(x_min, y_min, x_max, y_max, w, h):
    """
    Clip bounding box coordinates so they lie fully within [0, w] x [0, h].
    """
    x_min_cl = max(0, min(x_min, w - 1))
    y_min_cl = max(0, min(y_min, h - 1))
    x_max_cl = max(0, min(x_max, w - 1))
    y_max_cl = max(0, min(y_max, h - 1))
    return x_min_cl, y_min_cl, x_max_cl, y_max_cl

class MedicalAugMapper:
    """
    A custom mapper that:
      1) Loads an image.
      2) Clips bounding boxes to ensure [0, w] x [0, h].
      3) Applies Albumentations transforms.
      4) Converts bounding boxes back to Detectron2 format.
      5) Creates a Detectron2 `Instances` object for RPN training.
    """

    def __init__(self, cfg, is_train=True):
        self.is_train = is_train

        # Albumentations pipeline
        self.transform = A.Compose(
            [
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.1),
                A.Rotate(limit=10, border_mode=cv2.BORDER_CONSTANT, p=0.3),
                A.CLAHE(clip_limit=2.0, tile_grid_size=(8,8), p=0.3),
                A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.3),
                A.GaussianBlur(blur_limit=3, p=0.1),
            ],
            bbox_params=A.BboxParams(
                format='pascal_voc',         # [x_min, y_min, x_max, y_max], absolute coords
                label_fields=['category_ids'],
                min_area=0,
                min_visibility=0,
                # If bounding boxes can partially go outside after rotation, set:
                # allow_negative_coords=True  # if you prefer not to raise errors.
                # Or we rely on manual clipping below.
                check_each_transform=False
            )
        )

    def __call__(self, dataset_dict):
        dataset_dict = dataset_dict.copy()

        # 1) Load image
        image = utils.read_image(dataset_dict["file_name"], format="RGB")  # shape (H, W, C)
        h, w, _ = image.shape

        # 2) Convert from XYWH -> XYXY, then clip to [0, w] x [0, h]
        annos = dataset_dict.get("annotations", [])
        bboxes = []
        category_ids = []
        for anno in annos:
            x, y, ww, hh = anno["bbox"]  # XYWH
            x2 = x + ww
            y2 = y + hh

            # Clip before passing to Albumentations
            x_min_cl, y_min_cl, x_max_cl, y_max_cl = clip_bbox_to_image(x, y, x2, y2, w, h)
            # If the clipped box is effectively collapsed, skip it
            if x_max_cl <= x_min_cl or y_max_cl <= y_min_cl:
                continue

            bboxes.append([x_min_cl, y_min_cl, x_max_cl, y_max_cl])
            category_ids.append(anno["category_id"])

        # 3) Albumentations transforms
        transformed = self.transform(
            image=image,
            bboxes=bboxes,
            category_ids=category_ids
        )
        aug_img = transformed["image"]
        aug_bboxes = transformed["bboxes"]
        aug_cats = transformed["category_ids"]

        # 4) Convert back to XYWH in Detectron2 style
        new_annos = []
        for box, cat in zip(aug_bboxes, aug_cats):
            x_min, y_min, x_max, y_max = box
            w_box = x_max - x_min
            h_box = y_max - y_min
            if w_box <= 1 or h_box <= 1:
                continue

            new_annos.append(
                {
                    "bbox": [x_min, y_min, w_box, h_box],
                    "bbox_mode": BoxMode.XYWH_ABS,
                    "category_id": cat
                }
            )

        # 5) Convert image to torch.Tensor (CHW)
        aug_img_torch = torch.as_tensor(aug_img.transpose(2, 0, 1), dtype=torch.float32)
        image_shape = aug_img_torch.shape[1:]  # (H, W)

        # 6) Build Instances for RPN training
        instances = create_instances(new_annos, image_shape)

        # 7) Update dataset_dict
        dataset_dict["image"] = aug_img_torch
        dataset_dict["instances"] = instances

        return dataset_dict
