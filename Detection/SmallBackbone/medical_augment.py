# medical_augment.py

import albumentations as A
import cv2
import numpy as np
import torch

from detectron2.data import detection_utils as utils
from detectron2.structures import BoxMode
from detectron2.data import transforms as T

def create_instances(annos, image_shape):
    import detectron2.data.detection_utils as d2utils
    return d2utils.annotations_to_instances(annos, image_shape)

def clip_bbox_to_image(x_min, y_min, x_max, y_max, w, h):
    x_min_cl = max(0, min(x_min, w - 1))
    y_min_cl = max(0, min(y_min, h - 1))
    x_max_cl = max(0, min(x_max, w - 1))
    y_max_cl = max(0, min(y_max, h - 1))
    return x_min_cl, y_min_cl, x_max_cl, y_max_cl

class MedicalAugMapper:
    def __init__(self, cfg, is_train=True):
        self.is_train = is_train
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
                format='pascal_voc',
                label_fields=['category_ids'],
                check_each_transform=False
            )
        )

    def __call__(self, dataset_dict):
        dataset_dict = dataset_dict.copy()
        image = utils.read_image(dataset_dict["file_name"], format="RGB")
        h, w, _ = image.shape

        annos = dataset_dict.get("annotations", [])
        bboxes = []
        category_ids = []
        for anno in annos:
            x, y, ww, hh = anno["bbox"]  # XYWH
            x2 = x + ww
            y2 = y + hh
            x_min_cl, y_min_cl, x_max_cl, y_max_cl = clip_bbox_to_image(x, y, x2, y2, w, h)
            if x_max_cl <= x_min_cl or y_max_cl <= y_min_cl:
                continue
            bboxes.append([x_min_cl, y_min_cl, x_max_cl, y_max_cl])
            category_ids.append(anno["category_id"])

        transformed = self.transform(
            image=image,
            bboxes=bboxes,
            category_ids=category_ids
        )
        aug_img = transformed["image"]
        aug_bboxes = transformed["bboxes"]
        aug_cats = transformed["category_ids"]

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

        aug_img_torch = torch.as_tensor(aug_img.transpose(2, 0, 1), dtype=torch.float32)
        instances = create_instances(new_annos, aug_img_torch.shape[1:])

        dataset_dict["image"] = aug_img_torch
        dataset_dict["instances"] = instances
        return dataset_dict
