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

class MedicalAugMapper:
    """
    A custom mapper that:
      1) Loads an image.
      2) Applies Albumentations transforms.
      3) Converts bounding boxes from XYWH->XYXY->XYWH, storing them as 'annotations'.
      4) Creates a Detectron2 `Instances` object for RPN training.
      5) Returns dataset_dict with 'image' as a torch.Tensor and 'instances' as an Instances object.
    """

    def __init__(self, cfg, is_train=True):
        self.is_train = is_train

        # Example Albumentations pipeline:
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
        """
        Args:
            dataset_dict (dict): Metadata for one image, such as file_name, height, width, and annotations.
        Returns:
            dict: Updated dict with:
                "image": a CHW float tensor,
                "instances": a Detectron2 Instances object with ground-truth.
        """
        dataset_dict = dataset_dict.copy()

        # 1) Load image as HWC BGR or RGB
        image = utils.read_image(dataset_dict["file_name"], format="RGB")  # (H, W, C)

        # 2) Convert Detectron2 XYWH -> Albumentations XYXY
        annos = dataset_dict.get("annotations", [])
        bboxes = []
        category_ids = []
        for anno in annos:
            x, y, w, h = anno["bbox"]  # XYWH
            x2 = x + w
            y2 = y + h
            bboxes.append([x, y, x2, y2])
            category_ids.append(anno["category_id"])

        # 3) Apply Albumentations
        transformed = self.transform(
            image=image,
            bboxes=bboxes,
            category_ids=category_ids
        )
        aug_img = transformed["image"]  # HWC
        aug_bboxes = transformed["bboxes"]
        aug_cats = transformed["category_ids"]

        # 4) Convert Albumentations XYXY -> Detectron2 XYWH
        new_annos = []
        for box, cat in zip(aug_bboxes, aug_cats):
            x_min, y_min, x_max, y_max = box
            w = x_max - x_min
            h = y_max - y_min
            # Filter out invalid boxes
            if w <= 1 or h <= 1:
                continue
            new_annos.append({
                "bbox": [x_min, y_min, w, h],
                "bbox_mode": BoxMode.XYWH_ABS,
                "category_id": cat,
            })

        # 5) Convert image to torch.Tensor (CHW, float32)
        aug_img_torch = torch.as_tensor(
            aug_img.transpose(2, 0, 1), dtype=torch.float32
        )

        # 6) Build "instances" from new_annos
        #    This is crucial so RPN can see gt_instances
        image_shape = aug_img_torch.shape[1:]  # (H, W)
        instances = create_instances(new_annos, image_shape)

        # 7) Update dataset_dict
        dataset_dict["image"] = aug_img_torch
        dataset_dict["instances"] = instances

        return dataset_dict
