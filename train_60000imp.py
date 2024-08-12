import os
import torch
import numpy as np
import albumentations as A
from detectron2 import model_zoo
from detectron2.config import get_cfg
from detectron2.engine import DefaultTrainer, HookBase
from detectron2.evaluation import COCOEvaluator
from detectron2.data import DatasetMapper, build_detection_train_loader
from detectron2.data.datasets import register_coco_instances
from detectron2.solver import build_lr_scheduler
import detectron2.data.transforms as T
from detectron2.data.detection_utils import read_image
from detectron2.structures import BoxMode
from detectron2.data import detection_utils as utils
from detectron2.modeling import build_model
import torch.nn as nn
import torch.nn.functional as F

class FocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss
        return focal_loss.mean()

class AlbumentationsMapper(DatasetMapper):
    def __init__(self, cfg, is_train=True):
        super().__init__(cfg, is_train)
        self.albu_transform = A.Compose([
            A.OneOf([
                A.RandomRotate90(),
                A.Rotate(limit=30),
            ], p=0.5),
            A.OneOf([
                A.GaussNoise(var_limit=(10.0, 50.0)),
                A.GaussianBlur(blur_limit=(3, 7)),
                A.MotionBlur(),
            ], p=0.5),
            A.OneOf([
                A.RandomBrightnessContrast(),
                A.HueSaturationValue(),
            ], p=0.5),
            A.CLAHE(clip_limit=(1, 4), tile_grid_size=(8, 8), p=0.5),
            A.RandomGamma(gamma_limit=(80, 120), p=0.5),
            A.ImageCompression(quality_lower=85, quality_upper=95, p=0.5),
        ], bbox_params=A.BboxParams(format='pascal_voc', label_fields=['category_ids']))

    def __call__(self, dataset_dict):
        dataset_dict = dataset_dict.copy()
        image = read_image(dataset_dict["file_name"], format=self.image_format)

        aug_input = T.AugInput(image)
        transforms = self.augmentations(aug_input)
        image = aug_input.image

        dataset_dict["image"] = torch.as_tensor(image.transpose(2, 0, 1).astype("float32"))

        if "annotations" in dataset_dict:
            for anno in dataset_dict["annotations"]:
                if not self.use_instance_mask:
                    anno.pop("segmentation", None)
                if not self.use_keypoint:
                    anno.pop("keypoints", None)

            annos = [
                utils.transform_instance_annotations(obj, transforms, image.shape[:2])
                for obj in dataset_dict.pop("annotations")
                if obj.get("iscrowd", 0) == 0
            ]

            bboxes = [BoxMode.convert(obj["bbox"], obj["bbox_mode"], BoxMode.XYXY_ABS) for obj in annos]
            category_ids = [obj["category_id"] for obj in annos]

            transformed = self.albu_transform(image=image, bboxes=bboxes, category_ids=category_ids)
            image = transformed['image']
            transformed_bboxes = transformed['bboxes']

            instances = utils.annotations_to_instances(annos, image.shape[:2])
            dataset_dict["instances"] = utils.filter_empty_instances(instances)

            for anno, bbox in zip(annos, transformed_bboxes):
                anno["bbox"] = bbox
                anno["bbox_mode"] = BoxMode.XYXY_ABS

        return dataset_dict

class EarlyStoppingHook(HookBase):
    def __init__(self, eval_period, patience=5, delta=0.0001):
        self.eval_period = eval_period
        self.patience = patience
        self.delta = delta
        self.best_fitness = -np.inf
        self.counter = 0

    def after_step(self):
        next_iter = self.trainer.iter + 1
        if next_iter % self.eval_period == 0:
            results = self.trainer.test(self.trainer.cfg, self.trainer.model)
            fitness = results['bbox']['AP']

            if fitness > self.best_fitness + self.delta:
                self.best_fitness = fitness
                self.counter = 0
            else:
                self.counter += 1

            if self.counter >= self.patience:
                self.trainer.stop_training = True
                print(f"Early stopping triggered at iteration {next_iter}")

class CustomTrainer(DefaultTrainer):
    @classmethod
    def build_evaluator(cls, cfg, dataset_name, output_folder=None):
        if output_folder is None:
            output_folder = os.path.join(cfg.OUTPUT_DIR, "inference")
        return COCOEvaluator(dataset_name, cfg, True, output_folder)

    @classmethod
    def build_train_loader(cls, cfg):
        mapper = AlbumentationsMapper(cfg, is_train=True)
        return build_detection_train_loader(cfg, mapper=mapper)

    @classmethod
    def build_lr_scheduler(cls, cfg, optimizer):
        return build_lr_scheduler(cfg, optimizer)

    def build_hooks(self):
        hooks = super().build_hooks()
        hooks.append(EarlyStoppingHook(eval_period=self.cfg.TEST.EVAL_PERIOD, patience=5))
        return hooks

    @classmethod
    def build_model(cls, cfg):
        model = build_model(cfg)
        # Replace CrossEntropyLoss with FocalLoss
        for name, module in model.named_modules():
            if isinstance(module, nn.CrossEntropyLoss):
                setattr(model, name, FocalLoss(alpha=0.25, gamma=2))
        return model

def setup_cfg():
    cfg = get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file("COCO-Detection/faster_rcnn_X_101_32x8d_FPN_3x.yaml"))
    cfg.DATASETS.TRAIN = ("cesm_train",)
    cfg.DATASETS.TEST = ("cesm_val",)
    cfg.DATALOADER.NUM_WORKERS = 2
    cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-Detection/faster_rcnn_X_101_32x8d_FPN_3x.yaml")
    cfg.SOLVER.IMS_PER_BATCH = 2
    cfg.SOLVER.BASE_LR = 0.0005
    cfg.SOLVER.MAX_ITER = 60000
    cfg.SOLVER.STEPS = []
    cfg.SOLVER.LR_SCHEDULER_NAME = "WarmupCosineLR"
    cfg.SOLVER.WARMUP_FACTOR = 1.0 / 1000
    cfg.SOLVER.WARMUP_ITERS = 1000
    cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 128
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 3
    cfg.TEST.EVAL_PERIOD = 1000
    cfg.SOLVER.CHECKPOINT_PERIOD = 1000
    cfg.INPUT.MIN_SIZE_TRAIN = (640, 672, 704, 736, 768, 800)
    cfg.INPUT.MAX_SIZE_TRAIN = 1333
    cfg.INPUT.MIN_SIZE_TEST = 800
    cfg.INPUT.MAX_SIZE_TEST = 1333
    cfg.MODEL.DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    cfg.MODEL.ROI_HEADS.POSITIVE_FRACTION = 0.5

    cfg.DATALOADER.SAMPLER_TRAIN = "RepeatFactorTrainingSampler"
    cfg.DATALOADER.REPEAT_THRESHOLD = 0.001

    # Multi-scale training
    cfg.INPUT.MIN_SIZE_TRAIN = (640, 672, 704, 736, 768, 800)
    cfg.INPUT.MAX_SIZE_TRAIN = 1333

    # Test-time augmentation
    cfg.TEST.AUG.ENABLED = True
    cfg.TEST.AUG.MIN_SIZES = (400, 500, 600, 700, 800, 900, 1000, 1100, 1200)
    cfg.TEST.AUG.MAX_SIZE = 4000
    cfg.TEST.AUG.FLIP = True

    cfg.OUTPUT_DIR = "./60000_improved"
    os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
    return cfg

def main():
    register_coco_instances("cesm_train", {}, "output/train_annotations.json", "../data/images")
    register_coco_instances("cesm_val", {}, "output/val_annotations.json", "../data/images")
    register_coco_instances("cesm_test", {}, "output/test_annotations.json", "../data/images")

    cfg = setup_cfg()
    trainer = CustomTrainer(cfg)
    trainer.resume_or_load(resume=False)
    trainer.train()

if __name__ == "__main__":
    main()