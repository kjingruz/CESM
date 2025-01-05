# train.py

import os
import random
import logging
import torch
import albumentations as A  # make sure installed (pip install albumentations)

from detectron2 import model_zoo
from detectron2.config import get_cfg
from detectron2.data.datasets import register_coco_instances
from detectron2.data import build_detection_test_loader
from detectron2.evaluation import COCOEvaluator, inference_on_dataset
from detectron2.engine import DefaultPredictor, DefaultTrainer
from detectron2.utils.logger import setup_logger

# Our custom MedicalAugMapper
from medical_augment import MedicalAugMapper

logger = logging.getLogger("detectron2")

class MyTrainer(DefaultTrainer):
    @classmethod
    def build_train_loader(cls, cfg):
        # Use our custom mapper with medical augmentations
        from detectron2.data import build_detection_train_loader
        return build_detection_train_loader(
            cfg,
            mapper=MedicalAugMapper(cfg, is_train=True)
        )

def setup_cfg():
    cfg = get_cfg()
    # Base config from model zoo (Faster R-CNN + FPN + ResNet-50)
    cfg.merge_from_file(model_zoo.get_config_file("COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml"))
    
    # Switch backbone depth to ResNet-34 (could do 18 as well)
    cfg.MODEL.RESNETS.DEPTH = 34
    
    # Freeze early layers to mitigate overfitting
    cfg.MODEL.BACKBONE.FREEZE_AT = 2

    # Set datasets
    cfg.DATASETS.TRAIN = ("cesm_train",)
    cfg.DATASETS.TEST = ("cesm_val",)

    # Overfitting around 5k => reduce LR, increase total iters
    cfg.SOLVER.IMS_PER_BATCH = 4
    cfg.SOLVER.BASE_LR = 1e-4
    cfg.SOLVER.MAX_ITER = 40000
    cfg.SOLVER.STEPS = []  # no step-based LR changes

    # Make checkpointing infrequent
    cfg.SOLVER.CHECKPOINT_PERIOD = 999999

    # Increase region proposal batch size per image
    cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 256

    # 3 classes: Normal (0), Benign (1), Malignant (2)
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 3

    # Keep empty images for "Normal"
    cfg.DATALOADER.FILTER_EMPTY_ANNOTATIONS = False

    # Evaluate more frequently (every 1000 iters)
    cfg.TEST.EVAL_PERIOD = 1000

    cfg.OUTPUT_DIR = "./output_resnet34_augs"
    os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
    return cfg

def main():
    setup_logger()
    logger.info("Starting training with a custom backbone & medical augmentations...")

    # 1) Register COCO datasets
    data_dir = './coco_annotations'
    image_dir = './images'
    if not os.path.isdir(data_dir):
        logger.error(f"Annotation directory not found: {data_dir}")
        return
    if not os.path.isdir(image_dir):
        logger.error(f"Image directory not found: {image_dir}")
        return

    register_coco_instances("cesm_train", {}, os.path.join(data_dir, "train_annotations.json"), image_dir)
    register_coco_instances("cesm_val", {}, os.path.join(data_dir, "val_annotations.json"), image_dir)
    register_coco_instances("cesm_test", {}, os.path.join(data_dir, "test_annotations.json"), image_dir)

    # 2) Set up cfg
    cfg = setup_cfg()

    # 3) Create trainer & train
    trainer = MyTrainer(cfg)
    trainer.resume_or_load(resume=False)
    trainer.train()

    # 4) Evaluate on the validation set
    evaluator = COCOEvaluator("cesm_val", cfg, False, output_dir=cfg.OUTPUT_DIR)
    val_loader = build_detection_test_loader(cfg, "cesm_val")
    results = inference_on_dataset(trainer.model, val_loader, evaluator)
    logger.info(f"Final evaluation results: {results}")

    # Optionally, evaluate on test set too
    # test_evaluator = COCOEvaluator("cesm_test", cfg, False, output_dir=cfg.OUTPUT_DIR)
    # test_loader = build_detection_test_loader(cfg, "cesm_test")
    # test_results = inference_on_dataset(trainer.model, test_loader, test_evaluator)
    # logger.info(f"Test evaluation results: {test_results}")

if __name__ == "__main__":
    main()
