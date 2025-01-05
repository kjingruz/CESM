import os
import random
import logging
import torch

from detectron2 import model_zoo
from detectron2.config import get_cfg
from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.data import build_detection_test_loader
from detectron2.evaluation import COCOEvaluator, inference_on_dataset
from detectron2.engine import DefaultPredictor

from detectron2.engine import DefaultTrainer
from detectron2.utils.logger import setup_logger

# Our custom MedicalAugMapper
from medical_augment import MedicalAugMapper

logger = logging.getLogger("detectron2")

class MyTrainer(DefaultTrainer):
    @classmethod
    def build_train_loader(cls, cfg):
        return build_detection_train_loader(
            cfg,
            mapper=MedicalAugMapper(cfg, is_train=True)
        )

def setup_cfg():
    cfg = get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file("COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml"))
    
    # If you want to try ResNet-18 or 34:
    cfg.MODEL.RESNETS.DEPTH = 34  # pick 18 or 34
    cfg.MODEL.BACKBONE.FREEZE_AT = 2  # freeze early layers to reduce overfit

    # Your dataset
    cfg.DATASETS.TRAIN = ("cesm_train",)
    cfg.DATASETS.TEST = ("cesm_val",)

    # Overfitting is happening ~5k iters => try smaller LR, more iters:
    cfg.SOLVER.IMS_PER_BATCH = 4
    cfg.SOLVER.BASE_LR = 0.0001   # lower LR
    cfg.SOLVER.MAX_ITER = 40000   # more total steps
    cfg.SOLVER.STEPS = []         # no step-based LR changes
    cfg.SOLVER.CHECKPOINT_PERIOD = 999999

    # Increase region proposal batch size per image
    cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 256
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 3  # Normal, Benign, Malignant

    # Keep empty annotations for "normal" images
    cfg.DATALOADER.FILTER_EMPTY_ANNOTATIONS = False

    cfg.TEST.EVAL_PERIOD = 1000  # Evaluate more frequently if desired

    cfg.OUTPUT_DIR = "./output_resnet34_augs"
    os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
    return cfg

def main():
    setup_logger()
    logger.info("Starting training with a custom backbone & medical augmentations...")

    # Register your COCO sets (already done in your code)
    register_coco_instances("cesm_train", {}, "path/to/train_annotations.json", "path/to/images")
    register_coco_instances("cesm_val", {},   "path/to/val_annotations.json",   "path/to/images")

    cfg = setup_cfg()

    trainer = MyTrainer(cfg)
    trainer.resume_or_load(resume=False)
    trainer.train()

    # Evaluate on val set
    evaluator = COCOEvaluator("cesm_val", cfg, False, output_dir=cfg.OUTPUT_DIR)
    val_loader = build_detection_test_loader(cfg, "cesm_val")
    results = inference_on_dataset(trainer.model, val_loader, evaluator)
    logger.info(f"Final evaluation results: {results}")

if __name__ == "__main__":
    main()
