# small.py

import os
import logging
import torch

import wandb  # optional for logging; remove if not using
from detectron2 import model_zoo
from detectron2.config import get_cfg
from detectron2.engine import DefaultTrainer, HookBase
from detectron2.data.datasets import register_coco_instances
from detectron2.data import build_detection_test_loader
from detectron2.utils.logger import setup_logger
from detectron2.evaluation import inference_on_dataset

from detectron2.evaluation import COCOEvaluator
from froc_evaluator import FROCEvaluator
from medical_augment import MedicalAugMapper

logger = logging.getLogger("detectron2")

class MyTrainer(DefaultTrainer):
    @classmethod
    def build_train_loader(cls, cfg):
        from detectron2.data import build_detection_train_loader
        return build_detection_train_loader(
            cfg,
            mapper=MedicalAugMapper(cfg, is_train=True)
        )

class CheckpointAndValHook(HookBase):
    """
    Custom Hook to:
      1) Evaluate on val set every N iterations
      2) Save checkpoint every M iterations
    """
    def __init__(self, cfg, val_period=1000, ckpt_period=5000):
        super().__init__()
        self.cfg = cfg
        self.val_period = val_period
        self.ckpt_period = ckpt_period

    def after_step(self):
        next_iter = self.trainer.iter + 1
        # Evaluate every val_period
        if next_iter % self.val_period == 0:
            self._do_eval()
        # Save checkpoint every ckpt_period
        if next_iter % self.ckpt_period == 0:
            self._save_checkpoint()

    def _do_eval(self):
        logger.info(f"[CheckpointAndValHook] Running validation at iter={self.trainer.iter}...")
        evaluator_list = []
        # COCO evaluator for standard detection metrics
        evaluator_list.append(COCOEvaluator("cesm_val", self.cfg, False, output_dir=self.cfg.OUTPUT_DIR))
        # FROC evaluator
        evaluator_list.append(FROCEvaluator("cesm_val", iou_threshold=0.5))

        from detectron2.data import build_detection_test_loader
        val_loader = build_detection_test_loader(self.cfg, "cesm_val")
        eval_results = inference_on_dataset(self.trainer.model, val_loader, evaluator_list)
        logger.info(f"Validation results: {eval_results}")

        # If using wandb, log some metrics:
        if "bbox" in eval_results:  # from COCOEvaluator
            if "AP" in eval_results["bbox"]:
                wandb.log({"val/AP": eval_results["bbox"]["AP"]}, step=self.trainer.iter)
        if "FROC" in eval_results:
            # e.g. log FROC TPR at some FP=0.5? Or just store the curve
            froc_data = eval_results["FROC"]["froc_data"]
            # You might parse it, e.g. find TPR at FP=0.5
            best_tpr = 0.0
            for thr, tpr, fp_img in froc_data:
                if fp_img <= 0.5:
                    if tpr > best_tpr:
                        best_tpr = tpr
            wandb.log({"val/FROC_TPR@FP=0.5": best_tpr}, step=self.trainer.iter)
        logger.info("[CheckpointAndValHook] Done validation.")

    def _save_checkpoint(self):
        logger.info(f"[CheckpointAndValHook] Saving checkpoint at iter={self.trainer.iter}...")
        self.trainer.checkpointer.save(f"model_{self.trainer.iter:07d}")

def setup_cfg():
    cfg = get_cfg()
    # Base config from detectron2 model zoo
    cfg.merge_from_file(model_zoo.get_config_file("COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml"))

    # Switch to ResNet-34
    cfg.MODEL.RESNETS.DEPTH = 34
    cfg.MODEL.RESNETS.RES2_OUT_CHANNELS = 64
    cfg.MODEL.BACKBONE.FREEZE_AT = 2

    # Datasets
    cfg.DATASETS.TRAIN = ("cesm_train",)
    cfg.DATASETS.TEST = ("cesm_val",)

    # 10k iterations total
    cfg.SOLVER.MAX_ITER = 10000

    # Smaller batch size to avoid OOM
    cfg.SOLVER.IMS_PER_BATCH = 2
    cfg.SOLVER.BASE_LR = 1e-4
    cfg.SOLVER.STEPS = []
    # We won't rely on SOLVER.CHECKPOINT_PERIOD or TEST.EVAL_PERIOD here
    # We'll use our custom hook instead.

    # ROI HEADS
    cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 128
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 3  # e.g., if you're only detecting (Benign=0, Malignant=1).
    # or 3 if you're truly using Normal=0, but typically 2 is recommended.

    cfg.DATALOADER.FILTER_EMPTY_ANNOTATIONS = False

    cfg.OUTPUT_DIR = "./output_resnet34_augs"
    os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
    return cfg

def main():
    # Optional wandb
    wandb.init(project="cesm-proj", name="resnet34_froc", config={})

    setup_logger()
    logger.info("Starting training with custom checkpoint/eval schedule + FROC")

    # Register datasets
    data_dir = "./coco_annotations"
    img_dir = "./images"
    register_coco_instances("cesm_train", {}, os.path.join(data_dir, "train_annotations.json"), img_dir)
    register_coco_instances("cesm_val", {}, os.path.join(data_dir, "val_annotations.json"), img_dir)
    register_coco_instances("cesm_test", {}, os.path.join(data_dir, "test_annotations.json"), img_dir)

    cfg = setup_cfg()
    trainer = MyTrainer(cfg)

    trainer.resume_or_load(resume=False)

    # Add our custom hook that does validation every 1000 iters,
    # checkpoint every 5000 iters
    trainer.register_hooks([CheckpointAndValHook(cfg, val_period=1000, ckpt_period=5000)])

    # By default, detectron2's default hook does some steps too,
    # so reorder them so ours goes last:
    trainer._hooks = trainer._hooks[:-2] + trainer._hooks[-2:][::-1]

    trainer.train()

if __name__ == "__main__":
    main()
