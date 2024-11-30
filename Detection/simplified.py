import os
import torch
import numpy as np
import logging
from detectron2 import model_zoo
from detectron2.config import get_cfg
from detectron2.engine import DefaultTrainer, DefaultPredictor
from detectron2.evaluation import COCOEvaluator, inference_on_dataset
from detectron2.data import build_detection_train_loader, build_detection_test_loader
from detectron2.data.datasets import register_coco_instances
from detectron2.utils.visualizer import Visualizer, ColorMode
from detectron2.data import DatasetMapper
from detectron2.utils import comm
from detectron2.data import detection_utils as utils
from detectron2.data import transforms as T
from detectron2.data import MetadataCatalog, DatasetCatalog

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ===============================
# Configuration Setup Function
# ===============================

def setup_cfg():
    cfg = get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file("COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml"))

    # Dataset registration
    cfg.DATASETS.TRAIN = ("cesm_train",)
    cfg.DATASETS.TEST = ("cesm_val",)  # Use validation set for evaluation during training

    cfg.DATALOADER.NUM_WORKERS = 2
    cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml")

    cfg.SOLVER.IMS_PER_BATCH = 2
    cfg.SOLVER.BASE_LR = 0.00025  # Default learning rate
    cfg.SOLVER.MAX_ITER = 10000  # Adjust based on your dataset size

    cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 128  # Default value
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 3  # 0: Benign, 1: Malignant, 2: Normal

    cfg.OUTPUT_DIR = "./output_with_normals"
    os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)

    # Include images without annotations
    cfg.DATALOADER.FILTER_EMPTY_ANNOTATIONS = False  # Important to include images without annotations

    return cfg

# ============================
# Training and Evaluation Function
# ============================

def train_and_evaluate(cfg):
    trainer = DefaultTrainer(cfg)
    trainer.resume_or_load(resume=False)
    trainer.train()

    # Evaluate on validation set
    evaluator = COCOEvaluator("cesm_val", cfg, False, output_dir=cfg.OUTPUT_DIR)
    val_loader = build_detection_test_loader(cfg, "cesm_val")
    inference_on_dataset(trainer.model, val_loader, evaluator)

    # Evaluate on test set
    evaluator_test = COCOEvaluator("cesm_test", cfg, False, output_dir=cfg.OUTPUT_DIR)
    test_loader = build_detection_test_loader(cfg, "cesm_test")
    inference_on_dataset(trainer.model, test_loader, evaluator_test)

# ===========================
# Main Execution
# ===========================

def main():
    # ====================
    # Dataset Registration
    # ====================
    data_dir = './output'  # Directory containing annotations
    image_dir = '../data/images'  # Directory containing images

    if not os.path.exists(image_dir):
        logger.error(f"Image directory not found: {image_dir}")
        return

    # Register datasets using annotations
    register_coco_instances("cesm_train", {}, os.path.join(data_dir, "train_annotations.json"), image_dir)
    register_coco_instances("cesm_val", {}, os.path.join(data_dir, "val_annotations.json"), image_dir)
    register_coco_instances("cesm_test", {}, os.path.join(data_dir, "test_annotations.json"), image_dir)

    # =====================
    # Configuration for Model
    # =====================
    cfg = setup_cfg()

    # =====================
    # Training and Evaluation
    # =====================
    train_and_evaluate(cfg)

    # =====================
    # Visualize Test Predictions
    # =====================
    cfg.MODEL.WEIGHTS = os.path.join(cfg.OUTPUT_DIR, "model_final.pth")
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5  # Set a custom testing threshold
    predictor = DefaultPredictor(cfg)

    test_dataset_dicts = DatasetCatalog.get("cesm_test")
    test_metadata = MetadataCatalog.get("cesm_test")

    # Ensure there are enough samples
    num_samples = min(4, len(test_dataset_dicts))
    if num_samples == 0:
        logger.warning("No samples available in the test dataset for visualization.")
        return

    # Select random samples
    import random
    sample_dicts = random.sample(test_dataset_dicts, num_samples)

    # Create subplots
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, num_samples, figsize=(5*num_samples, 5))
    if num_samples == 1:
        axes = [axes]

    for ax, d in zip(axes, sample_dicts):
        image_path = d["file_name"]
        if not os.path.exists(image_path):
            logger.warning(f"Image not found: {image_path}")
            continue
        image = utils.read_image(image_path, format="BGR")
        outputs = predictor(image)
        v = Visualizer(image[:, :, ::-1],
                       metadata=test_metadata,
                       scale=0.5,
                       instance_mode=ColorMode.IMAGE_BW  # Remove the colors of unsegmented pixels.
                       )
        out = v.draw_instance_predictions(outputs["instances"].to("cpu"))
        ax.imshow(out.get_image()[:, :, ::-1])
        ax.axis('off')
        ax.set_title(os.path.basename(d["file_name"]))

    plt.tight_layout()
    viz_path = os.path.join(cfg.OUTPUT_DIR, "test_predictions.png")
    plt.savefig(viz_path)
    plt.close()
    logger.info(f"Test predictions visualized and saved to {viz_path}")

if __name__ == "__main__":
    main()
