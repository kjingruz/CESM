import os
import torch
import json
import logging
import datetime
import matplotlib.pyplot as plt
import pandas as pd
from collections import defaultdict
from sklearn.metrics import classification_report
from torchvision.ops import box_iou
import numpy as np
import cv2
import re
import random
from skimage import io

from detectron2 import model_zoo
from detectron2.config import get_cfg
from detectron2.engine import DefaultTrainer, HookBase, DefaultPredictor
from detectron2.evaluation import COCOEvaluator, inference_on_dataset
from detectron2.data.datasets import register_coco_instances
from detectron2.data import (
    DatasetMapper,
    build_detection_test_loader,
    build_detection_train_loader,
    MetadataCatalog,
    DatasetCatalog
)
from detectron2.utils.visualizer import Visualizer, ColorMode
import detectron2.data.transforms as T
import albumentations as A
from detectron2.solver import WarmupMultiStepL

from detectron2.data.detection_utils import read_image
from detectron2.structures import BoxMode
from detectron2.data import detection_utils as utils
from detectron2.modeling import build_model
import torch.nn as nn
import torch.nn.functional as F
from detectron2.utils.events import EventStorage
from detectron2.utils.logger import log_every_n_seconds
from detectron2.utils import comm

# Fix for np.bool deprecation
if not hasattr(np, 'bool') or not isinstance(np.bool, type):
    np.bool = bool

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ===========================
# Custom Evaluator Definition
# ===========================

class CustomEvaluator(COCOEvaluator):
    def __init__(self, dataset_name, cfg, distributed, output_dir=None, iou_threshold=0.5):
        super().__init__(dataset_name, cfg, distributed, output_dir)
        self._metadata = MetadataCatalog.get(dataset_name)
        self._cpu_device = torch.device("cpu")
        self.iou_threshold = iou_threshold
        # Collect predictions and ground truths
        self._predictions_list = []
        self._gt_annotations_list = []
        self._custom_metrics = None  # Store custom metrics here

    def process(self, inputs, outputs):
        super().process(inputs, outputs)
        for input, output in zip(inputs, outputs):
            # Ground truth annotations
            if "instances" in input:
                gt_instances = input["instances"].to(self._cpu_device)
                gt_boxes = gt_instances.gt_boxes.tensor.numpy()
                gt_classes = gt_instances.gt_classes.numpy()
            else:
                # If ground truth instances are not available, skip
                logger.warning(f"No ground truth instances found for input {input.get('image_id', 'Unknown')}")
                gt_boxes = np.array([])
                gt_classes = np.array([])

            gt_annotations = [{"bbox": bbox, "category_id": cls} for bbox, cls in zip(gt_boxes, gt_classes)]
            self._gt_annotations_list.append(gt_annotations)

            # Predicted instances
            pred_instances = output["instances"].to(self._cpu_device)
            pred_boxes = pred_instances.pred_boxes.tensor.numpy()
            pred_classes = pred_instances.pred_classes.numpy()
            scores = pred_instances.scores.numpy()
            predictions = [{"bbox": bbox, "category_id": cls, "score": score} for bbox, cls, score in zip(pred_boxes, pred_classes, scores)]
            self._predictions_list.append(predictions)

    def evaluate(self):
        coco_results = super().evaluate()
        # Compute custom metrics if needed (e.g., AP per category)
        # For simplicity, we'll focus on COCO metrics (AP)

        return coco_results  # Return only standard COCO results

# ===============================
# Loss Evaluation Hook Definition
# ===============================

class LossEvalHook(HookBase):
    def __init__(self, eval_period, model, data_loader, trainer):
        self._model = model
        self._period = eval_period
        self._data_loader = data_loader
        self.trainer = trainer
        self.best_val_loss = float('inf')

    def _do_loss_eval(self):
        self._model.eval()
        total_val_loss = 0.0
        count = 0
        with torch.no_grad():
            for inputs in self._data_loader:
                outputs = self._model(inputs)
                losses = outputs["losses"]
                if isinstance(losses, dict):
                    loss = sum(loss for loss in losses.values())
                elif isinstance(losses, list):
                    loss = sum(loss for loss in losses)
                else:
                    logger.error(f"Unexpected type for losses: {type(losses)}")
                    loss = 0.0
                total_val_loss += loss.item()
                count += 1
        mean_val_loss = total_val_loss / count if count > 0 else 0.0
        self.trainer.storage.put_scalar('validation_loss', mean_val_loss)
        self._model.train()
        logger.info(f"Validation Loss: {mean_val_loss}")

        # Implement Early Stopping: Stop if val_loss > best_val_loss
        if mean_val_loss > self.best_val_loss:
            logger.info("Validation loss increased. Stopping training to prevent overfitting.")
            raise KeyboardInterrupt("Early stopping triggered: Validation loss increased.")
        else:
            self.best_val_loss = mean_val_loss

    def after_step(self):
        if self.trainer.iter % self._period == 0 and self.trainer.iter != 0:
            self._do_loss_eval()

# ==============================
# Trainer Class Definition
# ==============================

class Trainer(DefaultTrainer):
    @classmethod
    def build_evaluator(cls, cfg, dataset_name, output_dir=None):
        return CustomEvaluator(dataset_name, cfg, False, output_dir)

    @classmethod
    def build_train_loader(cls, cfg):
        return build_detection_train_loader(
            cfg,
            mapper=DatasetMapper(
                cfg,
                is_train=True,
                augmentations=[
                    # Add desired augmentations here
                    T.RandomFlip(prob=0.5, horizontal=True, vertical=False),
                    T.RandomRotation(angle=[-10, 10], expand=False),
                    T.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, brightness_by_max=True),
                ],
                image_format=cfg.INPUT.FORMAT
            )
        )

    def build_hooks(self):
        hooks = super().build_hooks()
        # Insert LossEvalHook before the default evaluation hook
        hooks.insert(-1, LossEvalHook(
            eval_period=self.cfg.TEST.EVAL_PERIOD,
            model=self.model,
            data_loader=build_detection_test_loader(
                self.cfg,
                self.cfg.DATASETS.TEST[0],
                mapper=DatasetMapper(
                    self.cfg,
                    is_train=False,
                    augmentations=[],  # No augmentations for validation
                    image_format=self.cfg.INPUT.FORMAT,
                    use_instance_mask=self.cfg.MODEL.MASK_ON,
                )
            ),
            trainer=self
        ))
        return hooks

# ===============================
# Configuration Setup Function
# ===============================

def setup_cfg(config_name, lr, max_iter, batch_size, num_classes):
    cfg = get_cfg()
    # Use the appropriate config file
    cfg_file = f"COCO-Detection/{config_name}.yaml"
    cfg.merge_from_file(model_zoo.get_config_file(cfg_file))
    cfg.DATASETS.TRAIN = ("cesm_train",)
    cfg.DATASETS.TEST = ("cesm_val",)
    cfg.DATALOADER.NUM_WORKERS = 2
    cfg.SOLVER.IMS_PER_BATCH = batch_size
    cfg.SOLVER.BASE_LR = lr
    cfg.SOLVER.MAX_ITER = max_iter
    cfg.SOLVER.STEPS = []  # No learning rate decay
    cfg.SOLVER.GAMMA = 0.1
    cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 128
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.05
    cfg.INPUT.MIN_SIZE_TRAIN = (800,)
    cfg.INPUT.MAX_SIZE_TRAIN = 1333
    cfg.INPUT.MIN_SIZE_TEST = 800
    cfg.INPUT.MAX_SIZE_TEST = 1333
    cfg.INPUT.RANDOM_FLIP = "horizontal"  # Data augmentation
    cfg.TEST.EVAL_PERIOD = 1000  # Evaluate every 1000 iterations

    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    cfg.OUTPUT_DIR = f"./output_{config_name}_{lr}_{max_iter}_{batch_size}_{timestamp}"
    os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)

    # Set MODEL.WEIGHTS to the pre-trained model's URL
    cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(cfg_file)

    # Set the number of classes correctly based on the model
    if 'retinanet' in config_name.lower():
        cfg.MODEL.RETINANET.NUM_CLASSES = num_classes
    else:
        cfg.MODEL.ROI_HEADS.NUM_CLASSES = num_classes

    return cfg

# ===================================
# Plotting Metrics Function
# ===================================

def plot_metrics(output_dir):
    metrics_file = os.path.join(output_dir, "metrics.json")
    if not os.path.exists(metrics_file):
        logger.warning(f"No metrics.json found in {output_dir}")
        return [], [], [], [], [], []
    with open(metrics_file, "r") as f:
        metrics = [json.loads(line.strip()) for line in f]

    # Convert metrics to DataFrame
    metrics_df = pd.DataFrame(metrics)

    # Sort by iteration
    metrics_df = metrics_df.sort_values("iteration")

    # Extract training loss
    train_loss = metrics_df[~metrics_df["total_loss"].isna()]
    iterations_train = train_loss["iteration"]
    losses_train = train_loss["total_loss"]

    # Extract validation loss
    val_loss = metrics_df[~metrics_df["validation_loss"].isna()]
    iterations_val = val_loss["iteration"]
    losses_val = val_loss["validation_loss"]

    # Extract AP values
    ap = metrics_df[~metrics_df["bbox/AP"].isna()]
    iterations_ap = ap["iteration"]
    ap_values = ap["bbox/AP"]

    # Plot training and validation loss
    plt.figure(figsize=(10, 5))
    plt.plot(iterations_train, losses_train, label="Training Loss")
    if not losses_val.empty:
        plt.plot(iterations_val, losses_val, label="Validation Loss")
    plt.xlabel("Iteration")
    plt.ylabel("Loss")
    plt.title("Training and Validation Loss")
    plt.legend()
    plt.savefig(os.path.join(output_dir, "loss_curve.png"))
    plt.close()

    # Plot AP over iterations
    if not ap_values.empty:
        plt.figure(figsize=(10, 5))
        plt.plot(iterations_ap, ap_values, label="Average Precision (AP)")
        plt.xlabel("Iteration")
        plt.ylabel("AP")
        plt.title("Average Precision over Iterations")
        plt.legend()
        plt.savefig(os.path.join(output_dir, "ap_curve.png"))
        plt.close()

    return iterations_train, losses_train, iterations_val, losses_val, iterations_ap, ap_values

# ============================
# Training and Evaluation Function
# ============================

def train_and_evaluate(cfg, dataset_info, config_name):
    trainer = Trainer(cfg)
    trainer.resume_or_load(resume=False)

    # Ensure all layers are trainable
    for param in trainer.model.parameters():
        param.requires_grad = True

    try:
        # Start training
        trainer.train()
    except KeyboardInterrupt as e:
        logger.info("Training interrupted: Early stopping triggered.")

    # Plot metrics
    iterations_train, losses_train, iterations_val, losses_val, iterations_ap, ap_values = plot_metrics(cfg.OUTPUT_DIR)

    # Evaluate on test set
    cfg.MODEL.WEIGHTS = os.path.join(cfg.OUTPUT_DIR, "model_final.pth")
    cfg.DATASETS.TEST = ("cesm_test", )
    predictor = DefaultPredictor(cfg)
    evaluator = CustomEvaluator("cesm_test", cfg, False, output_dir=cfg.OUTPUT_DIR)
    test_loader = build_detection_test_loader(cfg, "cesm_test")
    results = inference_on_dataset(trainer.model, test_loader, evaluator)

    final_ap = results.get('bbox', {}).get('AP', 0.0)

    result_entry = {
        'Pretrained Model Used': config_name,
        'Loss Function Name': 'Default',
        'Framework': 'Detectron2',
        'Early Stopping': 'Yes' if isinstance(e, KeyboardInterrupt) else 'No',
        'Test AP': final_ap,
        'Key Hyperparameters': {
            'Learning Rate': cfg.SOLVER.BASE_LR,
            'Batch Size': cfg.SOLVER.IMS_PER_BATCH,
            'Max Iterations': cfg.SOLVER.MAX_ITER
        },
        'Augmentation Method': cfg.INPUT.RANDOM_FLIP,
        'Dataset Info': dataset_info,
    }

    return results, iterations_train, losses_train, iterations_val, losses_val, iterations_ap, ap_values, result_entry

# ===========================
# Main Function
# ===========================

def main():
    # ====================
    # Dataset Registration
    # ====================
    register_coco_instances("cesm_train", {}, "./output/train_annotations.json", "../data/images")
    register_coco_instances("cesm_val", {}, "./output/val_annotations.json", "../data/images")
    register_coco_instances("cesm_test", {}, "./output/test_annotations.json", "../data/images")

    # ======================
    # Dataset Information
    # ======================
    # Update these numbers based on your actual dataset splits
    num_normal_test = 50
    num_benign_test = 30
    num_malignant_test = 20

    dataset_info = {
        'Train/Val/Test Split': '70/15/15',
        'Num Normal in Test Set': num_normal_test,
        'Num Benign in Test Set': num_benign_test,
        'Num Malignant in Test Set': num_malignant_test,
    }

    # =====================
    # Configuration to Use
    # =====================
    # We'll train a single model: Faster R-CNN with ResNet-50-FPN backbone
    config_name = "faster_rcnn_R_50_FPN_3x"
    lr = 0.0001
    max_iter = 10000
    batch_size = 2
    num_classes = 3  # Benign, Malignant, Normal

    cfg = setup_cfg(config_name, lr, max_iter, batch_size, num_classes)

    results_data = []
    best_ap = 0
    best_config = None
    best_model_path = None

    logger.info(f"\nTraining with config: {config_name}, lr: {lr}, max_iter: {max_iter}, batch_size: {batch_size}")

    results, iterations_train, losses_train, iterations_val, losses_val, iterations_ap, ap_values, result_entry = train_and_evaluate(cfg, dataset_info, config_name)

    final_ap = result_entry['Test AP']
    if final_ap > best_ap:
        best_ap = final_ap
        best_config = (config_name, lr, max_iter, batch_size)
        best_model_path = os.path.join(cfg.OUTPUT_DIR, "model_final.pth")
    else:
        # Optionally remove non-best models to save space
        model_path = os.path.join(cfg.OUTPUT_DIR, "model_final.pth")
        if os.path.exists(model_path):
            os.remove(model_path)
            logger.info(f"Removed non-best model at {model_path}")

    results_data.append(result_entry)

    # =====================
    # Save Results to CSV
    # =====================
    # Convert the list of dictionaries to a DataFrame
    df = pd.DataFrame(results_data)
    # Normalize the 'Key Hyperparameters' dictionary into separate columns
    hyperparams_df = pd.json_normalize(df['Key Hyperparameters'])
    # Combine with the main DataFrame
    final_df = pd.concat([df.drop('Key Hyperparameters', axis=1), hyperparams_df], axis=1)
    # Save to CSV
    csv_path = 'model_comparison.csv'
    final_df.to_csv(csv_path, index=False)
    logger.info(f"\nResults saved to {csv_path}")
    logger.info(f"Best configuration: {best_config}")
    logger.info(f"Best AP: {best_ap}")
    logger.info(f"Best model saved at: {best_model_path}")

    # =====================
    # Plot Metrics
    # =====================
    plot_metrics(cfg.OUTPUT_DIR)

    # =====================
    # Visualize Test Predictions
    # =====================
    if best_model_path:
        # Visualize sample predictions
        cfg.MODEL.WEIGHTS = best_model_path
        cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5  # Set a custom testing threshold for this model
        cfg.DATASETS.TEST = ("cesm_test", )
        predictor = DefaultPredictor(cfg)

        test_dataset_dicts = DatasetCatalog.get("cesm_test")
        test_metadata = MetadataCatalog.get("cesm_test")

        # Ensure there are enough samples
        num_samples = min(4, len(test_dataset_dicts))
        if num_samples == 0:
            logger.warning("No samples available in the test dataset for visualization.")
            return

        # Select random samples
        sample_dicts = random.sample(test_dataset_dicts, num_samples)

        # Create subplots
        fig, axes = plt.subplots(1, num_samples, figsize=(5*num_samples, 5))
        if num_samples == 1:
            axes = [axes]

        for ax, d in zip(axes, sample_dicts):
            image = io.imread(d["file_name"])
            outputs = predictor(image)
            v = Visualizer(image[:, :, ::-1],
                           metadata=test_metadata, 
                           scale=0.5, 
                           instance_mode=ColorMode.IMAGE_BW   # Remove the colors of unsegmented pixels. This option is only available for segmentation models
                           )
            out = v.draw_instance_predictions(outputs["instances"].to("cpu"))
            ax.imshow(out.get_image()[:, :, ::-1])
            ax.axis('off')
            ax.set_title(d["file_name"].split('/')[-1])

        plt.tight_layout()
        viz_path = os.path.join(cfg.OUTPUT_DIR, "test_predictions.png")
        plt.savefig(viz_path)
        plt.close()
        logger.info(f"Test predictions visualized and saved to {viz_path}")

if __name__ == "__main__":
    main()
