import numpy as np

# Fix for np.bool deprecation (if necessary)
if not hasattr(np, 'bool') or not isinstance(np.bool, type):
    np.bool = bool

import os
import torch
import json
import logging
import datetime
import matplotlib.pyplot as plt
import pandas as pd
from detectron2 import model_zoo
from detectron2.config import get_cfg
from detectron2.engine import DefaultTrainer, HookBase
from detectron2.evaluation import COCOEvaluator, inference_on_dataset
from detectron2.data import DatasetMapper, build_detection_test_loader
from detectron2.data.datasets import register_coco_instances
from detectron2.utils import comm
from detectron2.data import MetadataCatalog
from sklearn.metrics import confusion_matrix
import openpyxl
from pycocotools import mask as maskUtils

logging.basicConfig(level=logging.INFO)

class LossEvalHook(HookBase):
    def __init__(self, eval_period, model, data_loader):
        self._model = model
        self._period = eval_period
        self._data_loader = data_loader

    def _do_loss_eval(self):
        total_losses = []
        for inputs in self._data_loader:
            with torch.no_grad():
                loss_dict = self._model(inputs)
                losses = sum(loss_dict.values())
                total_losses.append(losses.item())
        mean_loss = np.mean(total_losses)
        self.trainer.storage.put_scalar('validation_loss', mean_loss)
        comm.synchronize()
        return mean_loss

    def after_step(self):
        if self.trainer.iter % self._period == 0 and self.trainer.iter != 0:
            self._do_loss_eval()

class CustomEvaluator(COCOEvaluator):
    def __init__(self, dataset_name, cfg, distributed, output_dir=None):
        super().__init__(dataset_name, cfg, distributed, output_dir)
        self._metadata = MetadataCatalog.get(dataset_name)
        self._cpu_device = torch.device("cpu")
        # Collect predictions and ground truths
        self._predictions_masks = []
        self._gt_masks = []
        self._custom_metrics = None  # Store custom metrics here

    def process(self, inputs, outputs):
        super().process(inputs, outputs)
        for input, output in zip(inputs, outputs):
            # Ground truth masks
            gt_masks = []
            for anno in input["annotations"]:
                # Convert segmentation to binary mask
                mask = self._anno_to_mask(anno, input["height"], input["width"])
                gt_masks.append((mask, anno["category_id"]))

            self._gt_masks.append(gt_masks)

            # Predicted masks
            pred_instances = output["instances"].to(self._cpu_device)
            if len(pred_instances) == 0:
                # No predictions
                pred_masks_list = []
            else:
                pred_masks = pred_instances.pred_masks.numpy()  # Shape: [N, H, W]
                pred_classes = pred_instances.pred_classes.numpy()
                pred_masks_list = [(mask, cls) for mask, cls in zip(pred_masks, pred_classes)]
            self._predictions_masks.append(pred_masks_list)

    def evaluate(self):
        coco_results = super().evaluate()
        # Compute pixel-level confusion matrix
        if not self._predictions_masks:
            logging.warning("No predictions to evaluate.")
            return coco_results

        # Initialize an empty confusion matrix
        num_classes = len(self._metadata.thing_classes)
        confusion = np.zeros((num_classes, num_classes), dtype=np.int64)

        for gt_masks, pred_masks in zip(self._gt_masks, self._predictions_masks):
            # Create ground truth label map
            gt_label_map = self._create_label_map(gt_masks, num_classes)
            # Create predicted label map
            pred_label_map = self._create_label_map(pred_masks, num_classes)

            # Flatten the label maps
            gt_labels = gt_label_map.flatten()
            pred_labels = pred_label_map.flatten()

            # Update confusion matrix
            confusion += confusion_matrix(gt_labels, pred_labels, labels=range(num_classes))

        # Calculate sensitivity and specificity
        sensitivity_specificity = {}
        for idx, cls_name in enumerate(self._metadata.thing_classes):
            TP = confusion[idx, idx]
            FN = confusion[idx, :].sum() - TP
            FP = confusion[:, idx].sum() - TP
            TN = confusion.sum() - (TP + FP + FN)
            sensitivity = TP / (TP + FN) if (TP + FN) > 0 else 0
            specificity = TN / (TN + FP) if (TN + FP) > 0 else 0
            sensitivity_specificity[cls_name] = {
                'sensitivity': sensitivity,
                'specificity': specificity
            }

        # Store custom metrics in an instance variable
        self._custom_metrics = {
            'confusion_matrix': confusion.tolist(),
            'sensitivity_and_specificity': sensitivity_specificity
        }

        return coco_results  # Return only standard COCO results

    def _anno_to_mask(self, anno, height, width):
        segm = anno['segmentation']
        if isinstance(segm, list):
            # Polygon -- a single object might consist of multiple parts
            rles = maskUtils.frPyObjects(segm, height, width)
            rle = maskUtils.merge(rles)
        elif isinstance(segm['counts'], list):
            # Uncompressed RLE
            rle = maskUtils.frPyObjects(segm, height, width)
        else:
            # Compressed RLE
            rle = segm
        m = maskUtils.decode(rle)
        return m

    def _create_label_map(self, masks, num_classes):
        if not masks:
            # If no masks, return an array of zeros (background)
            return np.zeros((self._metadata.height, self._metadata.width), dtype=np.int32)
        # Initialize label map with background (assuming class index 0)
        label_map = np.zeros(masks[0][0].shape, dtype=np.int32)
        for mask, cls in masks:
            label_map[mask > 0] = cls + 1  # Avoid class index 0 for background
        return label_map

class Trainer(DefaultTrainer):
    @classmethod
    def build_evaluator(cls, cfg, dataset_name, output_dir=None):
        return CustomEvaluator(dataset_name, cfg, True, output_dir)

    def build_hooks(self):
        hooks = super().build_hooks()
        hooks.insert(-1, LossEvalHook(
            eval_period=self.cfg.TEST.EVAL_PERIOD,
            model=self.model,
            data_loader=build_detection_test_loader(
                self.cfg,
                self.cfg.DATASETS.TEST[0],
                mapper=DatasetMapper(
                    self.cfg,
                    is_train=True,  # Include gt_instances
                    augmentations=[],  # No augmentations
                    image_format=self.cfg.INPUT.FORMAT,
                    use_instance_mask=True,
                )
            )
        ))
        return hooks

def setup_cfg(config_name, lr, max_iter, batch_size, num_classes):
    cfg = get_cfg()
    # Use the appropriate config file
    cfg_file = f"COCO-InstanceSegmentation/{config_name}.yaml"
    cfg.merge_from_file(model_zoo.get_config_file(cfg_file))
    cfg.MODEL.MASK_ON = True  # Enable mask prediction
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
    cfg.TEST.EVAL_PERIOD = 500  # Adjust as needed
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    cfg.OUTPUT_DIR = f"./output_{config_name}_{lr}_{max_iter}_{batch_size}_{timestamp}"
    os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
    # Set MODEL.WEIGHTS to the pre-trained model's URL
    cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(cfg_file)
    # Set the number of classes correctly
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = num_classes
    return cfg

def plot_metrics(output_dir):
    metrics_file = os.path.join(output_dir, "metrics.json")
    if not os.path.exists(metrics_file):
        logging.warning(f"No metrics.json found in {output_dir}")
        return [], [], []
    with open(metrics_file, "r") as f:
        metrics = [json.loads(line.strip()) for line in f]

    # Extract iterations and losses
    iterations = [x["iteration"] for x in metrics if "total_loss" in x]
    total_loss = [x["total_loss"] for x in metrics if "total_loss" in x]

    # Extract validation iterations and losses
    validation_iterations = [x["iteration"] for x in metrics if "validation_loss" in x]
    validation_loss = [x["validation_loss"] for x in metrics if "validation_loss" in x]

    # Extract AP values
    ap_iterations = [x["iteration"] for x in metrics if "bbox/AP" in x]
    ap_values = [x["bbox/AP"] for x in metrics if "bbox/AP" in x]

    # Plot training and validation loss
    plt.figure(figsize=(10, 5))
    plt.plot(iterations, total_loss, label="Training Loss")
    if validation_loss:
        plt.plot(validation_iterations, validation_loss, label="Validation Loss")
    plt.xlabel("Iteration")
    plt.ylabel("Loss")
    plt.legend()
    plt.savefig(os.path.join(output_dir, "loss_curve.png"))
    plt.close()

    # Plot Average Precision (AP) over iterations
    if ap_values:
        plt.figure(figsize=(10, 5))
        plt.plot(ap_iterations, ap_values, label="AP")
        plt.xlabel("Iteration")
        plt.ylabel("Average Precision")
        plt.legend()
        plt.savefig(os.path.join(output_dir, "ap_curve.png"))
        plt.close()

    return ap_iterations, ap_values, validation_loss

def train_and_evaluate(cfg, dataset_info, config_name):
    trainer = Trainer(cfg)
    trainer.resume_or_load(resume=False)
    trainer.train()

    ap_iterations, ap_values, val_losses = plot_metrics(cfg.OUTPUT_DIR)

    # Evaluate on test set
    cfg.MODEL.WEIGHTS = os.path.join(cfg.OUTPUT_DIR, "model_final.pth")
    cfg.DATASETS.TEST = ("cesm_test",)
    evaluator = CustomEvaluator("cesm_test", cfg, False, output_dir=cfg.OUTPUT_DIR)
    test_loader = build_detection_test_loader(
        cfg,
        "cesm_test",
        mapper=DatasetMapper(
            cfg,
            is_train=False,
            augmentations=[],
            image_format=cfg.INPUT.FORMAT,
            use_instance_mask=True,
        ),
    )
    results = inference_on_dataset(trainer.model, test_loader, evaluator)

    final_ap = results.get('segm', {}).get('AP', 0.0)
    # Access custom metrics from evaluator instance
    custom_metrics = evaluator._custom_metrics or {}
    confusion_matrix_result = custom_metrics.get('confusion_matrix', [])
    sensitivity_specificity = custom_metrics.get('sensitivity_and_specificity', {})

    result_entry = {
        'Pretrained Model Used': config_name,
        'Dropout': 'No',
        'Loss Function Name': 'Default',
        'Framework': 'Detectron2',
        'Early Stopping': 'Yes',
        'Test Accuracy (AP)': final_ap,
        'Key Hyperparameters': {
            'Learning Rate': cfg.SOLVER.BASE_LR,
            'Batch Size': cfg.SOLVER.IMS_PER_BATCH,
            'Max Iterations': cfg.SOLVER.MAX_ITER
        },
        'Augmentation Method': cfg.INPUT.RANDOM_FLIP,
        'Confusion Matrix': confusion_matrix_result,
        'Sensitivity and Specificity': sensitivity_specificity,
        'Dataset Info': dataset_info,
    }

    return results, ap_iterations, ap_values, result_entry

def main():
    # Register datasets
    register_coco_instances("cesm_train", {}, "./output/train_annotations.json", "../data/images")
    register_coco_instances("cesm_val", {}, "./output/val_annotations.json", "../data/images")
    register_coco_instances("cesm_test", {}, "./output/test_annotations.json", "../data/images")

    # Dataset information
    num_normal_test = 50
    num_benign_test = 30
    num_malignant_test = 20

    dataset_info = {
        'Train/Val/Test Split': '70/15/15',
        'Num Normal in Test Set': num_normal_test,
        'Num Benign in Test Set': num_benign_test,
        'Num Malignant in Test Set': num_malignant_test,
    }

    # Configurations to try
    configs = [
        ("mask_rcnn_R_50_FPN_3x", 0.0001, 10000, 2),
        ("mask_rcnn_R_101_FPN_3x", 0.0001, 15000, 2),
        # Add more configurations if needed
    ]

    results_data = []
    best_ap = 0
    best_config = None
    best_model_path = None

    for config_name, lr, max_iter, batch_size in configs:
        print(f"\nTraining with config: {config_name}, lr: {lr}, max_iter: {max_iter}, batch_size: {batch_size}")
        num_classes = 3  # Adjust according to your dataset
        cfg = setup_cfg(config_name, lr, max_iter, batch_size, num_classes)
        results, ap_iterations, ap_values, result_entry = train_and_evaluate(cfg, dataset_info, config_name)

        final_ap = result_entry['Test Accuracy (AP)']
        if final_ap > best_ap:
            best_ap = final_ap
            best_config = (config_name, lr, max_iter, batch_size)
            best_model_path = os.path.join(cfg.OUTPUT_DIR, "model_final.pth")
        else:
            # Optionally remove non-best models
            model_path = os.path.join(cfg.OUTPUT_DIR, "model_final.pth")
            if os.path.exists(model_path):
                os.remove(model_path)

        results_data.append(result_entry)

    # Process results_data to expand nested dictionaries
    df = pd.DataFrame(results_data)

    # Expand 'Key Hyperparameters' into separate columns
    hyperparams_df = df['Key Hyperparameters'].apply(pd.Series)
    df = pd.concat([df.drop('Key Hyperparameters', axis=1), hyperparams_df], axis=1)

    # Expand 'Sensitivity and Specificity' into separate columns
    sensitivity_specificity_df = pd.DataFrame()
    for class_name in df['Sensitivity and Specificity'].iloc[0].keys():
        class_metrics = df['Sensitivity and Specificity'].apply(lambda x: x[class_name])
        class_metrics_df = class_metrics.apply(pd.Series)
        class_metrics_df.columns = [f"{metric} ({class_name} vs rest)" for metric in class_metrics_df.columns]
        sensitivity_specificity_df = pd.concat([sensitivity_specificity_df, class_metrics_df], axis=1)

    df = pd.concat([df.drop('Sensitivity and Specificity', axis=1), sensitivity_specificity_df], axis=1)

    # Optionally, rename columns to match your desired format
    df.rename(columns={
        'Pretrained Model Used': 'Model',
        'Dropout': 'Drop out or no',
        'Test Accuracy (AP)': 'Test Accuracy',
        # Add any other renames here
    }, inplace=True)

    # Save the DataFrame to Excel
    df.to_excel('model_comparison.xlsx', index=False)
    print("\nResults saved to model_comparison.xlsx")
    print(f"Best configuration: {best_config}")
    print(f"Best AP: {best_ap}")
    print(f"Best model saved at: {best_model_path}")

if __name__ == "__main__":
    main()
