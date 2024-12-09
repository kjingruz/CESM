import numpy as np
import os
import torch
import json
import logging
import datetime
import pandas as pd
from detectron2 import model_zoo
from detectron2.config import get_cfg
from detectron2.engine import DefaultPredictor
from detectron2.evaluation import COCOEvaluator, inference_on_dataset
from detectron2.data import build_detection_test_loader, DatasetMapper
from detectron2.data.datasets import register_coco_instances
from detectron2.data import MetadataCatalog
from sklearn.metrics import confusion_matrix, classification_report
from torchvision.ops import box_iou

logging.basicConfig(level=logging.INFO)

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
                logging.warning(f"No ground truth instances found for input {input['image_id']}")
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
        # Compute confusion matrix and other metrics
        if not self._predictions_list:
            logging.warning("No predictions to evaluate.")
            return coco_results

        gt_classes = []
        pred_classes = []

        for gt_annos, preds in zip(self._gt_annotations_list, self._predictions_list):
            gt_bboxes = [anno['bbox'] for anno in gt_annos]
            gt_cls = [anno['category_id'] for anno in gt_annos]
            pred_bboxes = [pred['bbox'] for pred in preds]
            pred_cls = [pred['category_id'] for pred in preds]

            # Match predictions to ground truths
            matches = self._match_predictions(gt_bboxes, gt_cls, pred_bboxes, pred_cls)

            # Collect classes based on matches
            for match in matches:
                gt_classes.append(match['gt_class'] if match['gt_class'] is not None else -1)
                pred_classes.append(match['pred_class'] if match['pred_class'] is not None else -1)

        # Now gt_classes and pred_classes have the same length
        labels = list(range(len(self._metadata.thing_classes)))
        # Include an extra label for unmatched predictions/ground truths
        labels_with_unmatched = labels + [-1]
        target_names = self._metadata.thing_classes + ['unmatched']
        cm = confusion_matrix(gt_classes, pred_classes, labels=labels_with_unmatched)
        report = classification_report(
            gt_classes,
            pred_classes,
            labels=labels,
            target_names=self._metadata.thing_classes,
            output_dict=True,
            zero_division=0
        )
        sensitivity_specificity = {}
        for idx, cls_name in enumerate(self._metadata.thing_classes):
            TP = cm[idx, idx]
            FN = cm[idx, :].sum() - TP
            FP = cm[:, idx].sum() - TP
            TN = cm.sum() - (TP + FP + FN)
            sensitivity = TP / (TP + FN) if (TP + FN) > 0 else 0
            specificity = TN / (TN + FP) if (TN + FP) > 0 else 0
            sensitivity_specificity[cls_name] = {
                'sensitivity': sensitivity,
                'specificity': specificity
            }

        # Store custom metrics in an instance variable
        self._custom_metrics = {
            'confusion_matrix': cm.tolist(),
            'classification_report': report,
            'sensitivity_and_specificity': sensitivity_specificity
        }

        return coco_results  # Return only standard COCO results

    def _match_predictions(self, gt_bboxes, gt_classes, pred_bboxes, pred_classes):
        matches = []
        if len(gt_bboxes) == 0 and len(pred_bboxes) == 0:
            return matches  # Nothing to match
        elif len(gt_bboxes) == 0:
            # All predictions are unmatched (False Positives)
            for cls in pred_classes:
                matches.append({'gt_class': None, 'pred_class': cls})
            return matches
        elif len(pred_bboxes) == 0:
            # All ground truths are unmatched (False Negatives)
            for cls in gt_classes:
                matches.append({'gt_class': cls, 'pred_class': None})
            return matches

        gt_bboxes_tensor = torch.tensor(gt_bboxes)
        pred_bboxes_tensor = torch.tensor(pred_bboxes)
        ious = box_iou(gt_bboxes_tensor, pred_bboxes_tensor)

        # For each GT box, find the best matching predicted box
        gt_matched = set()
        pred_matched = set()
        for gt_idx in range(len(gt_bboxes)):
            iou_row = ious[gt_idx]
            if len(iou_row) == 0:
                continue
            max_iou, pred_idx = iou_row.max(0)
            pred_idx = pred_idx.item()
            if max_iou >= self.iou_threshold and pred_idx not in pred_matched:
                matches.append({'gt_class': gt_classes[gt_idx], 'pred_class': pred_classes[pred_idx]})
                gt_matched.add(gt_idx)
                pred_matched.add(pred_idx)
            else:
                # Unmatched ground truth
                matches.append({'gt_class': gt_classes[gt_idx], 'pred_class': None})

        # Handle unmatched predicted boxes (False Positives)
        for pred_idx in range(len(pred_bboxes)):
            if pred_idx not in pred_matched:
                matches.append({'gt_class': None, 'pred_class': pred_classes[pred_idx]})

        return matches

def setup_cfg(config_name, num_classes, weight_path):
    cfg = get_cfg()
    # Use the appropriate config file
    cfg_file = f"COCO-Detection/{config_name}.yaml"
    cfg.merge_from_file(model_zoo.get_config_file(cfg_file))
    cfg.DATASETS.TEST = ("cesm_test",)
    cfg.DATALOADER.NUM_WORKERS = 2
    cfg.MODEL.WEIGHTS = weight_path
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.05
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = num_classes
    cfg.INPUT.MIN_SIZE_TEST = 800
    cfg.INPUT.MAX_SIZE_TEST = 1333
    cfg.TEST.EVAL_PERIOD = 0
    cfg.freeze()
    return cfg

def register_datasets():
    # Register datasets
    register_coco_instances("cesm_train", {}, "./output/train_annotations.json", "../data/images")
    register_coco_instances("cesm_val", {}, "./output/val_annotations.json", "../data/images")
    register_coco_instances("cesm_test", {}, "./output/test_annotations.json", "../data/images")

def perform_evaluation(configs, weight_paths, output_csv_path):
    results_data = []
    for config_name, weight_path, pretrained_from in zip(configs['config_names'], configs['weight_paths'], configs['pretrained_from']):
        print(f"\nEvaluating model: {config_name} with weights from {weight_path}")
        num_classes = 3  # Adjust according to your dataset
        cfg = setup_cfg(config_name, num_classes, weight_path)
        evaluator = CustomEvaluator("cesm_test", cfg, False, output_dir=None)
        test_loader = build_detection_test_loader(cfg, "cesm_test", mapper=DatasetMapper(cfg, False))
        results = inference_on_dataset(DefaultPredictor(cfg).model, test_loader, evaluator)

        # Access custom metrics from evaluator instance
        custom_metrics = evaluator._custom_metrics or {}
        confusion_matrix_result = custom_metrics.get('confusion_matrix', [])
        classification_report_result = custom_metrics.get('classification_report', {})
        sensitivity_specificity = custom_metrics.get('sensitivity_and_specificity', {})

        # AP value
        ap = results.get('bbox', {}).get('AP', 0.0)

        # Organize sensitivity and specificity
        sensitivity = {}
        specificity = {}
        for cls, metrics in sensitivity_specificity.items():
            sensitivity[f'sensitivity_{cls}'] = metrics['sensitivity']
            specificity[f'specificity_{cls}'] = metrics['specificity']

        # Flatten classification report if needed
        # For simplicity, we'll skip detailed classification report and focus on sensitivity and specificity

        result_entry = {
            'Model Configuration': config_name,
            'Pretrained From': pretrained_from,
            'AP': ap,
            'Confusion Matrix': json.dumps(confusion_matrix_result),
            'Classification Report': json.dumps(classification_report_result),
            **sensitivity,
            **specificity
        }

        results_data.append(result_entry)
        print(f"AP for {config_name}: {ap}")

    # Save results to CSV
    df = pd.DataFrame(results_data)
    df.to_csv(output_csv_path, index=False)
    print(f"\nEvaluation results saved to {output_csv_path}")

def main():
    # Register datasets
    register_datasets()

    # Define configurations and corresponding weight files
    configs = {
        'config_names': [
            "faster_rcnn_R_50_FPN_3x",
            "faster_rcnn_R_101_FPN_3x",
            "retinanet_R_50_FPN_3x"
        ],
        'weight_paths': [
            "./output_faster_rcnn_R_50_FPN_3x_0.0001_10000_2_20241007_015757/model_0004999.pth",
            "./output_faster_rcnn_R_101_FPN_3x_0.0001_15000_2_20241007_024330/model_0004999.pth",
            "./output_retinanet_R_50_FPN_3x_5e-05_20000_4_20241007_041412/model_0004999.pth"
        ],
        'pretrained_from': [
            "COCO-Detection/faster_rcnn_R_50_FPN_3x",
            "COCO-Detection/faster_rcnn_R_101_FPN_3x",
            "COCO-Detection/retinanet_R_50_FPN_3x"
        ]
    }

    # Output CSV path
    output_csv_path = "evaluation_results.csv"

    # Perform evaluation
    perform_evaluation(configs, configs['weight_paths'], output_csv_path)

if __name__ == "__main__":
    main()
