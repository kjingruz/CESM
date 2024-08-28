import os
import torch
import numpy as np
import albumentations as A
from detectron2 import model_zoo
from detectron2.config import get_cfg
from detectron2.engine import DefaultTrainer, HookBase
from detectron2.evaluation import COCOEvaluator
from detectron2.data import DatasetMapper, build_detection_train_loader, build_detection_test_loader
from detectron2.data.datasets import register_coco_instances
from detectron2.solver import WarmupMultiStepLR
import detectron2.data.transforms as T
from detectron2.data.detection_utils import read_image
from detectron2.structures import BoxMode
from detectron2.data import detection_utils as utils
from detectron2.modeling import build_model
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from detectron2.utils.events import EventStorage
from detectron2.utils.logger import log_every_n_seconds
import json
import time
import datetime
import logging
from detectron2.utils import comm
import pandas as pd

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

class LossEvalHook(HookBase):
    def __init__(self, eval_period, model, data_loader):
        self._model = model
        self._period = eval_period
        self._data_loader = data_loader
    
    def _do_loss_eval(self):
        total = len(self._data_loader)
        num_warmup = min(5, total - 1)

        start_time = time.perf_counter()
        total_compute_time = 0
        losses = []
        for idx, inputs in enumerate(self._data_loader):
            if idx == num_warmup:
                start_time = time.perf_counter()
                total_compute_time = 0
            start_compute_time = time.perf_counter()
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            total_compute_time += time.perf_counter() - start_compute_time
            iters_after_start = idx + 1 - num_warmup * int(idx >= num_warmup)
            seconds_per_img = total_compute_time / iters_after_start
            if idx >= num_warmup * 2 or seconds_per_img > 5:
                total_seconds_per_img = (time.perf_counter() - start_time) / iters_after_start
                eta = datetime.timedelta(seconds=int(total_seconds_per_img * (total - idx - 1)))
                log_every_n_seconds(
                    logging.INFO,
                    "Loss on Validation  done {}/{}. {:.4f} s / img. ETA={}".format(
                        idx + 1, total, seconds_per_img, str(eta)
                    ),
                    n=5,
                )
            loss_batch = self._get_loss(inputs)
            losses.append(loss_batch)
        mean_loss = np.mean(losses)
        self.trainer.storage.put_scalar('validation_loss', mean_loss)
        print(f"Iteration {self.trainer.iter}: Validation Loss = {mean_loss}")  # Debug print
        comm.synchronize()
        return losses
    
    def _get_loss(self, data):
        metrics_dict = self._model(data)
        metrics_dict = {
            k: v.detach().cpu().item() if isinstance(v, torch.Tensor) else float(v)
            for k, v in metrics_dict.items()
        }
        total_losses_reduced = sum(loss for loss in metrics_dict.values())
        return total_losses_reduced

    def after_step(self):
        next_iter = self.trainer.iter + 1
        is_final = next_iter == self.trainer.max_iter
        if is_final or (self._period > 0 and next_iter % self._period == 0):
            self._do_loss_eval()
        self.trainer.storage.put_scalars(timetest=12)

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
        return WarmupMultiStepLR(
            optimizer,
            cfg.SOLVER.STEPS,
            cfg.SOLVER.GAMMA,
            warmup_factor=cfg.SOLVER.WARMUP_FACTOR,
            warmup_iters=cfg.SOLVER.WARMUP_ITERS,
            warmup_method=cfg.SOLVER.WARMUP_METHOD,
        )

    def build_hooks(self):
        hooks = super().build_hooks()
        hooks.append(LossEvalHook(
            self.cfg.TEST.EVAL_PERIOD,
            self.model,
            build_detection_test_loader(
                self.cfg,
                self.cfg.DATASETS.TEST[0],
                DatasetMapper(self.cfg, True)
            )
        ))
        return hooks

    @classmethod
    def build_model(cls, cfg):
        model = build_model(cfg)
        for name, module in model.named_modules():
            if isinstance(module, nn.CrossEntropyLoss):
                setattr(model, name, FocalLoss(alpha=0.25, gamma=2))
        return model

def plot_losses(output_dir):
    experiment_folder = output_dir

    def load_json_arr(json_path):
        lines = []
        with open(json_path, 'r') as f:
            for line in f:
                lines.append(json.loads(line))
        return lines

    experiment_metrics = load_json_arr(os.path.join(experiment_folder, 'metrics.json'))

    # Plot both training and validation loss
    plt.figure(figsize=(15,8))
    plt.plot(
        [x['iteration'] for x in experiment_metrics if 'total_loss' in x], 
        [x['total_loss'] for x in experiment_metrics if 'total_loss' in x],
        label='training_loss'
    )
    plt.plot(
        [x['iteration'] for x in experiment_metrics if 'validation_loss' in x],
        [x['validation_loss'] for x in experiment_metrics if 'validation_loss' in x],
        label='validation_loss'
    )
    plt.legend(['training_loss', 'validation_loss'], loc='upper right')
    plt.xlabel('iteration')
    plt.ylabel('loss')
    plt.title('Training and Validation Loss (10,000 iterations)')
    plt.savefig(os.path.join(output_dir, 'combined_loss_plot_10000.png'))
    plt.close()


def plot_losses(output_dir):
    experiment_folder = output_dir

    def load_json_arr(json_path):
        lines = []
        with open(json_path, 'r') as f:
            for line in f:
                lines.append(json.loads(line))
        return lines

    experiment_metrics = load_json_arr(os.path.join(experiment_folder, 'metrics.json'))

    train_iterations = [x['iteration'] for x in experiment_metrics if 'total_loss' in x]
    train_losses = [x['total_loss'] for x in experiment_metrics if 'total_loss' in x]
    val_iterations = [x['iteration'] for x in experiment_metrics if 'validation_loss' in x]
    val_losses = [x['validation_loss'] for x in experiment_metrics if 'validation_loss' in x]

    # Plot training loss
    plt.figure(figsize=(10,5))
    plt.plot(train_iterations, train_losses)
    plt.xlabel('iteration')
    plt.ylabel('loss')
    plt.title('Training Loss')
    plt.savefig(os.path.join(output_dir, 'training_loss_plot.png'))
    plt.close()

    # Plot validation loss
    plt.figure(figsize=(10,5))
    if val_losses:
        plt.plot(val_iterations, val_losses)
        plt.xlabel('iteration')
        plt.ylabel('loss')
        plt.title('Validation Loss')
        plt.savefig(os.path.join(output_dir, 'validation_loss_plot.png'))
    else:
        plt.text(0.5, 0.5, 'No Validation Loss Data Available', ha='center', va='center')
        plt.title('Validation Loss (No Data)')
        plt.savefig(os.path.join(output_dir, 'validation_loss_plot_no_data.png'))
    plt.close()

    # Plot both training and validation loss
    plt.figure(figsize=(10,5))
    plt.plot(train_iterations, train_losses, label='training_loss')
    if val_losses:
        plt.plot(val_iterations, val_losses, label='validation_loss')
    plt.legend(loc='upper left')
    plt.xlabel('iteration')
    plt.ylabel('loss')
    plt.title('Training and Validation Loss')
    plt.savefig(os.path.join(output_dir, 'combined_loss_plot.png'))
    plt.close()

    # Print debug information
    print(f"Total metrics entries: {len(experiment_metrics)}")
    print(f"Training loss entries: {len(train_losses)}")
    print(f"Validation loss entries: {len(val_losses)}")

    # If no validation loss, print a sample of the metrics for debugging
    if not val_losses:
        print("Sample of metrics (first 10 entries):")
        for entry in experiment_metrics[:10]:
            print(entry)

def calculate_iou(box1, box2):
    # Calculate intersection coordinates
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    # Calculate intersection area
    intersection = max(0, x2 - x1) * max(0, y2 - y1)

    # Calculate union area
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union = area1 + area2 - intersection

    # Calculate IoU
    iou = intersection / union if union > 0 else 0

    return iou


def record_metrics(output_dir):
    experiment_folder = output_dir
    metrics_file = os.path.join(experiment_folder, 'metrics.json')

    metrics = []
    with open(metrics_file, 'r') as f:
        for line in f:
            metrics.append(json.loads(line))

    # Extract AP and other relevant metrics
    ap_metrics = [m for m in metrics if 'bbox/AP' in m]

    data = []
    for m in ap_metrics:
        iteration = m['iteration']
        ap = m['bbox/AP']
        ap_50 = m['bbox/AP50']
        ap_75 = m['bbox/AP75']
        ap_s = m['bbox/APs']
        ap_m = m['bbox/APm']
        ap_l = m['bbox/APl']

        # Calculate average IOU for this iteration
        if 'data' in m:
            ious = []
            for instance in m['data']:
                if 'groundtruth' in instance and 'prediction' in instance:
                    iou = calculate_iou(instance['groundtruth'], instance['prediction'])
                    ious.append(iou)
            avg_iou = sum(ious) / len(ious) if ious else 'N/A'
        else:
            avg_iou = 'N/A'

        data.append([iteration, ap, ap_50, ap_75, ap_s, ap_m, ap_l, avg_iou])

    # Create a DataFrame and save as CSV
    df = pd.DataFrame(data, columns=['Iteration', 'AP', 'AP50', 'AP75', 'APs', 'APm', 'APl', 'Avg_IOU'])
    df.to_csv(os.path.join(output_dir, 'metrics_table.csv'), index=False)



def setup_cfg():
    cfg = get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file("COCO-Detection/faster_rcnn_X_101_32x8d_FPN_3x.yaml"))
    cfg.DATASETS.TRAIN = ("cesm_train",)
    cfg.DATASETS.TEST = ("cesm_val",)  # Using validation set
    cfg.DATALOADER.NUM_WORKERS = 2
    cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-Detection/faster_rcnn_X_101_32x8d_FPN_3x.yaml")

    cfg.SOLVER.IMS_PER_BATCH = 2
    cfg.SOLVER.BASE_LR = 0.0004
    cfg.SOLVER.MAX_ITER = 4000  # Changed to 4000 iterations

    cfg.SOLVER.STEPS = (2000, 3000)  # Adjusted for 4000 iterations
    cfg.SOLVER.GAMMA = 0.1
    cfg.SOLVER.WARMUP_FACTOR = 1.0 / 1000
    cfg.SOLVER.WARMUP_ITERS = 200  # Kept the same
    cfg.SOLVER.WARMUP_METHOD = "linear"

    cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 128
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 2
    cfg.MODEL.ROI_HEADS.POSITIVE_FRACTION = 0.25

    cfg.TEST.EVAL_PERIOD = 100  # Evaluation every 100 iterations
    cfg.SOLVER.CHECKPOINT_PERIOD = 100  # Checkpoint every 500 iterations

    cfg.INPUT.MIN_SIZE_TRAIN = (480, 512, 544, 576, 608, 640, 672, 704, 736, 768, 800)
    cfg.INPUT.MAX_SIZE_TRAIN = 1333
    cfg.INPUT.MIN_SIZE_TEST = 800
    cfg.INPUT.MAX_SIZE_TEST = 1333

    cfg.MODEL.DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    cfg.DATALOADER.SAMPLER_TRAIN = "RepeatFactorTrainingSampler"
    cfg.DATALOADER.REPEAT_THRESHOLD = 0.001

    cfg.TEST.AUG.ENABLED = True
    cfg.TEST.AUG.MIN_SIZES = (400, 500, 600, 700, 800, 900, 1000, 1100, 1200)
    cfg.TEST.AUG.MAX_SIZE = 4000
    cfg.TEST.AUG.FLIP = True

    cfg.SOLVER.WEIGHT_DECAY = 0.0001

    cfg.MODEL.ROI_HEADS.NMS_THRESH_TEST = 0.4

    cfg.MODEL.FPN.DROPOUT = 0.2

    cfg.OUTPUT_DIR = "./4000_iterations_improved_with_val"
    os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
    return cfg

def main():
    register_coco_instances("cesm_train", {}, "./output/train_annotations.json", "../data/images")
    register_coco_instances("cesm_val", {}, "./output/val_annotations.json", "../data/images")
    register_coco_instances("cesm_test", {}, "./output/test_annotations.json", "../data/images")

    print("Starting training for 4000 iterations...")
    cfg = setup_cfg()
    trainer = CustomTrainer(cfg)
    trainer.resume_or_load(resume=False)
    trainer.train()
    plot_losses(cfg.OUTPUT_DIR)
    record_metrics(cfg.OUTPUT_DIR)
    print("Completed training for 4000 iterations.")

    # Evaluate on the test set after training
    print("Evaluating on test set...")
    cfg.DATASETS.TEST = ("cesm_test",)
    evaluator = COCOEvaluator("cesm_test", cfg, False, output_dir="./output/")
    test_loader = build_detection_test_loader(cfg, "cesm_test")
    results = inference_on_dataset(trainer.model, test_loader, evaluator)
    print("Test set evaluation results:", results)

if __name__ == "__main__":
    main()
