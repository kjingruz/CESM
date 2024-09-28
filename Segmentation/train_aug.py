import os
import torch
import json
import copy
import logging
import numpy as np
import time
import cv2
import random
import matplotlib.pyplot as plt
import albumentations as A
from detectron2 import model_zoo
from detectron2.config import get_cfg
from detectron2.utils import comm
from detectron2.engine import DefaultTrainer, HookBase
from detectron2.evaluation import COCOEvaluator
from detectron2.data import (
    DatasetCatalog,
    MetadataCatalog,
    build_detection_test_loader,
    build_detection_train_loader,
    get_detection_dataset_dicts,
)
from detectron2.data import detection_utils as utils
from detectron2.data.datasets import register_coco_instances
from detectron2.utils.events import EventStorage, get_event_storage
from detectron2.structures import BoxMode
from detectron2.data import DatasetMapper
from detectron2.data.samplers import RepeatFactorTrainingSampler
import pycocotools.mask as mask_util
import matplotlib.patches as patches

logging.basicConfig(level=logging.INFO)

# Fix for numpy boolean deprecation warning
if not hasattr(np, 'bool') or not isinstance(np.bool, type):
    np.bool = bool

class LossEvalHook(HookBase):
    def __init__(self, eval_period, model, data_loader):
        self._model = model
        self._period = eval_period
        self._data_loader = data_loader
        self.best_val_loss = float('inf')
        self.early_stop_patience = 5  # Number of evaluations with no improvement after which training will be stopped
        self.no_improve_counter = 0

    def _do_loss_eval(self):
        total_losses = []
        for inputs in self._data_loader:
            with torch.no_grad():
                loss_batch = self._get_loss(inputs)
                total_losses.append(loss_batch)
        mean_loss = np.mean(total_losses)
        logging.info(f"iter: {self.trainer.iter} validation_loss: {mean_loss:.4f}")
        self.trainer.storage.put_scalar('validation_loss', mean_loss)

        # Early stopping check
        if mean_loss < self.best_val_loss:
            self.best_val_loss = mean_loss
            self.no_improve_counter = 0
        else:
            self.no_improve_counter += 1
            logging.info(f"No improvement in validation loss for {self.no_improve_counter} evaluation(s)")
            if self.no_improve_counter >= self.early_stop_patience:
                logging.info("Early stopping triggered")
                self.trainer.early_stop = True

        return mean_loss

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
        if self._period > 0 and next_iter % self._period == 0:
            self._do_loss_eval()

class ValidationMapper(DatasetMapper):
    def __init__(self, cfg, is_train=False):
        super().__init__(
            cfg,
            is_train=is_train,
            augmentations=[],  # No augmentations for validation
            image_format=cfg.INPUT.FORMAT,
            use_instance_mask=True,
            instance_mask_format="bitmask",
        )

    def __call__(self, dataset_dict):
        dataset_dict = copy.deepcopy(dataset_dict)
        image = utils.read_image(dataset_dict["file_name"], format=self.image_format)
        utils.check_image_size(dataset_dict, image)

        image_shape = image.shape[:2]  # h, w

        dataset_dict["image"] = torch.as_tensor(
            np.ascontiguousarray(image.transpose(2, 0, 1))
        )

        # Ensure that 'instances' is always present
        annos = dataset_dict.get("annotations", [])
        instances = utils.annotations_to_instances(
            annos, image_shape, mask_format=self.instance_mask_format
        )
        instances = utils.filter_empty_instances(instances)
        dataset_dict["instances"] = instances

        return dataset_dict

class AlbumentationsMapper(DatasetMapper):
    def __init__(self, cfg, is_train=True, visualize=False, output_dir="./aug_visualizations"):
        super().__init__(
            cfg,
            is_train=is_train,
            augmentations=[],  # We'll handle augmentations manually
            image_format=cfg.INPUT.FORMAT,
            use_instance_mask=True,
            instance_mask_format="bitmask",
        )

        self.visualize = visualize
        self.output_dir = output_dir
        if self.visualize and not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)

        # Adjusted augmentations with less aggressive parameters
        self.spatial_transforms = A.Compose(
            [
                A.HorizontalFlip(p=0.5),
                # A.VerticalFlip(p=0.5),  # Uncomment if vertical flips are appropriate
                A.Rotate(limit=10, p=0.5),
                A.ShiftScaleRotate(
                    shift_limit=0.01, scale_limit=0.02, rotate_limit=10, p=0.5
                ),
            ],
            bbox_params=A.BboxParams(format='pascal_voc', label_fields=['category_ids']),
        )

        self.non_spatial_transforms = A.Compose(
            [
                A.RandomBrightnessContrast(p=0.5),
                A.RandomGamma(p=0.5),
                A.GaussNoise(p=0.5),
            ]
        )

    def __call__(self, dataset_dict):
        dataset_dict = copy.deepcopy(dataset_dict)  # Avoid modifying the original data

        # Check if 'file_name' key is present
        if "file_name" not in dataset_dict:
            logging.warning(
                f"Skipping sample: 'file_name' not found in dataset_dict"
            )
            return None

        image = utils.read_image(dataset_dict["file_name"], format=self.image_format)

        if self.is_train:
            # Check if annotations are present
            if "annotations" not in dataset_dict or len(dataset_dict["annotations"]) == 0:
                logging.warning(
                    f"Skipping sample: No annotations found for image {dataset_dict['file_name']}"
                )
                return None

            try:
                # Convert boxes to XYXY format (absolute coordinates)
                boxes = [
                    BoxMode.convert(
                        obj["bbox"], obj["bbox_mode"], BoxMode.XYXY_ABS
                    )
                    for obj in dataset_dict["annotations"]
                ]
                category_ids = [obj["category_id"] for obj in dataset_dict["annotations"]]

                # Create masks from polygons
                masks = []
                for annotation in dataset_dict["annotations"]:
                    mask = self.polygons_to_mask(
                        annotation["segmentation"], image.shape[:2]
                    )
                    masks.append(mask)

                # Apply spatial augmentations
                transformed = self.spatial_transforms(
                    image=image,
                    masks=masks,
                    bboxes=boxes,
                    category_ids=category_ids,
                )

                image = transformed["image"]
                augmented_masks = transformed["masks"]
                augmented_boxes = transformed["bboxes"]
                category_ids = transformed["category_ids"]  # Update category_ids after augmentation

                # Apply non-spatial augmentations
                image = self.non_spatial_transforms(image=image)["image"]

                # Get image dimensions
                height, width = image.shape[:2]

                # Clip bounding boxes and filter invalid ones
                valid_boxes = []
                valid_masks = []
                valid_category_ids = []
                for box, mask, category_id in zip(
                    augmented_boxes, augmented_masks, category_ids
                ):
                    x_min, y_min, x_max, y_max = box

                    # Clip the box coordinates to image boundaries
                    x_min = max(0, min(x_min, width - 1))
                    y_min = max(0, min(y_min, height - 1))
                    x_max = max(0, min(x_max, width - 1))
                    y_max = max(0, min(y_max, height - 1))

                    if x_max > x_min and y_max > y_min:
                        valid_boxes.append([x_min, y_min, x_max, y_max])
                        valid_masks.append(mask)
                        valid_category_ids.append(category_id)
                    else:
                        logging.warning(
                            f"Invalid box after clipping for image {dataset_dict['file_name']}: {[x_min, y_min, x_max, y_max]}"
                        )

                if len(valid_boxes) == 0:
                    logging.warning(
                        f"Skipping sample: All boxes became invalid after augmentation for image {dataset_dict['file_name']}"
                    )
                    return None

                # Update annotations with clipped boxes and masks
                dataset_dict["annotations"] = []
                for box, mask, category_id in zip(
                    valid_boxes, valid_masks, valid_category_ids
                ):
                    # Convert box back to XYWH format
                    bbox = BoxMode.convert(box, BoxMode.XYXY_ABS, BoxMode.XYWH_ABS)
                    annotation = {
                        "bbox": bbox,
                        "bbox_mode": BoxMode.XYWH_ABS,
                        "segmentation": mask_util.encode(
                            np.asfortranarray(mask.astype(np.uint8))
                        ),
                        "category_id": category_id,
                    }
                    dataset_dict["annotations"].append(annotation)

                # Visualization
                if self.visualize and random.random() < 0.01:  # Adjust the probability as needed
                    self.visualize_augmentation(image, valid_boxes, valid_masks, dataset_dict["file_name"])

            except Exception as e:
                logging.warning(
                    f"Error processing sample {dataset_dict['file_name']}: {str(e)}"
                )
                return None

        # Convert image to tensor
        dataset_dict["image"] = torch.as_tensor(
            image.transpose(2, 0, 1).astype("float32")
        )

        # Create instances
        instances = utils.annotations_to_instances(
            dataset_dict["annotations"],
            image.shape[:2],
            mask_format=self.instance_mask_format,
        )
        dataset_dict["instances"] = utils.filter_empty_instances(instances)

        return dataset_dict

    def polygons_to_mask(self, polygons, image_shape):
        mask = np.zeros(image_shape, dtype=np.uint8)
        for polygon in polygons:
            polygon = np.array(polygon).reshape(-1, 2)
            cv2.fillPoly(mask, [polygon.astype(np.int32)], 1)
        return mask

    def visualize_augmentation(self, image, boxes, masks, original_file_name):
        fig, ax = plt.subplots(1)
        # Convert image from BGR to RGB if necessary
        image_to_show = image[:, :, ::-1] if self.image_format == "BGR" else image
        ax.imshow(image_to_show)

        for box, mask in zip(boxes, masks):
            x_min, y_min, x_max, y_max = box
            width = x_max - x_min
            height = y_max - y_min
            rect = patches.Rectangle(
                (x_min, y_min),
                width,
                height,
                linewidth=2,
                edgecolor='r',
                facecolor='none',
            )
            ax.add_patch(rect)

            # Overlay the mask
            masked_image = np.ma.masked_where(mask == 0, mask)
            ax.imshow(masked_image, alpha=0.5, cmap='jet')

        # Save the figure
        base_name = os.path.basename(original_file_name)
        save_path = os.path.join(self.output_dir, f"aug_{base_name}")
        plt.axis('off')
        plt.savefig(save_path, bbox_inches='tight', pad_inches=0)
        plt.close(fig)

class Trainer(DefaultTrainer):
    def __init__(self, cfg):
        super().__init__(cfg)
        self._data_loader_iter = iter(self.data_loader)
        self.early_stop = False  # Flag for early stopping

    @classmethod
    def build_evaluator(cls, cfg, dataset_name, output_folder=None):
        if output_folder is None:
            output_folder = os.path.join(cfg.OUTPUT_DIR, "inference")
        return COCOEvaluator(dataset_name, cfg, True, output_folder)

    @classmethod
    def build_train_loader(cls, cfg):
        dataset_dicts = get_detection_dataset_dicts(
            cfg.DATASETS.TRAIN,
            filter_empty=True,
        )
        repeat_factors = RepeatFactorTrainingSampler.repeat_factors_from_category_frequency(
            dataset_dicts, repeat_thresh=0.001  # Adjust as needed
        )
        sampler = RepeatFactorTrainingSampler(repeat_factors)
        return build_detection_train_loader(
            cfg, sampler=sampler, mapper=AlbumentationsMapper(cfg, is_train=True, visualize=True)
        )

    def run_step(self):
        if self.early_stop:
            raise StopIteration  # Stop training

        assert self.model.training, "[SimpleTrainer] model was changed to eval mode!"
        start = time.perf_counter()

        while True:
            data = next(self._data_loader_iter, None)
            if data is None:
                # Reached the end of the dataset, create a new iterator
                self._data_loader_iter = iter(self.data_loader)
                data = next(self._data_loader_iter)
            if len(data) == 0:
                logging.warning("Encountered empty batch. Skipping step.")
                continue
            break

        data_time = time.perf_counter() - start

        loss_dict = self.model(data)
        losses = sum(loss_dict.values())
        self._detect_anomaly(losses, loss_dict)

        metrics_dict = loss_dict
        metrics_dict["data_time"] = data_time
        self._write_metrics(metrics_dict)

        self.optimizer.zero_grad()
        losses.backward()
        self.optimizer.step()

    def _detect_anomaly(self, losses, loss_dict):
        if not torch.isfinite(losses).all():
            raise FloatingPointError(
                "Loss became infinite or NaN at iteration={}!\nloss_dict = {}".format(
                    self.iter, loss_dict
                )
            )

    def _write_metrics(self, metrics_dict: dict):
        """
        Args:
            metrics_dict (dict): dict of scalar metrics
        """
        metrics_dict = {
            k: v.detach().cpu().item()
            if isinstance(v, torch.Tensor)
            else float(v)
            for k, v in metrics_dict.items()
        }
        # Gather metrics among all workers for logging
        all_metrics_dict = comm.gather(metrics_dict)

        if comm.is_main_process():
            if "data_time" in all_metrics_dict[0]:
                # Data_time among workers can have high variance
                data_time = np.max([x.pop("data_time") for x in all_metrics_dict])
                self.storage.put_scalar("data_time", data_time)

            # Average the rest metrics
            metrics_dict = {
                k: np.mean([x[k] for x in all_metrics_dict])
                for k in all_metrics_dict[0].keys()
            }
            total_losses_reduced = sum(loss for loss in metrics_dict.values())

            self.storage.put_scalar("total_loss", total_losses_reduced)
            if len(metrics_dict) > 1:
                self.storage.put_scalars(**metrics_dict)

    def build_hooks(self):
        hooks = super().build_hooks()
        val_loader = build_detection_test_loader(
            self.cfg,
            self.cfg.DATASETS.TEST[0],
            mapper=ValidationMapper(self.cfg, is_train=False),
        )
        hooks.insert(
            -1,
            LossEvalHook(
                eval_period=self.cfg.TEST.EVAL_PERIOD,
                model=self.model,
                data_loader=val_loader,
            ),
        )
        return hooks

    def after_step(self):
        super().after_step()
        if self.iter % 20 == 0:
            losses = self.storage.latest()
            total_loss = sum(
                loss.item()
                if isinstance(loss, torch.Tensor)
                else sum(loss)
                if isinstance(loss, tuple)
                else loss
                for loss in losses.values()
            )
            logging.info(f"iter: {self.iter} total_loss: {total_loss:.4f}")

def setup_cfg():
    cfg = get_cfg()
    cfg.merge_from_file(
        model_zoo.get_config_file(
            "COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"
        )
    )
    cfg.DATASETS.TRAIN = ("cesm_train",)
    cfg.DATASETS.TEST = ("cesm_val",)
    cfg.DATALOADER.NUM_WORKERS = 2
    cfg.INPUT.FORMAT = "BGR"  # Ensure image format is consistent
    cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(
        "COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"
    )
    cfg.SOLVER.IMS_PER_BATCH = 2
    cfg.SOLVER.BASE_LR = 0.00025
    cfg.SOLVER.MAX_ITER = 5000
    cfg.SOLVER.STEPS = (3000, 4000)
    cfg.SOLVER.CHECKPOINT_PERIOD = 500
    cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 128
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 2  # Adjusted for your dataset (e.g., Benign and Malignant)
    cfg.TEST.EVAL_PERIOD = 250
    cfg.OUTPUT_DIR = "./output_aug"
    os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
    return cfg

def validate_bboxes(json_file):
    with open(json_file, 'r') as f:
        data = json.load(f)

    invalid_annotations = []
    for ann in data['annotations']:
        x, y, w, h = ann['bbox']
        if w <= 0 or h <= 0:
            invalid_annotations.append(ann)

    if invalid_annotations:
        print(f"Found {len(invalid_annotations)} invalid bounding boxes in {json_file}:")
        for ann in invalid_annotations:
            print(f"Image ID: {ann['image_id']}, Annotation ID: {ann['id']}, bbox: {ann['bbox']}")
    else:
        print(f"All bounding boxes in {json_file} are valid.")

def validate_annotations(dataset_name):
    dataset_dicts = DatasetCatalog.get(dataset_name)
    invalid_annotations = []
    for data in dataset_dicts:
        width = data.get('width', None)
        height = data.get('height', None)
        if width is None or height is None:
            logging.warning(f"Image {data['file_name']} does not have width and height information.")
            continue
        for ann in data["annotations"]:
            x, y, w, h = ann['bbox']
            if w <= 0 or h <= 0:
                invalid_annotations.append((data['file_name'], ann))
            if x < 0 or y < 0 or x + w > width or y + h > height:
                invalid_annotations.append((data['file_name'], ann))

    if invalid_annotations:
        print(f"Found {len(invalid_annotations)} invalid annotations in {dataset_name}:")
        for file_name, ann in invalid_annotations:
            print(f"Image: {file_name}, Annotation: {ann}")
    else:
        print(f"All annotations in {dataset_name} are valid.")

def main():
    # Register datasets
    register_coco_instances(
        "cesm_train", {}, "./output/train_annotations.json", "../data/images"
    )
    register_coco_instances(
        "cesm_val", {}, "./output/val_annotations.json", "../data/images"
    )
    register_coco_instances(
        "cesm_test", {}, "./output/test_annotations.json", "../data/images"
    )

    # Validate bounding boxes
    validate_bboxes("./output/train_annotations.json")
    validate_bboxes("./output/val_annotations.json")
    validate_bboxes("./output/test_annotations.json")

    # Validate annotations
    validate_annotations("cesm_train")
    validate_annotations("cesm_val")
    validate_annotations("cesm_test")

    cfg = setup_cfg()
    trainer = Trainer(cfg)
    trainer.resume_or_load(resume=False)
    try:
        trainer.train()
    except StopIteration:
        logging.info("Training stopped due to early stopping.")

    # Evaluate on test set
    cfg.MODEL.WEIGHTS = os.path.join(cfg.OUTPUT_DIR, "model_final.pth")
    cfg.DATASETS.TEST = ("cesm_test",)
    evaluator = COCOEvaluator(
        "cesm_test", cfg, False, output_dir=cfg.OUTPUT_DIR
    )
    trainer.test(cfg, trainer.model, evaluators=[evaluator])

if __name__ == "__main__":
    main()
