import os
import torch
import json
import logging
import datetime
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from detectron2 import model_zoo
from detectron2.config import get_cfg
from detectron2.engine import DefaultTrainer, HookBase
from detectron2.evaluation import COCOEvaluator
from detectron2.data import DatasetMapper, build_detection_test_loader
from detectron2.data.datasets import register_coco_instances
from detectron2.utils.events import EventStorage
from detectron2.utils import comm

logging.basicConfig(level=logging.INFO)

if not hasattr(np, 'bool') or not isinstance(np.bool, type):
    np.bool = bool

class LossEvalHook(HookBase):
    def __init__(self, eval_period, model, data_loader):
        self._model = model
        self._period = eval_period
        self._data_loader = data_loader

    def _do_loss_eval(self):
        total_losses = []
        for inputs in self._data_loader:
            with torch.no_grad():
                loss_batch = self._get_loss(inputs)
                total_losses.append(loss_batch)
        mean_loss = np.mean(total_losses)
        self.trainer.storage.put_scalar('validation_loss', mean_loss)
        comm.synchronize()
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

class APEvalHook(HookBase):
    def __init__(self, eval_period, model, data_loader, cfg):
        self._model = model
        self._period = eval_period
        self._data_loader = data_loader
        self._cfg = cfg
        self._evaluator = COCOEvaluator("cesm_val", self._cfg, False, output_dir=self._cfg.OUTPUT_DIR)

    def after_step(self):
        next_iter = self.trainer.iter + 1
        if self._period > 0 and next_iter % self._period == 0:
            self._do_eval()

    def _do_eval(self):
        self._evaluator.reset()
        for inputs in self._data_loader:
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            with torch.no_grad():
                outputs = self._model(inputs)
            self._evaluator.process(inputs, outputs)

        eval_results = self._evaluator.evaluate()

        if comm.is_main_process():
            if "segm" in eval_results and "AP" in eval_results["segm"]:
                ap = eval_results["segm"]["AP"]
                self.trainer.storage.put_scalar('AP', ap)
            else:
                print("Warning: 'segm' or 'AP' not found in evaluation results")
                print(f"Evaluation results: {eval_results}")

class Trainer(DefaultTrainer):
    def __init__(self, cfg):
        super().__init__(cfg)
        self.best_ap = 0
        self.best_model_path = None

    @classmethod
    def build_evaluator(cls, cfg, dataset_name, output_folder=None):
        if output_folder is None:
            output_folder = os.path.join(cfg.OUTPUT_DIR, "inference")
        return COCOEvaluator(dataset_name, cfg, True, output_folder)

    def build_hooks(self):
        hooks = super().build_hooks()
        hooks.append(LossEvalHook(
            20,  # Evaluate every 20 iterations
            self.model,
            build_detection_test_loader(
                self.cfg,
                self.cfg.DATASETS.TEST[0],
                DatasetMapper(self.cfg, True)
            )
        ))
        hooks.append(APEvalHook(
            500,  # Evaluate AP every 500 iterations
            self.model,
            build_detection_test_loader(
                self.cfg,
                self.cfg.DATASETS.TEST[0],
                DatasetMapper(self.cfg, True)
            ),
            self.cfg
        ))
        return hooks

    def after_step(self):
        super().after_step()
        # Check if we should save the model
        if self.iter % 500 == 0:  # Check every 500 iterations
            current_ap = self.storage.latest().get('AP', 0)
            train_loss = self.storage.latest().get('total_loss', float('inf'))
            val_loss = self.storage.latest().get('validation_loss', float('inf'))

            if current_ap > self.best_ap and val_loss < train_loss:
                self.best_ap = current_ap
                # Remove previous best model if it exists
                if self.best_model_path and os.path.exists(self.best_model_path):
                    os.remove(self.best_model_path)
                # Save new best model
                self.best_model_path = os.path.join(self.cfg.OUTPUT_DIR, f"model_best_ap_{self.iter}.pth")
                torch.save(self.model.state_dict(), self.best_model_path)

def setup_cfg(config_name, lr, max_iter, batch_size):
    cfg = get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file(f"COCO-InstanceSegmentation/{config_name}.yaml"))
    cfg.DATASETS.TRAIN = ("cesm_train",)
    cfg.DATASETS.TEST = ("cesm_val",)
    cfg.DATALOADER.NUM_WORKERS = 2
    cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(f"COCO-InstanceSegmentation/{config_name}.yaml")
    cfg.SOLVER.IMS_PER_BATCH = batch_size
    cfg.SOLVER.BASE_LR = lr
    cfg.SOLVER.MAX_ITER = max_iter
    cfg.SOLVER.STEPS = (int(0.7 * max_iter), int(0.9 * max_iter))
    cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 128
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 2
    cfg.TEST.EVAL_PERIOD = 500

    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    cfg.OUTPUT_DIR = f"./output_{config_name}_{lr}_{max_iter}_{batch_size}_{timestamp}"
    os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
    return cfg

def plot_losses_and_ap(output_dir):
    with open(os.path.join(output_dir, "metrics.json"), "r") as f:
        experiment_metrics = [json.loads(line) for line in f]

    train_iterations = [x['iteration'] for x in experiment_metrics if 'total_loss' in x]
    train_losses = [x['total_loss'] for x in experiment_metrics if 'total_loss' in x]
    val_iterations = [x['iteration'] for x in experiment_metrics if 'validation_loss' in x]
    val_losses = [x['validation_loss'] for x in experiment_metrics if 'validation_loss' in x]
    ap_iterations = [x['iteration'] for x in experiment_metrics if 'AP' in x]
    ap_values = [x['AP'] for x in experiment_metrics if 'AP' in x]

    # Plot losses
    plt.figure(figsize=(10, 5))
    plt.plot(train_iterations, train_losses, label='training_loss')
    plt.plot(val_iterations, val_losses, label='validation_loss')
    plt.xlabel('iteration')
    plt.ylabel('loss')
    plt.legend()
    plt.title('Training and Validation Loss')
    plt.savefig(os.path.join(output_dir, 'loss_plot.png'))
    plt.close()

    # Plot AP
    plt.figure(figsize=(10, 5))
    plt.plot(ap_iterations, ap_values, label='AP', color='r')
    plt.xlabel('iteration')
    plt.ylabel('AP')
    plt.legend()
    plt.title('Average Precision (AP)')
    plt.savefig(os.path.join(output_dir, 'ap_plot.png'))
    plt.close()

    return ap_iterations, ap_values

def train_and_evaluate(cfg):
    trainer = Trainer(cfg)
    trainer.resume_or_load(resume=False)
    trainer.train()
    ap_iterations, ap_values = plot_losses_and_ap(cfg.OUTPUT_DIR)

    # Evaluate on test set
    cfg.MODEL.WEIGHTS = os.path.join(cfg.OUTPUT_DIR, "model_final.pth")
    cfg.DATASETS.TEST = ("cesm_test",)
    evaluator = COCOEvaluator("cesm_test", cfg, False, output_dir=cfg.OUTPUT_DIR)
    results = trainer.test(cfg, trainer.model, evaluators=[evaluator])

    # Check if results contain AP
    if 'segm' in results and 'AP' in results['segm']:
        return results, ap_iterations, ap_values, trainer.best_model_path
    else:
        print("Warning: AP not found in evaluation results")
        return None, ap_iterations, ap_values, trainer.best_model_path

def main():
    register_coco_instances("cesm_train", {}, "./output/train_annotations.json", "../data/images")
    register_coco_instances("cesm_val", {}, "./output/val_annotations.json", "../data/images")
    register_coco_instances("cesm_test", {}, "./output/test_annotations.json", "../data/images")

    # List of hyperparameter configurations to try
    configs = [
        # Mask R-CNN with ResNet-50-FPN backbone
        ("mask_rcnn_R_50_FPN_3x", 0.001, 5000, 2),
        ("mask_rcnn_R_50_FPN_3x", 0.0005, 7000, 4),
        ("mask_rcnn_R_50_FPN_3x", 0.00025, 10000, 8),

        # Mask R-CNN with ResNet-101-FPN backbone
        ("mask_rcnn_R_101_FPN_3x", 0.001, 5000, 2),
        ("mask_rcnn_R_101_FPN_3x", 0.0005, 7000, 4),
        ("mask_rcnn_R_101_FPN_3x", 0.00025, 10000, 8),

        # Mask R-CNN with ResNeXt-101-32x8d-FPN backbone
        ("mask_rcnn_X_101_32x8d_FPN_3x", 0.001, 5000, 2),
        ("mask_rcnn_X_101_32x8d_FPN_3x", 0.0005, 7000, 4),
        ("mask_rcnn_X_101_32x8d_FPN_3x", 0.00025, 10000, 8),
    ]

    results_data = []
    best_ap = 0
    best_config = None
    best_model_path = None

    for config_name, lr, max_iter, batch_size in configs:
        print(f"Training with config: {config_name}, lr: {lr}, max_iter: {max_iter}, batch_size: {batch_size}")
        cfg = setup_cfg(config_name, lr, max_iter, batch_size)
        results, ap_iterations, ap_values, model_best_ap_path = train_and_evaluate(cfg)

        final_ap = results['segm']['AP']
        if final_ap > best_ap:
            best_ap = final_ap
            best_config = (config_name, lr, max_iter, batch_size)
            best_model_path = model_best_ap_path

        for iteration, ap in zip(ap_iterations, ap_values):
            results_data.append({
                'config_name': config_name,
                'learning_rate': lr,
                'max_iterations': max_iter,
                'batch_size': batch_size,
                'iteration': iteration,
                'AP': ap
            })

        # Remove the final model weights if it's not the best
        final_model_path = os.path.join(cfg.OUTPUT_DIR, "model_final.pth")
        if os.path.exists(final_model_path) and final_model_path != best_model_path:
            os.remove(final_model_path)

    # Create DataFrame and save to Excel
    df = pd.DataFrame(results_data)
    df.to_excel('ap_results.xlsx', index=False)
    print("AP results saved to ap_results.xlsx")

    print(f"Best configuration: {best_config}")
    print(f"Best AP: {best_ap}")
    print(f"Best model saved at: {best_model_path}")

if __name__ == "__main__":
    main()
