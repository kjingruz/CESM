import os
import random
import torch
import logging
from detectron2 import model_zoo
from detectron2.config import get_cfg
from detectron2.engine import DefaultTrainer, DefaultPredictor, HookBase
from detectron2.evaluation import COCOEvaluator, inference_on_dataset
from detectron2.data import build_detection_test_loader
from detectron2.data.datasets import register_coco_instances
from detectron2.utils.visualizer import Visualizer, ColorMode
from detectron2.data import MetadataCatalog, DatasetCatalog

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class EvaluationHook(HookBase):
    """
    Custom Hook to evaluate the model on the validation set every 'period' iterations.
    """
    def __init__(self, evaluator, val_loader, period=1000):
        """
        Args:
            evaluator: COCOEvaluator instance.
            val_loader: DataLoader for validation set.
            period: Frequency (in iterations) to run evaluation.
        """
        self.evaluator = evaluator
        self.val_loader = val_loader
        self.period = period

    def after_step(self):
        """
        Called after each step. Runs evaluation every 'period' iterations.
        """
        next_iter = self.trainer.iter + 1
        if next_iter % self.period == 0:
            logger.info(f"Running evaluation at iteration {next_iter}...")
            results = inference_on_dataset(self.trainer.model, self.val_loader, self.evaluator)
            logger.info(f"Evaluation results at iteration {next_iter}: {results}")

def setup_cfg():
    """
    Sets up the Detectron2 configuration.
    """
    cfg = get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file("COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml"))

    # Dataset registration
    cfg.DATASETS.TRAIN = ("cesm_train",)
    cfg.DATASETS.TEST = ("cesm_val",)  # Use validation set for evaluation during training

    cfg.DATALOADER.NUM_WORKERS = 2
    cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml")

    cfg.SOLVER.IMS_PER_BATCH = 2
    cfg.SOLVER.BASE_LR = 0.00025  # Default learning rate
    cfg.SOLVER.MAX_ITER = 20000  # Adjust based on your dataset size

    cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 128  # Default value
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 3  # 0: Normal, 1: Benign, 2: Malignant

    cfg.OUTPUT_DIR = "./output_with_normals"
    os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)

    # Include images without annotations
    cfg.DATALOADER.FILTER_EMPTY_ANNOTATIONS = False  # Important to include images without annotations

    # Disable saving of checkpoints by setting a high CHECKPOINT_PERIOD
    cfg.SOLVER.CHECKPOINT_PERIOD = 100000  # Set to a value higher than MAX_ITER to effectively disable

    # Disable learning rate scheduling steps
    cfg.SOLVER.STEPS = []  # Disable learning rate scheduling steps

    return cfg

def train_and_evaluate(cfg):
    """
    Trains the model and sets up periodic evaluation on the validation set.
    """
    # Initialize the trainer
    trainer = DefaultTrainer(cfg)
    trainer.resume_or_load(resume=False)

    # Set up the evaluator
    evaluator = COCOEvaluator("cesm_val", cfg, False, output_dir=cfg.OUTPUT_DIR)
    val_loader = build_detection_test_loader(cfg, "cesm_val")

    # Register the custom evaluation hook
    eval_hook = EvaluationHook(evaluator, val_loader, period=1000)
    trainer.register_hooks([eval_hook])

    # Start training
    trainer.train()

def main():
    """
    Main function to execute training and evaluation steps.
    """
    # Paths
    data_dir = './coco_annotations'  # Directory containing annotations
    image_dir = './images'  # Flat directory containing images

    if not os.path.exists(image_dir):
        logger.error(f"Image directory not found: {image_dir}")
        return

    # Register datasets using annotations
    register_coco_instances("cesm_train", {}, os.path.join(data_dir, "train_annotations.json"), image_dir)
    register_coco_instances("cesm_val", {}, os.path.join(data_dir, "val_annotations.json"), image_dir)
    register_coco_instances("cesm_test", {}, os.path.join(data_dir, "test_annotations.json"), image_dir)

    # Configuration for Model
    cfg = setup_cfg()

    # Training and Evaluation
    train_and_evaluate(cfg)

    # Optional: Visualize Test Predictions after training
    visualize_predictions(cfg, image_dir)

def visualize_predictions(cfg, image_dir, num_samples=4):
    """
    Visualizes predictions on the test set and saves the annotated images.
    """
    from detectron2.utils.visualizer import Visualizer
    from detectron2.utils.visualizer import ColorMode
    import matplotlib.pyplot as plt

    # Load the trained model
    model_final_path = os.path.join(cfg.OUTPUT_DIR, "model_final.pth")
    if not os.path.exists(model_final_path):
        logger.error(f"Trained model not found at {model_final_path}. Cannot visualize predictions.")
        return

    cfg.MODEL.WEIGHTS = model_final_path
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5  # Set a custom testing threshold
    predictor = DefaultPredictor(cfg)

    # Get test dataset
    test_dataset_dicts = DatasetCatalog.get("cesm_test")
    test_metadata = MetadataCatalog.get("cesm_test")

    # Ensure there are enough samples
    num_samples = min(num_samples, len(test_dataset_dicts))
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
        image_path = d["file_name"]
        full_image_path = os.path.join(image_dir, image_path)
        if not os.path.exists(full_image_path):
            logger.warning(f"Image not found: {full_image_path}")
            continue
        image = cv2.imread(full_image_path)
        if image is None:
            logger.warning(f"Failed to load image: {full_image_path}")
            continue
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
