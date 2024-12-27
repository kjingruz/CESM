# evaluate.py
import os
import json
import torch
import numpy as np
from torch.utils.data import DataLoader
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from dataset_builder import MaskRCNNDataset, detection_collate_fn
from model_factory import create_maskrcnn_model

def run_inference(model, dataloader, device="cuda"):
    model.eval()
    results = []
    with torch.no_grad():
        for images, targets in dataloader:
            images = [img.to(device) for img in images]
            outputs = model(images)  # list of dicts: boxes, labels, scores, masks
            # Convert each output to COCO-style detection
            # ...
    return results

def main():
    data_dir = "./images"
    coco_test_json = "./coco_annotations/test_annotations.json"
    test_dataset = MaskRCNNDataset(coco_test_json, data_dir)
    test_loader = DataLoader(test_dataset, batch_size=2, shuffle=False, collate_fn=detection_collate_fn)

    model = create_maskrcnn_model(num_classes=3, pretrained=False)
    model.load_state_dict(torch.load("./output_maskrcnn/model_final.pth"))
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)

    # Build COCO object for ground truth
    coco_gt = COCO(coco_test_json)

    # Inference => get predictions in COCO "results" format
    coco_results = run_inference(model, test_loader, device=device)
    # example: coco_results = [
    #   {
    #     "image_id": ...,
    #     "category_id": ...,
    #     "bbox": [...],
    #     "score": ...,
    #     "segmentation": ...
    #   },
    #   ...
    # ]

    # Evaluate
    coco_dt = coco_gt.loadRes(coco_results)
    coco_eval = COCOeval(coco_gt, coco_dt, iouType='segm')
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()

if __name__ == "__main__":
    main()
