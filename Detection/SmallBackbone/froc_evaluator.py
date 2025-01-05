# froc_evaluator.py

import numpy as np
import torch
from detectron2.evaluation import DatasetEvaluator
from detectron2.structures import Boxes

def iou_box(b1, b2):
    """
    Compute IoU of two boxes [x1, y1, x2, y2].
    """
    xA = max(b1[0], b2[0])
    yA = max(b1[1], b2[1])
    xB = min(b1[2], b2[2])
    yB = min(b1[3], b2[3])

    interArea = max(0, (xB - xA)) * max(0, (yB - yA))
    area1 = (b1[2] - b1[0]) * (b1[3] - b1[1])
    area2 = (b2[2] - b2[0]) * (b2[3] - b2[1])
    iou = interArea / float(area1 + area2 - interArea + 1e-6)
    return iou

class FROCEvaluator(DatasetEvaluator):
    def __init__(self, dataset_name, iou_threshold=0.5):
        """
        iou_threshold: e.g. 0.5 for counting TPs
        """
        self.dataset_name = dataset_name
        self.iou_threshold = iou_threshold
        self.reset()

    def reset(self):
        # We'll store predictions & GT for all images
        self.all_predictions = []  # list of dicts
        self.all_gt = []          # list of dicts

    def process(self, inputs, outputs):
        """
        inputs: list of dict, each has "file_name", "height", "width", ...
        outputs: list of dict (model predictions) with "instances"
        """
        for inp, out in zip(inputs, outputs):
            # ground-truth boxes
            gt_boxes = []
            gt_classes = []
            if "instances" in inp:  # If available, might be in val set
                annos = inp["instances"].get_fields()
                # annos should be a dict of "gt_boxes", "gt_classes"
                boxes_xyxy = annos["gt_boxes"].tensor.cpu().numpy()  # shape [N, 4]
                classes = annos["gt_classes"].cpu().numpy()          # shape [N]
                for b, c in zip(boxes_xyxy, classes):
                    gt_boxes.append(b.tolist())  # [x1, y1, x2, y2]
                    gt_classes.append(int(c))

            # predictions
            pred_boxes = []
            scores = []
            pred_classes = []
            if "instances" in out:
                inst = out["instances"].to("cpu")
                pred_boxes_xyxy = inst.pred_boxes.tensor.numpy()  # shape [M, 4]
                sc = inst.scores.numpy()                          # shape [M]
                cls = inst.pred_classes.numpy()                   # shape [M]

                for b, s, c in zip(pred_boxes_xyxy, sc, cls):
                    pred_boxes.append(b.tolist())  # [x1, y1, x2, y2]
                    scores.append(float(s))
                    pred_classes.append(int(c))

            self.all_gt.append({
                "gt_boxes": gt_boxes,
                "gt_classes": gt_classes
            })
            self.all_predictions.append({
                "pred_boxes": pred_boxes,
                "scores": scores,
                "pred_classes": pred_classes
            })

    def evaluate(self):
        # Build arrays of all predictions with their scores, then match with GT
        # We'll do a simple FROC approach over many score thresholds
        iou_thresh = self.iou_threshold
        # Flatten all GT
        total_gt = 0
        for g in self.all_gt:
            total_gt += len(g["gt_boxes"])
        num_images = len(self.all_gt)

        # Collect predictions in one big list
        # [ (score, class, x1, y1, x2, y2, image_id) ]
        all_preds = []
        for img_id, preds in enumerate(self.all_predictions):
            for pb, sc, cls in zip(preds["pred_boxes"], preds["scores"], preds["pred_classes"]):
                all_preds.append((sc, cls, pb[0], pb[1], pb[2], pb[3], img_id))
        # Sort by descending score
        all_preds.sort(key=lambda x: x[0], reverse=True)

        # We'll define thresholds by unique scores or a fixed range
        # Simpler: let's define 100 thresholds between 0.0 and 1.0
        thr_list = np.linspace(1.0, 0.0, 101)
        froc_data = []  # list of (threshold, TPR, FP_per_image)

        for thr in thr_list:
            # For each GT, track matched state
            matched = [False]*total_gt  # or we can track per-image
            # We'll do a per-image approach: record each GT matched or not
            # Actually, let's store them in all_gt, but let's do a quick approach
            # We'll create an index offset for each image
            # For simplicity, let's flatten them "globally"
            # but let's do a simpler approach: match on the fly

            TP = 0
            FP = 0
            # We'll keep track of matched GT boxes (image_id, box_index)
            used_gt = set()

            for (sc, cls, x1, y1, x2, y2, img_id) in all_preds:
                if sc < thr:
                    break
                # see if we match any GT in that image
                gt_boxes_img = self.all_gt[img_id]["gt_boxes"]
                gt_classes_img = self.all_gt[img_id]["gt_classes"]
                found_match = False
                for gt_i, (gtb) in enumerate(gt_boxes_img):
                    if gt_classes_img[gt_i] != cls:
                        continue
                    if (img_id, gt_i) in used_gt:
                        continue
                    iou_val = iou_box([x1, y1, x2, y2], gtb)
                    if iou_val >= iou_thresh:
                        # We have a match
                        TP += 1
                        used_gt.add((img_id, gt_i))
                        found_match = True
                        break
                if not found_match:
                    FP += 1

            TPR = TP / float(total_gt) if total_gt > 0 else 0
            FP_per_image = FP / float(num_images)
            froc_data.append((thr, TPR, FP_per_image))

        # You can store or plot froc_data. We'll just return it in a dict
        results = {
            "froc_data": froc_data,  # list of (threshold, TPR, FP_per_image)
            "iou_threshold": iou_thresh
        }
        print("FROC results computed. Example top:", froc_data[:5])
        return {"FROC": results}
