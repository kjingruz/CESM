# model_factory.py
import torchvision
from torchvision.models.detection import maskrcnn_resnet50_fpn

def create_maskrcnn_model(num_classes=3, pretrained=True):
    """
    Creates a Mask R-CNN model from torchvision with a ResNet-50 FPN backbone.
    num_classes: The number of classes (including background or excluding, depending on your labeling).
                 e.g. if you have 3 classes total (0=background, 1=Benign, 2=Malignant),
                 set num_classes=3 and your dataset should produce labels in [0,1,2].
    pretrained: Whether to initialize from COCO pretrained weights.
    """
    model = maskrcnn_resnet50_fpn(pretrained=pretrained)

    # Replace the box predictor
    in_features_box = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = \
        torchvision.models.detection.faster_rcnn.FastRCNNPredictor(in_features_box, num_classes)

    # Replace the mask predictor
    in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
    hidden_layer = 256
    model.roi_heads.mask_predictor = \
        torchvision.models.detection.mask_rcnn.MaskRCNNPredictor(
            in_features_mask,
            hidden_layer,
            num_classes
        )
    return model
