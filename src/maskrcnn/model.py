import torch
import torchvision
from torchvision.models.detection import maskrcnn_resnet50_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor


def load_model(ckpt, num_classes=34, use_cuda=False):
    """
    Loads our MaskRCNN model based on a ckpt
    """
    model = maskrcnn_resnet50_fpn()
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    
    if ckpt is not None:
        if use_cuda:
            ckpt = torch.load(ckpt)
        else:
            ckpt = torch.load(ckpt, map_location=torch.device("cpu"))
            
        model.load_state_dict(ckpt)
    return model