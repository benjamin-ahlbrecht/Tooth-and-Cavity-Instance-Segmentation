import numpy as np
import cv2 as cv
import torch
import torchvision
torchvision.disable_beta_transforms_warning()
from torchvision.transforms import v2
from torchvision.datasets import (
    CocoDetection,
    wrap_dataset_for_transforms_v2
)

from pathlib import Path
from typing import Union, Tuple, Optional


CATEGORY_ID_TO_NAME = {
    0: "Null",
    1: "Caries",
    2: "Cavity",
    3: "Crack",
    4: "Tooth"
}

TRANSFORMS_AUGMENT = v2.Compose([
    v2.ToImageTensor(),
    v2.RandomPhotometricDistort(p=0.5),
    v2.SanitizeBoundingBox(),
    v2.ConvertImageDtype()
])

TRANSFORMS_PROCESS = v2.Compose([
    v2.ToImageTensor(),
    v2.SanitizeBoundingBox(),
    v2.ConvertImageDtype()
])


def get_cocodetection_dataset(
    data_path: Union[Path, str],
    annotation_path: Union[Path, str],
    train: bool = True
):
    if train:
        return wrap_dataset_for_transforms_v2(CocoDetection(
            data_path,
            annotation_path,
            transforms=TRANSFORMS_AUGMENT
        ))
    
    return wrap_dataset_for_transforms_v2(CocoDetection(
        data_path,
        annotation_path,
        transforms=TRANSFORMS_PROCESS
    ))


def custom_collate_function(batch, to_cuda: bool = True):
    images, targets = tuple(zip(*batch))

    if not to_cuda:
        return (images, targets)

    images = torch.stack(images).cuda()
    for target in targets:
        target["boxes"] = target["boxes"].cuda()
        target["bbox"] = torch.Tensor(target["bbox"]).cuda()
        target["labels"] = torch.Tensor(target["labels"]).cuda()
        
    return (images, targets)


def process_output(
        output,
        spatial_size: Optional[Tuple[int, int]] = None,
        iou_threshold: float = 0.5
    ):

    # Perform Non-Max Suppression to select bounding boxes to keep
    indices_keep = torchvision.ops.nms(output["boxes"], output["scores"], iou_threshold).cpu()

    # Move to CPU and remove unnecessary boxes
    for key, val in output.items():
        if isinstance(val, torch.Tensor):
            output[key] = val.cpu()

    output["boxes"] = output["boxes"][indices_keep]
    output["labels"] = output["labels"][indices_keep]
    output["scores"] = output["scores"][indices_keep]
    output["masks"] = output["masks"][indices_keep]

    if spatial_size is None:
        spatial_size = output["boxes"].shape[2:]

    # Add in the "bbox" key
    bbox = torchvision.datapoints.BoundingBox(output["boxes"], format="XYXY", spatial_size=spatial_size)
    output["bbox"] = v2.ConvertBoundingBoxFormat("XYWH")(bbox)

    # We can remove the 2nd dimension of our masks.
    output["masks"] = output["masks"].squeeze(1)
    output["masks"] = np.round(np.array(output["masks"])).astype(np.uint8)
    
    # Extract segmentation information from our masks
    output["segmentation"] = []
    for mask in output["masks"]:
        contours, hierarchy = cv.findContours(
            mask,
            mode=cv.RETR_EXTERNAL,
            method=cv.CHAIN_APPROX_NONE
        )
    
        segmentation = np.expand_dims(contours[-1].flatten(), 0)
        output["segmentation"].append(segmentation)

    # category_id is just a simple transformation of our labels
    output["category_id"] = [label.item() for label in output["labels"]]

    return output
    

def process_outputs(
        outputs,
        spatial_size: Optional[Tuple[int, int]] = None,
        iou_threshold: float = 0.5
    ):
    return [process_output(output, spatial_size, iou_threshold) for output in outputs]