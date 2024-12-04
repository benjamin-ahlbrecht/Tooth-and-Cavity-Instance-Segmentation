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
from typing import Union, Tuple, Optional, Dict, List


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
        iou_threshold: float = 0.6,
        score_threshold: float = 0.5
    ):
    # Filter out low confidence bounding boxes: `score < score_threshold`
    indices_keep_scores = (output["scores"] >= score_threshold)
    output["boxes"] = output["boxes"][indices_keep_scores]
    output["labels"] = output["labels"][indices_keep_scores]
    output["scores"] = output["scores"][indices_keep_scores]
    output["masks"] = output["masks"][indices_keep_scores]

    # Perform Non-Max Suppression to help remove duplicate bounding boxes
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
    output["masks_raw"] = output["masks"]
    output["masks"] = output["masks"] >= 0.5
    
    # Extract segmentation information from our masks
    output["segmentation"] = []
    for mask in output["masks"]:
        mask = np.array(mask).astype(np.uint8)
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
        iou_threshold: float = 0.6,
        score_threshold: float = 0.5
    ):
    return [
        process_output(output, spatial_size, iou_threshold, score_threshold)
        for output in outputs
    ]


def mask_iou_matrix(masks_pred, masks_true):
    """Compute the IoU matrix between a set of segmentation masks.
    
    Parameters:
        masks_pred (torch.Tensor[bool]): The predicted segmentation masks.
        masks_true (torch.Tensor[bool]): The ground truth segmentation masks.
    
    Returns:
        iou_matrix (torch.Tensor[float]): The IoU Matrix where `iou_matrix[i, j]`
            is the IoU between predicted mask `i` and ground truth mask `j`.
    """
    n = len(masks_pred)
    m = len(masks_true)
    iou_matrix = torch.zeros((n, m), dtype=torch.float, device=masks_pred.device)

    for i, mask_pred in enumerate(masks_pred):
        for j, mask_true in enumerate(masks_true):
            intersection = torch.logical_and(mask_pred, mask_true).sum()
            union = torch.logical_or(mask_pred, mask_true).sum()
            iou_matrix[i, j] = intersection / union
    
    return iou_matrix


def match_predicted_and_true_masks(
        targets_pred: Dict,
        targets_true: Dict,
        iou_threshold: float = 0.5
    ):
    """Generate the confusion matrix beween the segmentations and labels of
    predicted targets and ground truth targets.

    Parameters:
        targets_pred (Dict[str, torch.Tensor]): The predicted outputs such that...
            - `targets_pred["masks"]` provide the predicted segmentation masks
            - `targets_pred["labels"]` provide the predicted labels
        targets_true (Dict[str, torch.Tensor]): The ground truth outputs. Expected
            to contain similar outputs as `targets_pred`.
        iou_threshold (float): The minimum IoU threshold required for a predicted
            mask to be considered a match (true positive).
    
    Returns:
        true_positives (List[Tuple[int, int, float]]): A list of true positive
            predictions indices. Each list contains a 3-tuple such that, for a
            true postive at index `i`, `idx_pred, idx_true, iou = true_positives[i]`.
        false_positives (List[int]): A list of false positive indices corresponding
            to the predictions.
        false_negatives (List[int]): A list of false negative indices corresponding
            to the ground truth.
    """
    masks_pred = targets_pred["masks"]
    masks_true = targets_true["masks"]

    labels_pred = targets_pred["labels"]
    labels_true = targets_true["labels"]

    iou_matrix = mask_iou_matrix(masks_pred, masks_true)

    # To avoid double-counting, keep track of which ground-truth indices we've matched
    matched = torch.zeros(len(masks_true), dtype=torch.bool, device=masks_true.device)

    # Iterate through each prediction label and IoU to see if we have a match (TP)
    true_positives = []
    false_positives = []
    for idx_pred, label_pred in enumerate(labels_pred):
        # Our goal is to find the ground truth index that maximizes our IoU
        max_iou = -1
        best_idx_true = -1

        for idx_true, (iou, label_true) in enumerate(zip(iou_matrix[idx_pred], labels_true)):
            if not matched[idx_true] and iou >= iou_threshold and label_pred == label_true:
                if iou > max_iou:
                    max_iou = iou
                    best_idx_true = idx_true
                    matched[idx_true] = True
        
        if best_idx_true >= 0:
            # Match found - TP
            true_positives.append((idx_pred, best_idx_true, max_iou.item()))
        else:
            # No match found - FP
            false_positives.append(idx_pred)
    
    # Remaining unmatched ground truth indices are false negatives
    false_negatives = torch.where(~matched)[0].tolist()

    return true_positives, false_positives, false_negatives


def f_score_from_counts(tp: int, fp: int, fn: int, beta: float = 1.0):
    """Compute the F-Score from raw confusion matrix counts.

    Parameters:
        tp (int): The number of true positives (TP).
        fp (int): The number of false positives (FP).
        fn (int): The number of false negatives (FN).
        beta (float): A weighting factor such that the recall is considered
            `beta` times as important as precision 
    
    Returns:
        f_score (float): The computed F-Score.
    """
    factor = (1 + beta**2) * tp
    denominator = beta**2 * fn + fp
    f_score = factor / (factor + denominator)
    return f_score


def f_score_from_matches(tp: List, fp: List, fn: List, beta: float = 1.0):
    """For details, see `f_score_from_counts`. Takes in output from
    `match_predicted_and_true_masks`.
    """
    tp = len(tp)
    fp = len(fp)
    fn = len(fn)
    return f_score_from_counts(tp, fp, fn, beta=beta)
