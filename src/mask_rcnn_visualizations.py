import matplotlib
import matplotlib.pyplot as plt
import torch

from typing import Optional


def plot_coco_image(
        image,
        target,
        plot_masks: bool = True,
        plot_bboxes: bool = True,
        plot_segmentations: bool = True,
        plot_category_id: bool = True,
        category_names: Optional[dict[int, str]] = None
    ):
    if image.shape[0] == 3:
        image = torch.moveaxis(image, 0, 2)

    bboxes = target["bbox"]
    masks = target["masks"]
    segmentations = target["segmentation"]
    categories = target["category_id"]

    fig, ax = plt.subplots(figsize=(8, 8), tight_layout=True)
    ax.imshow(image)
    ax.set_xticks([])
    ax.set_yticks([])

    if plot_masks:
        axis_add_masks(ax, masks)

    if plot_bboxes:
        axis_add_bboxes(ax, bboxes)

    if plot_segmentations:
        axis_add_segmentations(ax, segmentations)

    if plot_category_id:
        axis_add_category_id(ax, bboxes, categories, category_names)
        
    return fig


def axis_add_masks(ax, masks):
    for i, mask in enumerate(masks):
        ax.imshow(mask + 0.5 * i, alpha=mask * 0.25, cmap="tab10", vmin=0, vmax=len(masks) / 2)


def axis_add_bboxes(ax, bboxes):
    for bbox in bboxes:
        x, y, width, height = bbox
        patch = matplotlib.patches.Rectangle(
            (x, y),
            width,
            height,
            alpha=1,
            fill=False,
            edgecolor="red",
            linewidth=1,
            mouseover=True
        )
        ax.add_patch(patch)


def axis_add_segmentations(ax, segmentations):
    for segmentation in segmentations:
        x = [val for i, val in enumerate(segmentation[0]) if i % 2 == 0]
        y = [val for i, val in enumerate(segmentation[0]) if i % 2 != 0]
        ax.plot(x, y, color="black")


def axis_add_category_id(ax, bboxes, categories, category_names):
    for i, bbox in enumerate(bboxes):
        x, y, width, height = bbox
        category_id = categories[i]

        if category_names is not None:
            display_text = category_names[category_id]
        else:
            display_text = category_id
        
        ax.text(x+6, y-8, display_text, color="white", fontsize=6, backgroundcolor=(1, 0 ,0, 0.25))
