import os
import cv2
import numpy
import torch
import matplotlib.pyplot

from src.dataset import IMAGENET_MEAN, IMAGENET_STD, get_thin_class_indices

_global_counter = 0


def safe_filename(text):
    text = str(text)
    keep = []
    for ch in text:
        if ch.isalnum() or ch in ["-", "_", "."]:
            keep.append(ch)
        else:
            keep.append("_")
    name = "".join(keep).strip("_")
    return name if name else "sample"


def unique_path(path):
    """Return a non-existing path by adding _001, _002, ... if needed."""
    path = str(path)
    if not os.path.exists(path):
        return path
    root, ext = os.path.splitext(path)
    idx = 1
    while True:
        candidate = f"{root}_{idx:03d}{ext}"
        if not os.path.exists(candidate):
            return candidate
        idx += 1


def make_color_map(num_classes):
    base = [
        (0, 0, 0),
        (255, 0, 0),
        (0, 255, 0),
        (0, 0, 255),
        (255, 255, 0),
        (255, 0, 255),
        (0, 255, 255),
        (255, 128, 0),
        (128, 0, 255),
        (255, 255, 255),
    ]
    return {i: base[i % len(base)] for i in range(num_classes)}


def denormalize_rgb_for_display(img):
    """Convert the first three channels of a model input back to [0,255] RGB."""
    img = img[:, :, :3].astype(numpy.float32).copy()
    # Pretrained inputs have ImageNet-normalized RGB, which usually includes
    # negative values or values > 1. Reverse it for visualization.
    if img.min() < -0.01 or img.max() > 1.01:
        img = img * IMAGENET_STD.reshape(1, 1, 3) + IMAGENET_MEAN.reshape(1, 1, 3)
    return (img * 255.0).clip(0, 255).astype(numpy.uint8)


def mask_to_overlay(img, mask, label_colors, alpha=0.45):
    h, w = img.shape[:2]
    overlay = numpy.zeros((h, w, 3), dtype=numpy.uint8)
    for c, color in label_colors.items():
        overlay[mask == c] = color
    return cv2.addWeighted(img, 1.0 - alpha, overlay, alpha, 0)


def find_error_bbox(gt_mask, pred_mask, target_classes=None, min_size=32, pad=18):
    if target_classes is None:
        target_classes = get_thin_class_indices()

    error = numpy.zeros(gt_mask.shape, dtype=bool)
    for c in target_classes:
        error |= (gt_mask == c) | (pred_mask == c)
    error &= (gt_mask != pred_mask)

    if error.sum() == 0:
        error = gt_mask != pred_mask
    if error.sum() == 0:
        h, w = gt_mask.shape
        return 0, 0, w, h

    ys, xs = numpy.where(error)
    x1, x2 = int(xs.min()), int(xs.max())
    y1, y2 = int(ys.min()), int(ys.max())

    x1 = max(0, x1 - pad)
    y1 = max(0, y1 - pad)
    x2 = min(gt_mask.shape[1] - 1, x2 + pad)
    y2 = min(gt_mask.shape[0] - 1, y2 + pad)

    if x2 - x1 < min_size:
        extra = (min_size - (x2 - x1)) // 2
        x1 = max(0, x1 - extra)
        x2 = min(gt_mask.shape[1] - 1, x2 + extra)
    if y2 - y1 < min_size:
        extra = (min_size - (y2 - y1)) // 2
        y1 = max(0, y1 - extra)
        y2 = min(gt_mask.shape[0] - 1, y2 + extra)

    return x1, y1, x2, y2


def draw_bbox(image, bbox, color=(255, 0, 0), thickness=2):
    out = image.copy()
    x1, y1, x2, y2 = bbox
    cv2.rectangle(out, (x1, y1), (x2, y2), color, thickness)
    return out


def visualize_predictions(images, labels, predictions, label_colors=None, visualize_path=None, metadata=None,
                          max_images=None, make_zoom=True, target_classes=None):
    """
    Save qualitative figures and return the number of images saved in this call.
    max_images is now a caller-provided remaining budget, so the total cap can be
    enforced once per model/modality instead of once per batch.
    """
    global _global_counter

    preds = torch.argmax(predictions, dim=1).cpu().numpy()
    imgs = images.permute(0, 2, 3, 1).cpu().numpy()
    labels_np = labels.cpu().numpy()

    if label_colors is None:
        label_colors = make_color_map(predictions.shape[1])

    if target_classes is None:
        target_classes = get_thin_class_indices()

    if visualize_path:
        os.makedirs(visualize_path, exist_ok=True)

    saved_in_this_call = 0
    for i in range(len(imgs)):
        if max_images is not None and max_images >= 0 and saved_in_this_call >= max_images:
            return saved_in_this_call

        img = denormalize_rgb_for_display(imgs[i])
        gt_mask = labels_np[i]
        pred_mask = preds[i]

        gt_overlay = mask_to_overlay(img, gt_mask, label_colors)
        pred_overlay = mask_to_overlay(img, pred_mask, label_colors)

        bbox = find_error_bbox(gt_mask, pred_mask, target_classes=target_classes)
        img_box = draw_bbox(img, bbox)
        gt_box = draw_bbox(gt_overlay, bbox)
        pred_box = draw_bbox(pred_overlay, bbox)

        if make_zoom:
            x1, y1, x2, y2 = bbox
            zoom_rgb = img[y1:y2 + 1, x1:x2 + 1]
            zoom_gt = gt_overlay[y1:y2 + 1, x1:x2 + 1]
            zoom_pred = pred_overlay[y1:y2 + 1, x1:x2 + 1]
            fig, axes = matplotlib.pyplot.subplots(2, 3, figsize=(13, 8))
            panels = [img_box, gt_box, pred_box, zoom_rgb, zoom_gt, zoom_pred]
            titles = ["RGB with thin/error box", "Ground truth", "Prediction",
                      "Zoomed RGB", "Zoomed ground truth", "Zoomed prediction"]
            for ax, panel, title in zip(axes.ravel(), panels, titles):
                ax.imshow(panel)
                ax.set_title(title)
                ax.axis("off")
        else:
            fig, axes = matplotlib.pyplot.subplots(1, 3, figsize=(13, 4))
            panels = [img_box, gt_overlay, pred_overlay]
            titles = ["RGB with box", "Ground truth", "Prediction"]
            for ax, panel, title in zip(axes, panels, titles):
                ax.imshow(panel)
                ax.set_title(title)
                ax.axis("off")

        if metadata is not None and i < len(metadata):
            sample_key = metadata[i].get("sample_key", f"sample_{_global_counter}")
            fig.suptitle(f"Qualitative/failure example: {sample_key}", fontsize=11)

        if visualize_path:
            if metadata is not None and i < len(metadata):
                sample_key = safe_filename(metadata[i].get("sample_key", f"sample_{_global_counter}"))
            else:
                sample_key = f"sample_{_global_counter}"

            out_file = os.path.join(
                visualize_path,
                f"vis_zoom_failure_{_global_counter:04d}_{sample_key}.png"
            )
            out_file = unique_path(out_file)
            matplotlib.pyplot.savefig(out_file, bbox_inches="tight", dpi=200)
            print(f"[INFO] Saved {out_file}")
        matplotlib.pyplot.close(fig)
        _global_counter += 1
        saved_in_this_call += 1

    return saved_in_this_call
