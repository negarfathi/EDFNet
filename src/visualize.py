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
    """
    Stable semantic colors for paper figures.

    Red is intentionally not used for semantic classes because red is reserved
    for missed thin/ultra-thin pixels in the thin-error map.
    """
    base = [
        (0, 0, 0),        # Background
        (128, 128, 128),  # Other
        (0, 170, 0),      # Animals
        (0, 80, 255),     # Vehicles
        (255, 190, 0),    # Buildings
        (0, 100, 0),      # Trees
        (0, 220, 220),    # Large Mesh
        (255, 128, 0),    # Small Mesh
        (140, 70, 220),   # Thin Structures
        (255, 255, 255),  # Ultra-thin
    ]
    return {i: base[i % len(base)] for i in range(num_classes)}


def denormalize_rgb_for_display(img):
    """Convert the first three channels of a model input back to [0,255] RGB."""
    img = img[:, :, :3].astype(numpy.float32).copy()
    if img.min() < -0.01 or img.max() > 1.01:
        img = img * IMAGENET_STD.reshape(1, 1, 3) + IMAGENET_MEAN.reshape(1, 1, 3)
    return (img * 255.0).clip(0, 255).astype(numpy.uint8)


def mask_to_overlay(img, mask, label_colors, alpha=0.45):
    h, w = img.shape[:2]
    overlay = numpy.zeros((h, w, 3), dtype=numpy.uint8)
    for c, color in label_colors.items():
        overlay[mask == c] = color
    return cv2.addWeighted(img, 1.0 - alpha, overlay, alpha, 0)


def thin_error_map(gt_mask, pred_mask, target_classes=None):
    """
    Return a paper-oriented thin-error map.

    Green = correct thin/ultra-thin prediction
    Red   = missed thin/ultra-thin pixel
    Blue  = false-positive thin/ultra-thin prediction
    Black = not thin-related
    """
    if target_classes is None:
        target_classes = get_thin_class_indices()

    gt_thin = numpy.zeros(gt_mask.shape, dtype=bool)
    pred_thin = numpy.zeros(pred_mask.shape, dtype=bool)
    for c in target_classes:
        gt_thin |= gt_mask == c
        pred_thin |= pred_mask == c

    true_positive = gt_thin & pred_thin
    false_negative = gt_thin & ~pred_thin
    false_positive = ~gt_thin & pred_thin

    out = numpy.zeros((*gt_mask.shape, 3), dtype=numpy.uint8)
    out[true_positive] = (0, 200, 0)
    out[false_negative] = (255, 0, 0)
    out[false_positive] = (0, 90, 255)

    score = int(false_negative.sum() + false_positive.sum())
    thin_pixels = int(gt_thin.sum())
    pred_thin_pixels = int(pred_thin.sum())
    return out, {
        "thin_error_score": score,
        "thin_gt_pixels": thin_pixels,
        "thin_pred_pixels": pred_thin_pixels,
        "thin_tp_pixels": int(true_positive.sum()),
        "thin_fn_pixels": int(false_negative.sum()),
        "thin_fp_pixels": int(false_positive.sum()),
    }


def find_thin_error_bbox(gt_mask, pred_mask, target_classes=None, min_size=32, pad=18):
    if target_classes is None:
        target_classes = get_thin_class_indices()

    region = numpy.zeros(gt_mask.shape, dtype=bool)
    for c in target_classes:
        region |= (gt_mask == c) | (pred_mask == c)
    # Prefer true thin/ultra-thin errors, but fall back to all thin pixels.
    error_region = region & (gt_mask != pred_mask)
    if error_region.sum() > 0:
        region = error_region
    if region.sum() == 0:
        region = gt_mask != pred_mask
    if region.sum() == 0:
        h, w = gt_mask.shape
        return 0, 0, w - 1, h - 1

    ys, xs = numpy.where(region)
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


def save_four_panel_figure(panels, titles, out_file, suptitle=None, figsize=(14, 4)):
    fig, axes = matplotlib.pyplot.subplots(1, 4, figsize=figsize)
    for ax, panel, title in zip(axes, panels, titles):
        ax.imshow(panel)
        ax.set_title(title)
        ax.axis("off")
    if suptitle:
        fig.suptitle(suptitle, fontsize=11)
    matplotlib.pyplot.savefig(out_file, bbox_inches="tight", dpi=250)
    matplotlib.pyplot.close(fig)


def visualize_predictions(images, labels, predictions, label_colors=None, visualize_path=None, metadata=None,
                          max_images=None, make_zoom=True, target_classes=None):
    """
    Save paper-oriented qualitative figures and return a list of saved records.

    Full figure:
        RGB | Ground truth | Prediction | Thin-error map

    Zoom figure:
        Zoomed RGB | Zoomed GT | Zoomed Prediction | Zoomed thin-error map

    A zoom figure is saved only when the sample has thin/ultra-thin error.
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

    saved_records = []
    saved_in_this_call = 0
    for i in range(len(imgs)):
        if max_images is not None and max_images >= 0 and saved_in_this_call >= max_images:
            return saved_records

        img = denormalize_rgb_for_display(imgs[i])
        gt_mask = labels_np[i]
        pred_mask = preds[i]
        sample_key = metadata[i].get("sample_key", f"sample_{_global_counter}") if metadata is not None and i < len(metadata) else f"sample_{_global_counter}"
        safe_key = safe_filename(sample_key)

        gt_overlay = mask_to_overlay(img, gt_mask, label_colors)
        pred_overlay = mask_to_overlay(img, pred_mask, label_colors)
        thin_map, thin_info = thin_error_map(gt_mask, pred_mask, target_classes=target_classes)
        bbox = find_thin_error_bbox(gt_mask, pred_mask, target_classes=target_classes)

        img_box = draw_bbox(img, bbox)
        gt_box = draw_bbox(gt_overlay, bbox)
        pred_box = draw_bbox(pred_overlay, bbox)
        thin_box = draw_bbox(thin_map, bbox)

        full_path = ""
        zoom_path = ""

        if visualize_path:
            full_path = unique_path(os.path.join(
                visualize_path,
                f"full_{_global_counter:04d}_{safe_key}.png"
            ))
            save_four_panel_figure(
                panels=[img_box, gt_box, pred_box, thin_box],
                titles=["RGB", "Ground truth", "Prediction", "Thin-error map"],
                out_file=full_path,
                suptitle=f"Qualitative result: {sample_key}",
                figsize=(14, 4),
            )
            print(f"[INFO] Saved {full_path}")

            if make_zoom and thin_info["thin_error_score"] > 0:
                x1, y1, x2, y2 = bbox
                zoom_path = unique_path(os.path.join(
                    visualize_path,
                    f"zoom_{_global_counter:04d}_{safe_key}.png"
                ))
                save_four_panel_figure(
                    panels=[
                        img[y1:y2 + 1, x1:x2 + 1],
                        gt_overlay[y1:y2 + 1, x1:x2 + 1],
                        pred_overlay[y1:y2 + 1, x1:x2 + 1],
                        thin_map[y1:y2 + 1, x1:x2 + 1],
                    ],
                    titles=["Zoomed RGB", "Zoomed GT", "Zoomed prediction", "Zoomed thin-error"],
                    out_file=zoom_path,
                    suptitle=f"Zoomed thin/error region: {sample_key}",
                    figsize=(14, 4),
                )
                print(f"[INFO] Saved {zoom_path}")

        saved_records.append({
            "sample_key": sample_key,
            "visualization_full_path": full_path,
            "visualization_zoom_path": zoom_path,
            **thin_info,
        })
        _global_counter += 1
        saved_in_this_call += 1

    return saved_records
