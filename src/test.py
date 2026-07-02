import os
import csv
import time
import glob
import numpy
import torch
from torch.utils.data import DataLoader

from src.dataset import (
    DDOSDataset,
    CLASS_NAMES,
    CANONICAL_CLASS_IDS,
    get_thin_class_indices,
    normalization_for_model,
)
from src.model import build_model
from src.visualize import visualize_predictions, make_color_map


def compute_boundary(mask, dilation_ratio=0.02):
    import cv2
    h, w = mask.shape
    diag_len = numpy.sqrt(h**2 + w**2)
    dilation = max(1, int(round(dilation_ratio * diag_len)))
    kernel = numpy.ones((3, 3), numpy.uint8)
    dilated = cv2.dilate(mask.astype(numpy.uint8), kernel, iterations=dilation)
    eroded = cv2.erode(mask.astype(numpy.uint8), kernel, iterations=dilation)
    boundary = dilated - eroded
    return boundary


def safe_nanmean(values):
    values = numpy.asarray(values, dtype=numpy.float64)
    if values.size == 0 or numpy.all(numpy.isnan(values)):
        return float("nan")
    return float(numpy.nanmean(values))


def compute_tse(miou, biou, recall, fpr):
    return 0.45 * biou + 0.30 * recall - 0.15 * fpr + 0.10 * miou


class SegmentationMetricAccumulator:
    """
    Dataset-level metric accumulator.

    IoU/Recall/bIoU for absent classes are represented as NaN and ignored in
    macro means. False-positive rate is computed from the dataset-level
    confusion matrix.
    """
    def __init__(self, num_classes, ignore_index=-100, compute_boundary_iou=True):
        self.num_classes = int(num_classes)
        self.ignore_index = ignore_index
        self.compute_boundary_iou = compute_boundary_iou
        self.confusion = numpy.zeros((self.num_classes, self.num_classes), dtype=numpy.int64)
        self.boundary_intersections = numpy.zeros(self.num_classes, dtype=numpy.float64)
        self.boundary_unions = numpy.zeros(self.num_classes, dtype=numpy.float64)

    def update_from_logits(self, predictions, labels):
        preds = torch.argmax(predictions, dim=1).detach().cpu().numpy()
        labels_np = labels.detach().cpu().numpy()
        self.update(preds, labels_np)

    def update(self, preds, labels_np):
        valid = (labels_np != self.ignore_index) & (labels_np >= 0) & (labels_np < self.num_classes)
        labels_flat = labels_np[valid].astype(numpy.int64)
        preds_flat = preds[valid].astype(numpy.int64)

        valid_pred = (preds_flat >= 0) & (preds_flat < self.num_classes)
        labels_flat = labels_flat[valid_pred]
        preds_flat = preds_flat[valid_pred]

        if labels_flat.size > 0:
            encoded = self.num_classes * labels_flat + preds_flat
            counts = numpy.bincount(encoded, minlength=self.num_classes * self.num_classes)
            self.confusion += counts.reshape(self.num_classes, self.num_classes)

        if self.compute_boundary_iou:
            for i in range(preds.shape[0]):
                pred_mask = preds[i]
                true_mask = labels_np[i]
                valid_mask = true_mask != self.ignore_index
                for c in range(self.num_classes):
                    pred_boundary = compute_boundary((pred_mask == c) & valid_mask)
                    gt_boundary = compute_boundary((true_mask == c) & valid_mask)
                    intersection = numpy.logical_and(pred_boundary, gt_boundary).sum()
                    union = numpy.logical_or(pred_boundary, gt_boundary).sum()
                    self.boundary_intersections[c] += intersection
                    self.boundary_unions[c] += union

    def compute(self):
        tp = numpy.diag(self.confusion).astype(numpy.float64)
        fp = self.confusion.sum(axis=0).astype(numpy.float64) - tp
        fn = self.confusion.sum(axis=1).astype(numpy.float64) - tp
        total = self.confusion.sum().astype(numpy.float64)
        tn = total - tp - fp - fn

        iou_den = tp + fp + fn
        recall_den = tp + fn
        fpr_den = fp + tn

        per_class_iou = numpy.divide(tp, iou_den, out=numpy.full_like(tp, numpy.nan), where=iou_den > 0)
        per_class_recall = numpy.divide(tp, recall_den, out=numpy.full_like(tp, numpy.nan), where=recall_den > 0)
        per_class_fpr = numpy.divide(fp, fpr_den, out=numpy.full_like(tp, numpy.nan), where=fpr_den > 0)

        if self.compute_boundary_iou:
            per_class_biou = numpy.divide(
                self.boundary_intersections,
                self.boundary_unions,
                out=numpy.full_like(self.boundary_intersections, numpy.nan, dtype=numpy.float64),
                where=self.boundary_unions > 0,
            )
        else:
            per_class_biou = numpy.full(self.num_classes, numpy.nan, dtype=numpy.float64)

        miou = safe_nanmean(per_class_iou)
        recall = safe_nanmean(per_class_recall)
        fpr = safe_nanmean(per_class_fpr)
        biou = safe_nanmean(per_class_biou)
        tse = compute_tse(miou, biou, recall, fpr)

        return {
            "miou": miou,
            "per_class_iou": [float(x) if not numpy.isnan(x) else float("nan") for x in per_class_iou],
            "per_class_recall": [float(x) if not numpy.isnan(x) else float("nan") for x in per_class_recall],
            "per_class_fpr": [float(x) if not numpy.isnan(x) else float("nan") for x in per_class_fpr],
            "biou": biou,
            "per_class_biou": [float(x) if not numpy.isnan(x) else float("nan") for x in per_class_biou],
            "recall": recall,
            "fpr": fpr,
            "tse": tse,
        }


def compute_metrics(predictions, labels, num_classes, ignore_index=-100):
    accumulator = SegmentationMetricAccumulator(num_classes=num_classes, ignore_index=ignore_index)
    accumulator.update_from_logits(predictions, labels)
    return accumulator.compute()


def maybe_float_for_csv(value):
    try:
        if value is None or numpy.isnan(value):
            return ""
    except TypeError:
        pass
    return float(value)


def compute_sample_error_rows(predictions, labels, metadata, num_classes, ignore_index=-100):
    preds = torch.argmax(predictions, dim=1).cpu().numpy()
    labels_np = labels.cpu().numpy()
    rows = []
    for i in range(preds.shape[0]):
        pred_mask = preds[i]
        true_mask = labels_np[i]
        valid_mask = true_mask != ignore_index
        sample_key = metadata[i].get("sample_key", f"sample_{i}") if metadata else f"sample_{i}"
        for c in range(num_classes):
            pred_c = (pred_mask == c) & valid_mask
            true_c = (true_mask == c) & valid_mask
            intersection = (pred_c & true_c).sum()
            union = (pred_c | true_c).sum()
            tp = intersection
            fn = true_c.sum() - tp
            fp = pred_c.sum() - tp
            iou = intersection / union if union > 0 else float("nan")
            recall = tp / (tp + fn) if (tp + fn) > 0 else float("nan")
            class_name = CLASS_NAMES[c] if c < len(CLASS_NAMES) else f"class_{c}"
            rows.append({
                "sample_key": sample_key,
                "class_index": c,
                "original_class_id": CANONICAL_CLASS_IDS[c] if c < len(CANONICAL_CLASS_IDS) else "",
                "class_name": class_name,
                "gt_pixels": int(true_c.sum()),
                "pred_pixels": int(pred_c.sum()),
                "tp": int(tp),
                "fp": int(fp),
                "fn": int(fn),
                "iou": maybe_float_for_csv(iou),
                "recall": maybe_float_for_csv(recall),
            })
    return rows


def write_csv(path, rows):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    if not rows:
        return
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"[INFO] Saved {path}")


def metadata_collate(batch):
    images = torch.stack([item[0] for item in batch], dim=0)
    masks = torch.stack([item[1] for item in batch], dim=0)
    metadata = [item[2] for item in batch]
    return images, masks, metadata


def get_inference_timer(device):
    def sync():
        if str(device).startswith("cuda") and torch.cuda.is_available():
            torch.cuda.synchronize()
    return sync


def test_model(test_path, modality, device, edge_method, checkpoint_path, batch_size, visualize=False,
               results_dir="results", visualization_dir="visualizations", visualize_max=20, seed=None, run_name=None):
    print(f"\n=== Testing {modality.upper()} using checkpoint: {checkpoint_path} ===")
    os.makedirs(results_dir, exist_ok=True)

    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    checkpoint = torch.load(checkpoint_path, map_location=device)
    num_classes = checkpoint["num_classes"]
    in_channels = checkpoint["in_channels"]
    model_name = checkpoint.get("model_name")
    normalization = checkpoint.get("normalization", normalization_for_model(model_name))
    ignore_index = checkpoint.get("ignore_index", -100)

    test_dataset = DDOSDataset(dataset_path=test_path, modality=modality, edge_method=edge_method,
                               augment=False, return_metadata=True, normalization=normalization,
                               ignore_index=ignore_index)
    if test_dataset.num_classes != num_classes:
        raise ValueError(
            f"Checkpoint has {num_classes} classes but dataset uses {test_dataset.num_classes}. "
            "Check canonical class mapping and checkpoint compatibility."
        )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=metadata_collate,
        num_workers=0,
    )
    test_dataset.save_class_frequency_csv(os.path.join(results_dir, f"class_frequency_test_{model_name}_{modality}.csv"))

    model = build_model(model_name, num_classes=num_classes, in_channels=in_channels)
    model.load_state_dict(checkpoint["state_dict"])
    model.to(device).eval()

    if visualize:
        visualize_path = os.path.join(visualization_dir, f"{model_name}_{modality}")
        os.makedirs(visualize_path, exist_ok=True)
        print(f"[INFO] Saving visual results to: {visualize_path}")
    else:
        visualize_path = None

    accumulator = SegmentationMetricAccumulator(num_classes=num_classes, ignore_index=ignore_index)
    error_rows = []
    inference_time = 0.0
    inference_batches = 0
    saved_visualizations = 0
    sync = get_inference_timer(device)
    wall_start = time.perf_counter()

    with torch.no_grad():
        for batch_idx, (images, masks, metadata) in enumerate(test_loader):
            images, masks = images.to(device), masks.to(device)

            sync()
            forward_start = time.perf_counter()
            outputs = model(images)
            sync()
            inference_time += time.perf_counter() - forward_start
            inference_batches += 1

            accumulator.update_from_logits(outputs, masks)
            error_rows.extend(compute_sample_error_rows(outputs, masks, metadata, num_classes, ignore_index=ignore_index))

            if visualize:
                if visualize_max is None:
                    max_images = 20
                else:
                    max_images = visualize_max

                if max_images < 0 or saved_visualizations < max_images:
                    remaining = None if max_images < 0 else max_images - saved_visualizations
                    label_colors = make_color_map(num_classes)
                    saved_now = visualize_predictions(
                        images, masks, outputs, label_colors, visualize_path=visualize_path,
                        metadata=metadata, max_images=remaining, make_zoom=True,
                        target_classes=get_thin_class_indices()
                    )
                    saved_visualizations += saved_now

    total_wall_time = time.perf_counter() - wall_start
    inference_latency = inference_time / max(1, inference_batches)
    inference_fps = batch_size / inference_latency if inference_latency > 0 else 0.0

    mean_metrics = accumulator.compute()
    mean_metrics["fps"] = inference_fps
    mean_metrics["latency_ms"] = inference_latency * 1000.0
    mean_metrics["eval_wall_time_s"] = total_wall_time

    per_class_str = ", ".join([
        "nan" if numpy.isnan(iou) else f"{iou:.3f}" for iou in mean_metrics["per_class_iou"]
    ])
    print(
        f"Test Results => mean IoU: {mean_metrics['miou']:.5f} | "
        f"per-class IoU: [{per_class_str}] | boundary IoU: {mean_metrics['biou']:.5f} | "
        f"Recall: {mean_metrics['recall']:.5f} | FPR: {mean_metrics['fpr']:.5f} | "
        f"TSE: {mean_metrics['tse']:.5f} | Inference FPS: {inference_fps:>5.2f} | "
        f"Inference Latency: {inference_latency*1000:>7.2f} ms | Eval wall time: {total_wall_time:.2f}s"
    )

    result_row = {
        "model": model_name,
        "modality": modality,
        "miou": mean_metrics["miou"],
        "biou": mean_metrics["biou"],
        "recall": mean_metrics["recall"],
        "fpr": mean_metrics["fpr"],
        "tse": mean_metrics["tse"],
        "fps": mean_metrics["fps"],
        "latency_ms": mean_metrics["latency_ms"],
        "eval_wall_time_s": mean_metrics["eval_wall_time_s"],
        "best_epoch": checkpoint.get("best_epoch", ""),
        "best_val_tse": checkpoint.get("best_val_tse", ""),
        "seed": seed if seed is not None else checkpoint.get("seed", ""),
        "run_name": run_name if run_name is not None else checkpoint.get("run_name", ""),
        "normalization": normalization,
        "class_weight_max": checkpoint.get("class_weight_max", ""),
    }
    for c, iou in enumerate(mean_metrics["per_class_iou"]):
        class_name = CLASS_NAMES[c].replace(" ", "_") if c < len(CLASS_NAMES) else f"class_{c}"
        result_row[f"iou_{class_name}"] = maybe_float_for_csv(iou)
    for c, rec in enumerate(mean_metrics["per_class_recall"]):
        class_name = CLASS_NAMES[c].replace(" ", "_") if c < len(CLASS_NAMES) else f"class_{c}"
        result_row[f"recall_{class_name}"] = maybe_float_for_csv(rec)
    for c, biou in enumerate(mean_metrics["per_class_biou"]):
        class_name = CLASS_NAMES[c].replace(" ", "_") if c < len(CLASS_NAMES) else f"class_{c}"
        result_row[f"biou_{class_name}"] = maybe_float_for_csv(biou)

    write_csv(os.path.join(results_dir, f"test_summary_{model_name}_{modality}.csv"), [result_row])
    write_csv(os.path.join(results_dir, f"error_analysis_{model_name}_{modality}.csv"), error_rows)

    return mean_metrics


def summarize_results(results_dir="results"):
    summary_files = sorted(glob.glob(os.path.join(results_dir, "**", "test_summary_*.csv"), recursive=True))
    rows = []
    for path in summary_files:
        with open(path, "r", newline="") as f:
            reader = csv.DictReader(f)
            rows.extend(list(reader))
    if not rows:
        return

    numeric_fields = ["miou", "biou", "recall", "fpr", "tse", "fps", "latency_ms", "eval_wall_time_s"]
    higher_is_better = {"miou", "biou", "recall", "tse", "fps"}
    lower_is_better = {"fpr", "latency_ms", "eval_wall_time_s"}

    rows_sorted = sorted(rows, key=lambda r: float(r.get("tse", 0.0) or 0.0), reverse=True)
    combined_path = os.path.join(results_dir, "combined_test_results_ranked.csv")
    with open(combined_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows_sorted[0].keys()))
        writer.writeheader()
        writer.writerows(rows_sorted)
    print(f"[INFO] Saved ranked combined results: {combined_path}")

    modality_groups = {}
    for r in rows:
        modality_groups.setdefault(r["modality"], []).append(r)
    cue_rows = []
    for modality, group in modality_groups.items():
        item = {"modality": modality, "n_models": len(group)}
        for field in numeric_fields:
            vals = [float(g[field]) for g in group if field in g and g[field] not in ("", "nan", "NaN")]
            item[f"mean_{field}"] = float(numpy.mean(vals)) if vals else ""
            if vals:
                if field in lower_is_better:
                    item[f"best_{field}"] = float(numpy.min(vals))
                    item[f"worst_{field}"] = float(numpy.max(vals))
                else:
                    item[f"best_{field}"] = float(numpy.max(vals))
                    item[f"worst_{field}"] = float(numpy.min(vals))
            else:
                item[f"best_{field}"] = ""
                item[f"worst_{field}"] = ""
        cue_rows.append(item)
    cue_path = os.path.join(results_dir, "modality_cue_summary.csv")
    write_csv(cue_path, cue_rows)
