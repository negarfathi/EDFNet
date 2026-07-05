import os
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
    is_pretrained_model,
)
from src.model import build_model
from src.visualize import visualize_predictions, make_color_map
from src.output import (
    append_csv_row,
    append_csv_rows,
    csv_path,
    ensure_outputs_structure,
    read_csv,
    safe_float,
    write_csv,
)


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


def maybe_float_for_csv(value):
    try:
        if value is None or numpy.isnan(value):
            return ""
    except TypeError:
        pass
    return float(value)


class SegmentationMetricAccumulator:
    """
    Dataset-level metric accumulator.

    IoU/Recall/Precision/bIoU for absent classes are represented as NaN and
    ignored in macro means. False-positive rate is computed from the dataset-
    level confusion matrix.
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

    def counts(self):
        tp = numpy.diag(self.confusion).astype(numpy.float64)
        fp = self.confusion.sum(axis=0).astype(numpy.float64) - tp
        fn = self.confusion.sum(axis=1).astype(numpy.float64) - tp
        total = self.confusion.sum().astype(numpy.float64)
        tn = total - tp - fp - fn
        return tp, fp, fn, tn

    def compute(self):
        tp, fp, fn, tn = self.counts()

        iou_den = tp + fp + fn
        recall_den = tp + fn
        precision_den = tp + fp
        fpr_den = fp + tn

        per_class_iou = numpy.divide(tp, iou_den, out=numpy.full_like(tp, numpy.nan), where=iou_den > 0)
        per_class_recall = numpy.divide(tp, recall_den, out=numpy.full_like(tp, numpy.nan), where=recall_den > 0)
        per_class_precision = numpy.divide(tp, precision_den, out=numpy.full_like(tp, numpy.nan), where=precision_den > 0)
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
        precision = safe_nanmean(per_class_precision)
        fpr = safe_nanmean(per_class_fpr)
        biou = safe_nanmean(per_class_biou)
        tse = compute_tse(miou, biou, recall, fpr)

        return {
            "miou": miou,
            "per_class_iou": [float(x) if not numpy.isnan(x) else float("nan") for x in per_class_iou],
            "per_class_recall": [float(x) if not numpy.isnan(x) else float("nan") for x in per_class_recall],
            "per_class_precision": [float(x) if not numpy.isnan(x) else float("nan") for x in per_class_precision],
            "per_class_fpr": [float(x) if not numpy.isnan(x) else float("nan") for x in per_class_fpr],
            "biou": biou,
            "per_class_biou": [float(x) if not numpy.isnan(x) else float("nan") for x in per_class_biou],
            "recall": recall,
            "precision": precision,
            "fpr": fpr,
            "tse": tse,
            "tp": [int(x) for x in tp],
            "fp": [int(x) for x in fp],
            "fn": [int(x) for x in fn],
            "tn": [int(x) for x in tn],
        }


def compute_metrics(predictions, labels, num_classes, ignore_index=-100):
    accumulator = SegmentationMetricAccumulator(num_classes=num_classes, ignore_index=ignore_index)
    accumulator.update_from_logits(predictions, labels)
    return accumulator.compute()


def model_pretrained_flag(model_name):
    return int(is_pretrained_model(model_name))


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


def compute_sample_error_rows(predictions, labels, metadata, num_classes, ignore_index=-100,
                              model_name="", modality="", seed="", run_name=""):
    preds = torch.argmax(predictions, dim=1).cpu().numpy()
    labels_np = labels.cpu().numpy()
    thin_indices = get_thin_class_indices()
    rows = []

    for i in range(preds.shape[0]):
        pred_mask = preds[i]
        true_mask = labels_np[i]
        sample_meta = metadata[i] if metadata and i < len(metadata) else {}
        sample_key = sample_meta.get("sample_key", f"sample_{i}")

        accumulator = SegmentationMetricAccumulator(num_classes=num_classes, ignore_index=ignore_index)
        accumulator.update(pred_mask[None, :, :], true_mask[None, :, :])
        metrics = accumulator.compute()

        thin_iou_values = [metrics["per_class_iou"][c] for c in thin_indices if c < len(metrics["per_class_iou"])]
        thin_recall_values = [metrics["per_class_recall"][c] for c in thin_indices if c < len(metrics["per_class_recall"])]
        thin_precision_values = [metrics["per_class_precision"][c] for c in thin_indices if c < len(metrics["per_class_precision"])]

        thin_tp = sum(metrics["tp"][c] for c in thin_indices if c < len(metrics["tp"]))
        thin_fp = sum(metrics["fp"][c] for c in thin_indices if c < len(metrics["fp"]))
        thin_fn = sum(metrics["fn"][c] for c in thin_indices if c < len(metrics["fn"]))

        per_class_iou = metrics["per_class_iou"]
        finite_ious = [(idx, value) for idx, value in enumerate(per_class_iou) if not numpy.isnan(value)]
        if finite_ious:
            worst_idx, worst_iou = min(finite_ious, key=lambda item: item[1])
            worst_name = CLASS_NAMES[worst_idx] if worst_idx < len(CLASS_NAMES) else f"class_{worst_idx}"
        else:
            worst_idx, worst_iou, worst_name = "", float("nan"), ""

        paper_candidate_score = int(thin_fn + thin_fp)
        if paper_candidate_score > 0:
            candidate_type = "thin_error_case"
        elif metrics["tse"] >= 0.75:
            candidate_type = "success_case"
        else:
            candidate_type = "typical_or_failure_case"

        rows.append({
            "model": model_name,
            "pretrained": model_pretrained_flag(model_name),
            "modality": modality,
            "run_name": run_name,
            "seed": seed,
            "sample_key": sample_key,
            "image_path": sample_meta.get("rgb_path", ""),
            "miou": maybe_float_for_csv(metrics["miou"]),
            "biou": maybe_float_for_csv(metrics["biou"]),
            "recall": maybe_float_for_csv(metrics["recall"]),
            "precision": maybe_float_for_csv(metrics["precision"]),
            "fpr": maybe_float_for_csv(metrics["fpr"]),
            "tse": maybe_float_for_csv(metrics["tse"]),
            "thin_iou": maybe_float_for_csv(safe_nanmean(thin_iou_values)),
            "ultra_thin_iou": maybe_float_for_csv(metrics["per_class_iou"][thin_indices[-1]] if thin_indices else float("nan")),
            "thin_recall": maybe_float_for_csv(safe_nanmean(thin_recall_values)),
            "thin_precision": maybe_float_for_csv(safe_nanmean(thin_precision_values)),
            "thin_tp_pixels": int(thin_tp),
            "thin_fp_pixels": int(thin_fp),
            "thin_fn_pixels": int(thin_fn),
            "worst_class_name": worst_name,
            "worst_class_iou": maybe_float_for_csv(worst_iou),
            "visualization_full_path": "",
            "visualization_zoom_path": "",
            "paper_candidate_type": candidate_type,
            "paper_candidate_score": paper_candidate_score,
        })
    return rows


OVERALL_FIELDS = [
    "rank_by_tse", "model", "pretrained", "modality", "run_name", "seed", "checkpoint_epoch",
    "miou", "biou", "recall", "precision", "fpr", "fps", "latency_ms", "tse",
    "eval_wall_time_s", "best_val_tse", "normalization", "class_weight_max",
]

PER_CLASS_FIELDS = [
    "model", "pretrained", "modality", "run_name", "seed", "class_id", "class_name",
    "is_thin_class", "iou", "recall", "precision", "fpr", "gt_pixels", "pred_pixels",
    "tp_pixels", "fp_pixels", "fn_pixels",
]

THIN_FIELDS = [
    "rank_by_mean_thin_iou", "model", "pretrained", "modality", "run_name", "seed",
    "thin_structures_iou", "ultra_thin_iou", "mean_thin_iou",
    "thin_structures_recall", "ultra_thin_recall", "mean_thin_recall",
    "thin_structures_precision", "ultra_thin_precision", "mean_thin_precision",
    "thin_structures_fpr", "ultra_thin_fpr", "mean_thin_fpr",
    "thin_gt_pixels", "ultra_thin_gt_pixels",
]

ERROR_FIELDS = [
    "model", "pretrained", "modality", "run_name", "seed", "sample_key", "image_path",
    "miou", "biou", "recall", "precision", "fpr", "tse",
    "thin_iou", "ultra_thin_iou", "thin_recall", "thin_precision",
    "thin_tp_pixels", "thin_fp_pixels", "thin_fn_pixels",
    "worst_class_name", "worst_class_iou",
    "visualization_full_path", "visualization_zoom_path",
    "paper_candidate_type", "paper_candidate_score",
]

CHECK_FIELDS = [
    "stage", "model", "pretrained", "modality", "run_name", "seed",
    "train_samples", "validation_samples", "test_samples", "num_classes",
    "expected_num_classes", "checkpoint_exists", "checkpoint_epoch", "test_completed",
    "all_metrics_finite", "unknown_labels_found", "prediction_shape_ok", "mask_shape_ok",
    "logits_class_count_ok", "visualizations_saved", "status", "message",
]


def per_class_rows_from_metrics(metrics, model_name, modality, seed, run_name):
    rows = []
    thin_indices = set(get_thin_class_indices())
    for c, class_id in enumerate(CANONICAL_CLASS_IDS):
        tp = int(metrics["tp"][c])
        fp = int(metrics["fp"][c])
        fn = int(metrics["fn"][c])
        rows.append({
            "model": model_name,
            "pretrained": model_pretrained_flag(model_name),
            "modality": modality,
            "run_name": run_name,
            "seed": seed,
            "class_id": class_id,
            "class_name": CLASS_NAMES[c],
            "is_thin_class": int(c in thin_indices),
            "iou": maybe_float_for_csv(metrics["per_class_iou"][c]),
            "recall": maybe_float_for_csv(metrics["per_class_recall"][c]),
            "precision": maybe_float_for_csv(metrics["per_class_precision"][c]),
            "fpr": maybe_float_for_csv(metrics["per_class_fpr"][c]),
            "gt_pixels": int(tp + fn),
            "pred_pixels": int(tp + fp),
            "tp_pixels": tp,
            "fp_pixels": fp,
            "fn_pixels": fn,
        })
    return rows


def thin_row_from_metrics(metrics, model_name, modality, seed, run_name):
    thin_indices = get_thin_class_indices()
    thin_struct_idx = thin_indices[0]
    ultra_idx = thin_indices[1]

    def metric_list(name):
        return [metrics[name][c] for c in thin_indices if c < len(metrics[name])]

    thin_gt_pixels = int(metrics["tp"][thin_struct_idx] + metrics["fn"][thin_struct_idx])
    ultra_gt_pixels = int(metrics["tp"][ultra_idx] + metrics["fn"][ultra_idx])

    return {
        "rank_by_mean_thin_iou": "",
        "model": model_name,
        "pretrained": model_pretrained_flag(model_name),
        "modality": modality,
        "run_name": run_name,
        "seed": seed,
        "thin_structures_iou": maybe_float_for_csv(metrics["per_class_iou"][thin_struct_idx]),
        "ultra_thin_iou": maybe_float_for_csv(metrics["per_class_iou"][ultra_idx]),
        "mean_thin_iou": maybe_float_for_csv(safe_nanmean(metric_list("per_class_iou"))),
        "thin_structures_recall": maybe_float_for_csv(metrics["per_class_recall"][thin_struct_idx]),
        "ultra_thin_recall": maybe_float_for_csv(metrics["per_class_recall"][ultra_idx]),
        "mean_thin_recall": maybe_float_for_csv(safe_nanmean(metric_list("per_class_recall"))),
        "thin_structures_precision": maybe_float_for_csv(metrics["per_class_precision"][thin_struct_idx]),
        "ultra_thin_precision": maybe_float_for_csv(metrics["per_class_precision"][ultra_idx]),
        "mean_thin_precision": maybe_float_for_csv(safe_nanmean(metric_list("per_class_precision"))),
        "thin_structures_fpr": maybe_float_for_csv(metrics["per_class_fpr"][thin_struct_idx]),
        "ultra_thin_fpr": maybe_float_for_csv(metrics["per_class_fpr"][ultra_idx]),
        "mean_thin_fpr": maybe_float_for_csv(safe_nanmean(metric_list("per_class_fpr"))),
        "thin_gt_pixels": thin_gt_pixels,
        "ultra_thin_gt_pixels": ultra_gt_pixels,
    }


def test_model(test_path, modality, device, edge_method, checkpoint_path, batch_size, visualize=False,
               outputs_dir="outputs", visualization_dir="visualizations", visualize_max=20, seed=None, run_name=None):
    print(f"\n=== Testing {modality.upper()} using checkpoint: {checkpoint_path} ===")
    ensure_outputs_structure(outputs_dir)

    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    checkpoint = torch.load(checkpoint_path, map_location=device)
    num_classes = checkpoint["num_classes"]
    in_channels = checkpoint["in_channels"]
    model_name = checkpoint.get("model_name")
    normalization = checkpoint.get("normalization", normalization_for_model(model_name))
    ignore_index = checkpoint.get("ignore_index", -100)
    seed_value = seed if seed is not None else checkpoint.get("seed", "")
    run_value = run_name if run_name is not None else checkpoint.get("run_name", "")

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
    prediction_shape_ok = 1
    mask_shape_ok = 1
    logits_class_count_ok = 1

    with torch.no_grad():
        for batch_idx, (images, masks, metadata) in enumerate(test_loader):
            images, masks = images.to(device), masks.to(device)

            sync()
            forward_start = time.perf_counter()
            outputs = model(images)
            sync()
            inference_time += time.perf_counter() - forward_start
            inference_batches += 1

            if outputs.ndim != 4:
                prediction_shape_ok = 0
            if masks.ndim != 3:
                mask_shape_ok = 0
            if outputs.shape[1] != num_classes:
                logits_class_count_ok = 0

            accumulator.update_from_logits(outputs, masks)

            batch_error_rows = compute_sample_error_rows(
                outputs, masks, metadata, num_classes, ignore_index=ignore_index,
                model_name=model_name, modality=modality, seed=seed_value, run_name=run_value
            )

            if visualize:
                max_images = visualize_max if visualize_max is not None else 20
                if max_images < 0 or saved_visualizations < max_images:
                    remaining = None if max_images < 0 else max_images - saved_visualizations
                    label_colors = make_color_map(num_classes)
                    vis_records = visualize_predictions(
                        images, masks, outputs, label_colors, visualize_path=visualize_path,
                        metadata=metadata, max_images=remaining, make_zoom=True,
                        target_classes=get_thin_class_indices()
                    )
                    saved_visualizations += len(vis_records)
                    vis_by_key = {record["sample_key"]: record for record in vis_records}
                    for row in batch_error_rows:
                        record = vis_by_key.get(row["sample_key"])
                        if record:
                            row["visualization_full_path"] = record.get("visualization_full_path", "")
                            row["visualization_zoom_path"] = record.get("visualization_zoom_path", "")
                            row["paper_candidate_score"] = record.get("thin_error_score", row["paper_candidate_score"])

            error_rows.extend(batch_error_rows)

    total_wall_time = time.perf_counter() - wall_start
    inference_latency = inference_time / max(1, inference_batches)
    inference_fps = batch_size / latency if (latency := inference_latency) > 0 else 0.0

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
        f"Recall: {mean_metrics['recall']:.5f} | Precision: {mean_metrics['precision']:.5f} | "
        f"FPR: {mean_metrics['fpr']:.5f} | TSE: {mean_metrics['tse']:.5f} | "
        f"Inference FPS: {inference_fps:>5.2f} | "
        f"Inference Latency: {inference_latency*1000:>7.2f} ms | Eval wall time: {total_wall_time:.2f}s"
    )

    result_row = {
        "rank_by_tse": "",
        "model": model_name,
        "pretrained": model_pretrained_flag(model_name),
        "modality": modality,
        "run_name": run_value,
        "seed": seed_value,
        "checkpoint_epoch": checkpoint.get("best_epoch", ""),
        "miou": maybe_float_for_csv(mean_metrics["miou"]),
        "biou": maybe_float_for_csv(mean_metrics["biou"]),
        "recall": maybe_float_for_csv(mean_metrics["recall"]),
        "precision": maybe_float_for_csv(mean_metrics["precision"]),
        "fpr": maybe_float_for_csv(mean_metrics["fpr"]),
        "fps": maybe_float_for_csv(mean_metrics["fps"]),
        "latency_ms": maybe_float_for_csv(mean_metrics["latency_ms"]),
        "tse": maybe_float_for_csv(mean_metrics["tse"]),
        "eval_wall_time_s": maybe_float_for_csv(mean_metrics["eval_wall_time_s"]),
        "best_val_tse": maybe_float_for_csv(checkpoint.get("best_val_tse", "")),
        "normalization": normalization,
        "class_weight_max": checkpoint.get("class_weight_max", ""),
    }

    append_csv_row(csv_path(outputs_dir, "results", "overall.csv"), result_row, fieldnames=OVERALL_FIELDS)
    append_csv_rows(csv_path(outputs_dir, "results", "per_class.csv"),
                    per_class_rows_from_metrics(mean_metrics, model_name, modality, seed_value, run_value),
                    fieldnames=PER_CLASS_FIELDS)
    append_csv_row(csv_path(outputs_dir, "results", "thin.csv"),
                   thin_row_from_metrics(mean_metrics, model_name, modality, seed_value, run_value),
                   fieldnames=THIN_FIELDS)
    append_csv_rows(csv_path(outputs_dir, "debug", "error_analysis.csv"), error_rows, fieldnames=ERROR_FIELDS)

    finite_values = [
        mean_metrics["miou"], mean_metrics["biou"], mean_metrics["recall"],
        mean_metrics["precision"], mean_metrics["fpr"], mean_metrics["tse"],
    ]
    all_metrics_finite = int(all(numpy.isfinite(v) for v in finite_values))
    append_csv_row(csv_path(outputs_dir, "debug", "checks.csv"), {
        "stage": "test",
        "model": model_name,
        "pretrained": model_pretrained_flag(model_name),
        "modality": modality,
        "run_name": run_value,
        "seed": seed_value,
        "train_samples": "",
        "validation_samples": "",
        "test_samples": len(test_dataset),
        "num_classes": num_classes,
        "expected_num_classes": len(CANONICAL_CLASS_IDS),
        "checkpoint_exists": int(os.path.exists(checkpoint_path)),
        "checkpoint_epoch": checkpoint.get("best_epoch", ""),
        "test_completed": 1,
        "all_metrics_finite": all_metrics_finite,
        "unknown_labels_found": int(bool(test_dataset.observed_unknown_class_counts)),
        "prediction_shape_ok": prediction_shape_ok,
        "mask_shape_ok": mask_shape_ok,
        "logits_class_count_ok": logits_class_count_ok,
        "visualizations_saved": saved_visualizations,
        "status": "ok" if all_metrics_finite and prediction_shape_ok and mask_shape_ok and logits_class_count_ok else "check",
        "message": "",
    }, fieldnames=CHECK_FIELDS)

    return mean_metrics


def _mean_std(values):
    vals = [v for v in values if v is not None and numpy.isfinite(v)]
    if not vals:
        return "", ""
    return float(numpy.mean(vals)), float(numpy.std(vals))


def _best_row(rows, field, lower=False):
    candidates = [(safe_float(r.get(field)), r) for r in rows]
    candidates = [(v, r) for v, r in candidates if v is not None and numpy.isfinite(v)]
    if not candidates:
        return None
    return min(candidates, key=lambda x: x[0])[1] if lower else max(candidates, key=lambda x: x[0])[1]


def _group_summary(rows, group_field, counterpart_field):
    numeric_fields = ["miou", "biou", "tse", "fps", "latency_ms"]
    summary_rows = []
    groups = {}
    for row in rows:
        groups.setdefault(row[group_field], []).append(row)

    for group_name, group_rows in sorted(groups.items()):
        item = {group_field: group_name, "num_models" if group_field == "modality" else "num_modalities": len(group_rows)}
        if group_field == "model":
            item["pretrained"] = group_rows[0].get("pretrained", "")
        for field in numeric_fields:
            mean, std = _mean_std([safe_float(r.get(field)) for r in group_rows])
            item[f"mean_{field}"] = mean
            item[f"std_{field}"] = std
            best = _best_row(group_rows, field, lower=(field == "latency_ms"))
            item[f"best_{field}"] = best.get(field, "") if best else ""
            item[f"best_{field}_{counterpart_field}"] = best.get(counterpart_field, "") if best else ""
        # Thin summary is joined from thin.csv later when possible.
        item["mean_thin_iou"] = ""
        item["best_thin_iou"] = ""
        item[f"best_thin_iou_{counterpart_field}"] = ""
        summary_rows.append(item)
    return summary_rows


def summarize_results(outputs_dir="outputs"):
    ensure_outputs_structure(outputs_dir)
    overall_path = csv_path(outputs_dir, "results", "overall.csv")
    thin_path = csv_path(outputs_dir, "results", "thin.csv")

    rows = read_csv(overall_path)
    if not rows:
        # Multi-run fallback: collect nested run overall files.
        summary_files = [
            p for p in glob.glob(os.path.join(outputs_dir, "**", "results", "overall.csv"), recursive=True)
            if os.path.abspath(p) != os.path.abspath(overall_path)
        ]
        for path in sorted(summary_files):
            rows.extend(read_csv(path))
    if not rows:
        return

    rows_sorted = sorted(rows, key=lambda r: safe_float(r.get("tse"), -float("inf")), reverse=True)
    for rank, row in enumerate(rows_sorted, start=1):
        row["rank_by_tse"] = rank
    write_csv(overall_path, rows_sorted, fieldnames=OVERALL_FIELDS)

    thin_rows = read_csv(thin_path)
    if thin_rows:
        thin_sorted = sorted(thin_rows, key=lambda r: safe_float(r.get("mean_thin_iou"), -float("inf")), reverse=True)
        for rank, row in enumerate(thin_sorted, start=1):
            row["rank_by_mean_thin_iou"] = rank
        write_csv(thin_path, thin_sorted, fieldnames=THIN_FIELDS)

    modality_rows = _group_summary(rows_sorted, "modality", "model")
    model_rows = _group_summary(rows_sorted, "model", "modality")

    # Attach thin summaries to modality/model summaries.
    if thin_rows:
        for summary_rows, group_field, counterpart_field in [
            (modality_rows, "modality", "model"),
            (model_rows, "model", "modality"),
        ]:
            groups = {}
            for row in thin_rows:
                groups.setdefault(row[group_field], []).append(row)
            for item in summary_rows:
                group = groups.get(item[group_field], [])
                mean, _ = _mean_std([safe_float(r.get("mean_thin_iou")) for r in group])
                best = _best_row(group, "mean_thin_iou")
                item["mean_thin_iou"] = mean
                item["best_thin_iou"] = best.get("mean_thin_iou", "") if best else ""
                item[f"best_thin_iou_{counterpart_field}"] = best.get(counterpart_field, "") if best else ""

    write_csv(csv_path(outputs_dir, "results", "modality.csv"), modality_rows)
    write_csv(csv_path(outputs_dir, "results", "model.csv"), model_rows)
