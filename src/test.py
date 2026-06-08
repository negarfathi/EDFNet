import os
import csv
import time
import glob
import numpy
import torch
from torch.utils.data import DataLoader

from src.dataset import DDOSDataset, CLASS_NAMES
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


def compute_biou(pred, gt, num_classes):
    biou_scores = []
    for c in range(num_classes):
        pred_boundary = compute_boundary(pred == c)
        gt_boundary = compute_boundary(gt == c)
        intersection = numpy.logical_and(pred_boundary, gt_boundary).sum()
        union = numpy.logical_or(pred_boundary, gt_boundary).sum()
        biou = intersection / union if union > 0 else 0.0
        biou_scores.append(biou)
    return numpy.mean(biou_scores), biou_scores


def compute_metrics(predictions, labels, num_classes):
    preds = torch.argmax(predictions, dim=1).cpu().numpy()
    labels = labels.cpu().numpy()
    all_ious, all_recalls, all_fprs, all_bious = [], [], [], []
    per_class_ious = [[] for _ in range(num_classes)]
    per_class_recalls = [[] for _ in range(num_classes)]
    per_class_fprs = [[] for _ in range(num_classes)]

    for i in range(preds.shape[0]):
        pred_mask = preds[i]
        true_mask = labels[i]
        for c in range(num_classes):
            pred_c = pred_mask == c
            true_c = true_mask == c
            intersection = (pred_c & true_c).sum()
            union = (pred_c | true_c).sum()
            tp = intersection
            fn = true_c.sum() - tp
            fp = pred_c.sum() - tp
            tn = numpy.logical_not(pred_c | true_c).sum()
            iou = intersection / union if union > 0 else 0.0
            recall = tp / (tp + fn + 1e-8) if (tp + fn) > 0 else 0.0
            fpr = fp / (fp + tn + 1e-8) if (fp + tn) > 0 else 0.0
            per_class_ious[c].append(iou)
            per_class_recalls[c].append(recall)
            per_class_fprs[c].append(fpr)
            all_ious.append(iou)
            all_recalls.append(recall)
            all_fprs.append(fpr)
        biou_sample, _ = compute_biou(pred_mask, true_mask, num_classes)
        all_bious.append(biou_sample)

    return {
        "miou": float(numpy.mean(all_ious)),
        "per_class_iou": [float(numpy.mean(v)) if len(v) > 0 else 0.0 for v in per_class_ious],
        "per_class_recall": [float(numpy.mean(v)) if len(v) > 0 else 0.0 for v in per_class_recalls],
        "per_class_fpr": [float(numpy.mean(v)) if len(v) > 0 else 0.0 for v in per_class_fprs],
        "biou": float(numpy.mean(all_bious)),
        "recall": float(numpy.mean(all_recalls)),
        "fpr": float(numpy.mean(all_fprs)),
    }


def compute_sample_error_rows(predictions, labels, metadata, num_classes):
    preds = torch.argmax(predictions, dim=1).cpu().numpy()
    labels_np = labels.cpu().numpy()
    rows = []
    for i in range(preds.shape[0]):
        pred_mask = preds[i]
        true_mask = labels_np[i]
        sample_key = metadata[i].get("sample_key", f"sample_{i}") if metadata else f"sample_{i}"
        for c in range(num_classes):
            pred_c = pred_mask == c
            true_c = true_mask == c
            intersection = (pred_c & true_c).sum()
            union = (pred_c | true_c).sum()
            tp = intersection
            fn = true_c.sum() - tp
            fp = pred_c.sum() - tp
            iou = intersection / union if union > 0 else 0.0
            recall = tp / (tp + fn + 1e-8) if (tp + fn) > 0 else 0.0
            class_name = CLASS_NAMES[c] if c < len(CLASS_NAMES) else f"class_{c}"
            rows.append({
                "sample_key": sample_key,
                "class_index": c,
                "class_name": class_name,
                "gt_pixels": int(true_c.sum()),
                "pred_pixels": int(pred_c.sum()),
                "tp": int(tp),
                "fp": int(fp),
                "fn": int(fn),
                "iou": float(iou),
                "recall": float(recall),
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


def test_model(test_path, modality, device, edge_method, checkpoint_path, batch_size, visualize=False,
               results_dir="results", visualize_max=20):
    print(f"\n=== Testing {modality.upper()} using checkpoint: {checkpoint_path} ===")
    os.makedirs(results_dir, exist_ok=True)

    test_dataset = DDOSDataset(dataset_path=test_path, modality=modality, edge_method=edge_method,
                               augment=False, return_metadata=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, collate_fn=metadata_collate)
    test_dataset.save_class_frequency_csv(os.path.join(results_dir, f"class_frequency_test_{modality}.csv"))

    checkpoint = torch.load(checkpoint_path, map_location=device)
    num_classes = checkpoint["num_classes"]
    in_channels = checkpoint["in_channels"]
    model_name = checkpoint.get("model_name")

    model = build_model(model_name, num_classes=num_classes, in_channels=in_channels)
    model.load_state_dict(checkpoint["state_dict"])
    model.to(device).eval()

    if visualize:
        visualize_path = os.path.join("visualizations", f"{model_name}_{modality}")
        os.makedirs(visualize_path, exist_ok=True)
        print(f"[INFO] Saving visual results to: {visualize_path}")
    else:
        visualize_path = None

    metrics_all = {"miou": [], "per_class_iou": [], "per_class_recall": [], "per_class_fpr": [], "biou": [], "recall": [], "fpr": []}
    error_rows = []
    start_time = time.time()

    with torch.no_grad():
        for batch_idx, (images, masks, metadata) in enumerate(test_loader):
            images, masks = images.to(device), masks.to(device)
            outputs = model(images)
            batch_metrics = compute_metrics(outputs, masks, num_classes)
            for k in metrics_all:
                metrics_all[k].append(batch_metrics[k])
            error_rows.extend(compute_sample_error_rows(outputs, masks, metadata, num_classes))

            if visualize:
                label_colors = make_color_map(num_classes)
                max_images = visualize_max if visualize_max is not None else 20
                visualize_predictions(images, masks, outputs, label_colors, visualize_path=visualize_path,
                                      metadata=metadata, max_images=max_images, make_zoom=True,
                                      target_classes=[6, 7])

    total_time = time.time() - start_time
    latency = total_time / max(1, len(test_loader))
    fps = batch_size / latency if latency > 0 else 0

    mean_metrics = {}
    for k, v in metrics_all.items():
        if isinstance(v[0], list):
            mean_metrics[k] = [float(numpy.mean([b[i] for b in v if i < len(b)])) for i in range(num_classes)]
        else:
            mean_metrics[k] = float(numpy.mean(v))

    tse = 0.45 * mean_metrics["biou"] + 0.30 * mean_metrics["recall"] - 0.15 * mean_metrics["fpr"] + 0.10 * mean_metrics["miou"]
    mean_metrics["tse"] = tse
    mean_metrics["fps"] = fps
    mean_metrics["latency_ms"] = latency * 1000

    per_class_str = ", ".join([f"{iou:.3f}" for iou in mean_metrics["per_class_iou"]])
    print(
        f"Test Results => mean IoU: {mean_metrics['miou']:.5f} | "
        f"per-class IoU: [{per_class_str}] | boundary IoU: {mean_metrics['biou']:.5f} | "
        f"Recall: {mean_metrics['recall']:.5f} | FPR: {mean_metrics['fpr']:.5f} | "
        f"TSE: {mean_metrics['tse']:.5f} | FPS: {fps:>5.2f} | Latency: {latency*1000:>7.2f} ms"
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
    }
    for c, iou in enumerate(mean_metrics["per_class_iou"]):
        class_name = CLASS_NAMES[c].replace(" ", "_") if c < len(CLASS_NAMES) else f"class_{c}"
        result_row[f"iou_{class_name}"] = iou
    for c, rec in enumerate(mean_metrics["per_class_recall"]):
        class_name = CLASS_NAMES[c].replace(" ", "_") if c < len(CLASS_NAMES) else f"class_{c}"
        result_row[f"recall_{class_name}"] = rec

    write_csv(os.path.join(results_dir, f"test_summary_{model_name}_{modality}.csv"), [result_row])
    write_csv(os.path.join(results_dir, f"error_analysis_{model_name}_{modality}.csv"), error_rows)

    return mean_metrics


def metadata_collate(batch):
    images = torch.stack([item[0] for item in batch], dim=0)
    masks = torch.stack([item[1] for item in batch], dim=0)
    metadata = [item[2] for item in batch]
    return images, masks, metadata


def summarize_results(results_dir="results"):
    summary_files = sorted(glob.glob(os.path.join(results_dir, "test_summary_*.csv")))
    rows = []
    for path in summary_files:
        with open(path, "r", newline="") as f:
            reader = csv.DictReader(f)
            rows.extend(list(reader))
    if not rows:
        return

    # Convert numeric fields when possible.
    numeric_fields = ["miou", "biou", "recall", "fpr", "tse", "fps", "latency_ms"]
    rows_sorted = sorted(rows, key=lambda r: float(r.get("tse", 0.0)), reverse=True)
    combined_path = os.path.join(results_dir, "combined_test_results_ranked.csv")
    with open(combined_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows_sorted[0].keys()))
        writer.writeheader()
        writer.writerows(rows_sorted)
    print(f"[INFO] Saved ranked combined results: {combined_path}")

    # Modality-level cue summary: helps explain which cue helps which metric.
    modality_groups = {}
    for r in rows:
        modality_groups.setdefault(r["modality"], []).append(r)
    cue_rows = []
    for modality, group in modality_groups.items():
        item = {"modality": modality, "n_models": len(group)}
        for field in numeric_fields:
            vals = [float(g[field]) for g in group if field in g and g[field] != ""]
            item[f"mean_{field}"] = float(numpy.mean(vals)) if vals else ""
            item[f"best_{field}"] = float(numpy.max(vals)) if vals else ""
        cue_rows.append(item)
    cue_path = os.path.join(results_dir, "modality_cue_summary.csv")
    write_csv(cue_path, cue_rows)
