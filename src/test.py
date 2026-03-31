import os
import cv2
import time
import numpy
import torch
from torch.utils.data import DataLoader

from src.dataset import DDOSDataset
from src.model import build_model
from src.visualize import visualize_predictions

def compute_boundary(mask, dilation_ratio=0.02):
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
            iou = intersection / union if union > 0 else 0.0
            recall = tp / (tp + fn + 1e-8) if (tp + fn) > 0 else 0.0
            fpr = fp / (fp + (true_c == 0).sum() + 1e-8)
            per_class_ious[c].append(iou)
            all_ious.append(iou)
            all_recalls.append(recall)
            all_fprs.append(fpr)
        biou_sample, _ = compute_biou(pred_mask, true_mask, num_classes)
        all_bious.append(biou_sample)
    mean_per_class_ious = [numpy.mean(v) if len(v) > 0 else 0.0 for v in per_class_ious]
    return {
        "miou": numpy.mean(all_ious),
        "per_class_iou": mean_per_class_ious,
        "biou": numpy.mean(all_bious),
        "recall": numpy.mean(all_recalls),
        "fpr": numpy.mean(all_fprs)
    }



def test_model(test_path, modality, device, edge_method, checkpoint_path, batch_size, visualize=False):
    print(f"\n=== Testing {modality.upper()} using checkpoint: {checkpoint_path} ===")

    test_dataset = DDOSDataset(dataset_path=test_path, modality=modality, edge_method=edge_method)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

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

    metrics_all = {"miou": [], "per_class_iou": [], "biou": [], "recall": [], "fpr": []}
    start_time = time.time()
    with torch.no_grad():
        for batch_idx, (images, masks) in enumerate(test_loader):
            images, masks = images.to(device), masks.to(device)
            outputs = model(images)
            batch_metrics = compute_metrics(outputs, masks, num_classes)
            for k in metrics_all:
                metrics_all[k].append(batch_metrics[k])
            if visualize:
                color_map = {0: (0, 0, 0), 1: (255, 0, 0)}
                visualize_predictions(images, masks, outputs, color_map, visualize_path=visualize_path)
    total_time = time.time() - start_time
    latency = total_time / len(test_loader)
    fps = batch_size / latency if latency > 0 else 0
    mean_metrics = {}
    for k, v in metrics_all.items():
        if isinstance(v[0], list):
            mean_metrics[k] = [numpy.mean([b[i] for b in v if i < len(b)]) for i in range(num_classes)]
        else:
            mean_metrics[k] = numpy.mean(v)
    per_class_str = ", ".join([f"{iou:.3f}" for iou in mean_metrics["per_class_iou"]])
    print(
        f"Test Results => "
        f"mean IoU: {mean_metrics['miou']:.5f} | "
        f"per-class IoU: [{per_class_str}] | "
        f"boundary IoU: {mean_metrics['biou']:.5f} | "
        f"Recall: {mean_metrics['recall']:.5f} | "
        f"FPR: {mean_metrics['fpr']:.5f} | "
        f"FPS: {fps:>5.2f} | "
        f"Latency: {latency*1000:>7.2f} ms"
    )
    return mean_metrics