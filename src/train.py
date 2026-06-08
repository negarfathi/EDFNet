import os
import csv
import time
import numpy
import torch
from torch.utils.data import DataLoader

from src.dataset import DDOSDataset
from src.model import build_model
from src.test import compute_metrics


def get_in_channels(modality):
    if modality == "rgb":
        return 3
    if modality in ["rgbd", "rgbe"]:
        return 4
    if modality == "rgbde":
        return 5
    raise ValueError(f"Unknown modality: {modality}")


def train_model(train_path, validation_path, modality, model_name, device, edge_method, epochs, batch_size, learning_rate,
                results_dir="results"):
    print(f"\n=== Training {model_name.upper()} with Modality: {modality.upper()} ===")
    os.makedirs("checkpoints", exist_ok=True)
    os.makedirs(results_dir, exist_ok=True)

    train_dataset = DDOSDataset(dataset_path=train_path, modality=modality, edge_method=edge_method, augment=True)
    validation_dataset = DDOSDataset(dataset_path=validation_path, modality=modality, edge_method=edge_method, augment=False)

    # Export class-frequency information so the paper can explain rare thin/ultra-thin classes.
    train_dataset.save_class_frequency_csv(os.path.join(results_dir, f"class_frequency_train_{model_name}_{modality}.csv"))

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    validation_loader = DataLoader(validation_dataset, batch_size=batch_size, shuffle=False)

    num_classes = train_dataset.num_classes
    in_channels = get_in_channels(modality)
    model = build_model(model_name=model_name, num_classes=num_classes, in_channels=in_channels).to(device)

    class_weights = torch.tensor(train_dataset.class_weights, dtype=torch.float32).to(device)
    criterion = torch.nn.CrossEntropyLoss(weight=class_weights)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    log_path = os.path.join(results_dir, f"training_log_{model_name}_{modality}.csv")
    with open(log_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "epoch", "loss", "val_miou", "val_biou", "val_recall", "val_fpr", "val_tse",
            "val_fps", "val_latency_ms", "train_time_s", "peak_gpu_train_mb", "peak_gpu_val_mb"
        ])
        writer.writeheader()

        for epoch in range(epochs):
            model.train()
            total_loss = 0.0
            start_time = time.time()
            if torch.cuda.is_available():
                torch.cuda.reset_peak_memory_stats(device)

            for images, masks in train_loader:
                images, masks = images.to(device), masks.to(device)
                optimizer.zero_grad()
                outputs = model(images)
                loss = criterion(outputs, masks)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()

            avg_loss = total_loss / max(1, len(train_loader))
            train_time = time.time() - start_time
            peak_mem_train = torch.cuda.max_memory_allocated(device) / 1024**2 if torch.cuda.is_available() else 0

            model.eval()
            metrics = {"miou": [], "per_class_iou": [], "recall": [], "fpr": [], "biou": []}
            with torch.no_grad():
                val_start = time.time()
                if torch.cuda.is_available():
                    torch.cuda.reset_peak_memory_stats(device)
                for images, masks in validation_loader:
                    images, masks = images.to(device), masks.to(device)
                    outputs = model(images)
                    batch_metrics = compute_metrics(outputs, masks, num_classes)
                    for k in metrics:
                        metrics[k].append(batch_metrics[k])
                latency = (time.time() - val_start) / max(1, len(validation_loader))
                fps = batch_size / latency if latency > 0 else 0
                peak_mem_val = torch.cuda.max_memory_allocated(device) / 1024**2 if torch.cuda.is_available() else 0

            mean_metrics = {}
            for k, v in metrics.items():
                if isinstance(v[0], list):
                    mean_metrics[k] = [numpy.nanmean([batch[i] for batch in v]) for i in range(num_classes)]
                else:
                    mean_metrics[k] = float(numpy.mean(v))

            tse = 0.45 * mean_metrics["biou"] + 0.30 * mean_metrics["recall"] - 0.15 * mean_metrics["fpr"] + 0.10 * mean_metrics["miou"]
            per_class_str = ", ".join([f"{iou:.3f}" for iou in mean_metrics["per_class_iou"]])

            print(
                f"Epoch [{epoch+1:>2}/{epochs}] | Loss: {avg_loss:.5f} | "
                f"Validation => mIoU: {mean_metrics['miou']:.5f} | bIoU: {mean_metrics['biou']:.5f} | "
                f"Recall: {mean_metrics['recall']:.5f} | FPR: {mean_metrics['fpr']:.5f} | TSE: {tse:.5f} | "
                f"per-class IoU: [{per_class_str}] | FPS: {fps:>5.2f} | Latency: {latency*1000:>7.2f} ms"
            )

            writer.writerow({
                "epoch": epoch + 1,
                "loss": avg_loss,
                "val_miou": mean_metrics["miou"],
                "val_biou": mean_metrics["biou"],
                "val_recall": mean_metrics["recall"],
                "val_fpr": mean_metrics["fpr"],
                "val_tse": tse,
                "val_fps": fps,
                "val_latency_ms": latency * 1000,
                "train_time_s": train_time,
                "peak_gpu_train_mb": peak_mem_train,
                "peak_gpu_val_mb": peak_mem_val,
            })
            f.flush()

    checkpoint_path = os.path.join("checkpoints", f"{model_name}_{modality}.pth")
    torch.save({
        "state_dict": model.state_dict(),
        "num_classes": num_classes,
        "in_channels": in_channels,
        "model_name": model_name,
        "modality": modality,
        "class_ids": train_dataset.class_ids,
    }, checkpoint_path)

    print(f"[INFO] Model saved as: {checkpoint_path}")
    print(f"[INFO] Training log saved as: {log_path}\n")
    return model
