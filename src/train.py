import os
import csv
import time
import numpy
import torch
from torch.utils.data import DataLoader

from src.dataset import DDOSDataset, CANONICAL_CLASS_IDS, CLASS_NAMES, normalization_for_model
from src.model import build_model
from src.test import SegmentationMetricAccumulator


def get_in_channels(modality):
    if modality == "rgb":
        return 3
    if modality in ["rgbd", "rgbe"]:
        return 4
    if modality == "rgbde":
        return 5
    raise ValueError(f"Unknown modality: {modality}")


def get_inference_timer(device):
    def sync():
        if str(device).startswith("cuda") and torch.cuda.is_available():
            torch.cuda.synchronize()
    return sync


def save_checkpoint(path, model, model_name, modality, num_classes, in_channels, train_dataset,
                    epoch, best_val_tse, normalization, ignore_index,
                    class_weight_max=None, seed=None, run_name=None):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    torch.save({
        "state_dict": model.state_dict(),
        "num_classes": num_classes,
        "in_channels": in_channels,
        "model_name": model_name,
        "modality": modality,
        "class_ids": list(CANONICAL_CLASS_IDS),
        "class_names": list(CLASS_NAMES),
        "class_to_index": dict(train_dataset.class_to_index),
        "best_epoch": epoch,
        "best_val_tse": best_val_tse,
        "normalization": normalization,
        "ignore_index": ignore_index,
        "class_weights": list(train_dataset.class_weights),
        "raw_class_weights": list(train_dataset.raw_class_weights),
        "class_weight_max": class_weight_max,
        "seed": seed,
        "run_name": run_name,
    }, path)


def train_model(train_path, validation_path, modality, model_name, device, edge_method, epochs, batch_size, learning_rate,
                results_dir="results", checkpoint_dir="checkpoints", class_weight_max=10.0,
                seed=42, run_name="run_01"):
    def seed_worker(worker_id):
        worker_seed = int(seed) + worker_id
        numpy.random.seed(worker_seed)
        torch.manual_seed(worker_seed)

    loader_generator = torch.Generator()
    loader_generator.manual_seed(int(seed))
    print(f"\n=== Training {model_name.upper()} with Modality: {modality.upper()} ===")
    os.makedirs(checkpoint_dir, exist_ok=True)
    os.makedirs(results_dir, exist_ok=True)

    normalization = normalization_for_model(model_name)
    train_dataset = DDOSDataset(dataset_path=train_path, modality=modality, edge_method=edge_method, augment=True,
                                normalization=normalization, class_weight_max=class_weight_max)
    validation_dataset = DDOSDataset(dataset_path=validation_path, modality=modality, edge_method=edge_method,
                                     augment=False, normalization=normalization,
                                     class_weight_max=class_weight_max)

    train_dataset.save_class_frequency_csv(os.path.join(results_dir, f"class_frequency_train_{model_name}_{modality}.csv"))
    validation_dataset.save_class_frequency_csv(os.path.join(results_dir, f"class_frequency_validation_{model_name}_{modality}.csv"))

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        worker_init_fn=seed_worker,
        generator=loader_generator,
    )
    validation_loader = DataLoader(
        validation_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        worker_init_fn=seed_worker,
        generator=loader_generator,
    )

    num_classes = train_dataset.num_classes
    in_channels = get_in_channels(modality)
    model = build_model(model_name=model_name, num_classes=num_classes, in_channels=in_channels).to(device)

    class_weights = torch.tensor(train_dataset.class_weights, dtype=torch.float32).to(device)
    criterion = torch.nn.CrossEntropyLoss(weight=class_weights, ignore_index=train_dataset.ignore_index)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    checkpoint_path = os.path.join(checkpoint_dir, f"{model_name}_{modality}.pth")
    last_checkpoint_path = os.path.join(checkpoint_dir, f"{model_name}_{modality}_last.pth")
    best_val_tse = -float("inf")
    best_epoch = 0

    log_path = os.path.join(results_dir, f"training_log_{model_name}_{modality}.csv")
    with open(log_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "epoch", "loss", "val_miou", "val_biou", "val_recall", "val_fpr", "val_tse",
            "val_fps", "val_latency_ms", "val_wall_time_s", "train_time_s",
            "peak_gpu_train_mb", "peak_gpu_val_mb", "is_best", "seed", "run_name", "normalization", "class_weight_max"
        ])
        writer.writeheader()

        for epoch in range(epochs):
            model.train()
            total_loss = 0.0
            start_time = time.perf_counter()
            if torch.cuda.is_available() and str(device).startswith("cuda"):
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
            train_time = time.perf_counter() - start_time
            peak_mem_train = (
                torch.cuda.max_memory_allocated(device) / 1024**2
                if torch.cuda.is_available() and str(device).startswith("cuda") else 0
            )

            model.eval()
            accumulator = SegmentationMetricAccumulator(
                num_classes=num_classes,
                ignore_index=validation_dataset.ignore_index,
            )
            inference_time = 0.0
            inference_batches = 0
            val_wall_start = time.perf_counter()
            sync = get_inference_timer(device)
            if torch.cuda.is_available() and str(device).startswith("cuda"):
                torch.cuda.reset_peak_memory_stats(device)

            with torch.no_grad():
                for images, masks in validation_loader:
                    images, masks = images.to(device), masks.to(device)
                    sync()
                    forward_start = time.perf_counter()
                    outputs = model(images)
                    sync()
                    inference_time += time.perf_counter() - forward_start
                    inference_batches += 1
                    accumulator.update_from_logits(outputs, masks)

            val_wall_time = time.perf_counter() - val_wall_start
            latency = inference_time / max(1, inference_batches)
            fps = batch_size / latency if latency > 0 else 0
            peak_mem_val = (
                torch.cuda.max_memory_allocated(device) / 1024**2
                if torch.cuda.is_available() and str(device).startswith("cuda") else 0
            )

            mean_metrics = accumulator.compute()
            tse = mean_metrics["tse"]
            is_best = bool(numpy.isfinite(tse) and tse > best_val_tse)
            if is_best:
                best_val_tse = tse
                best_epoch = epoch + 1
                save_checkpoint(
                    checkpoint_path, model, model_name, modality, num_classes, in_channels,
                    train_dataset, epoch + 1, best_val_tse, normalization, train_dataset.ignore_index,
                    class_weight_max=class_weight_max, seed=seed, run_name=run_name
                )

            per_class_str = ", ".join([
                "nan" if numpy.isnan(iou) else f"{iou:.3f}" for iou in mean_metrics["per_class_iou"]
            ])

            print(
                f"Epoch [{epoch+1:>2}/{epochs}] | Loss: {avg_loss:.5f} | "
                f"Validation => mIoU: {mean_metrics['miou']:.5f} | bIoU: {mean_metrics['biou']:.5f} | "
                f"Recall: {mean_metrics['recall']:.5f} | FPR: {mean_metrics['fpr']:.5f} | TSE: {tse:.5f} | "
                f"per-class IoU: [{per_class_str}] | Inference FPS: {fps:>5.2f} | "
                f"Inference Latency: {latency*1000:>7.2f} ms | best={is_best}"
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
                "val_wall_time_s": val_wall_time,
                "train_time_s": train_time,
                "peak_gpu_train_mb": peak_mem_train,
                "peak_gpu_val_mb": peak_mem_val,
                "is_best": int(is_best),
                "seed": seed,
                "run_name": run_name,
                "normalization": normalization,
                "class_weight_max": class_weight_max,
            })
            f.flush()

    # Always save the last checkpoint too, but keep the standard path for the best checkpoint.
    save_checkpoint(
        last_checkpoint_path, model, model_name, modality, num_classes, in_channels,
        train_dataset, epochs, best_val_tse, normalization, train_dataset.ignore_index,
        class_weight_max=class_weight_max, seed=seed, run_name=run_name
    )
    if best_epoch == 0:
        # Fallback if validation TSE was never finite.
        save_checkpoint(
            checkpoint_path, model, model_name, modality, num_classes, in_channels,
            train_dataset, epochs, best_val_tse, normalization, train_dataset.ignore_index,
            class_weight_max=class_weight_max, seed=seed, run_name=run_name
        )
        best_epoch = epochs

    print(f"[INFO] Best model saved as: {checkpoint_path} (epoch={best_epoch}, val_tse={best_val_tse:.6f})")
    print(f"[INFO] Last model saved as: {last_checkpoint_path}")
    print(f"[INFO] Training log saved as: {log_path}\n")
    return model
