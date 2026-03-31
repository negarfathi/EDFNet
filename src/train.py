import os
import time
import numpy
import torch
from torch.utils.data import DataLoader

from src.dataset import DDOSDataset
from src.model import build_model
from src.test import compute_metrics

def train_model(train_path, validation_path, modality, model_name, device, edge_method, epochs, batch_size, learning_rate):
    print(f"\n=== Training {model_name.upper()} with Modality: {modality.upper()} ===")

    train_dataset = DDOSDataset(dataset_path=train_path, modality=modality, edge_method=edge_method)
    validation_dataset = DDOSDataset(dataset_path=validation_path, modality=modality, edge_method=edge_method)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    validation_loader = DataLoader(validation_dataset, batch_size=batch_size, shuffle=False)

    num_classes = train_dataset.num_classes

    if modality == "rgb":
        in_channels = 3
    elif modality in ["rgbd", "rgbe"]:
        in_channels = 4
    elif modality == "rgbde":
        in_channels = 5
    else:
        raise ValueError(f"Unknown modality: {modality}")

    model = build_model(model_name=model_name, num_classes=num_classes, in_channels=in_channels).to(device)

    class_weights = torch.tensor(train_dataset.class_weights, dtype=torch.float32).to(device)
    criterion = torch.nn.CrossEntropyLoss(weight=class_weights)

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

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
        avg_loss = total_loss / len(train_loader)
        train_time = time.time() - start_time
        peak_mem_train = torch.cuda.max_memory_allocated(device) / 1024**2 if torch.cuda.is_available() else 0
        print(
            f"Epoch [{epoch+1:>2}/{epochs}] | "
            f"Loss: {avg_loss:.5f} | Train Time: {train_time:>6.2f}s | "
            f"Peak GPU: {peak_mem_train:>7.2f} MB"
        )

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
            latency = (time.time() - val_start) / len(validation_loader)
            fps = batch_size / latency if latency > 0 else 0
            peak_mem_val = torch.cuda.max_memory_allocated(device) / 1024**2 if torch.cuda.is_available() else 0
        mean_metrics = {}
        for k, v in metrics.items():
            if isinstance(v[0], list):  # handle per-class IoU
                mean_metrics[k] = [numpy.nanmean([batch[i] for batch in v]) for i in range(num_classes)]
            else:
                mean_metrics[k] = numpy.mean(v)
        per_class_str = ", ".join([f"{iou:.3f}" for iou in mean_metrics["per_class_iou"]])
        print(
            f"Validation Results => "
            f"mean IoU: {mean_metrics['miou']:.5f} | "
            f"per-class IoU: [{per_class_str}] | "
            f"boundary IoU: {mean_metrics['biou']:.5f} | "
            f"Recall: {mean_metrics['recall']:.5f} | "
            f"FPR: {mean_metrics['fpr']:.5f} | "
            f"FPS: {fps:>5.2f} | "
            f"Latency: {latency*1000:>7.2f} ms | "
            f"Peak GPU: {peak_mem_val:>7.2f} MB"
        )

    checkpoint_path = os.path.join("checkpoints", f"{model_name}_{modality}.pth")

    torch.save({
        "state_dict": model.state_dict(),
        "num_classes": num_classes,
        "in_channels": in_channels,
        "model_name": model_name,
    }, checkpoint_path)

    print(f"[INFO] Model saved as: {checkpoint_path}\n")

    return model