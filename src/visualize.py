import os
import cv2
import numpy
import torch
import matplotlib.pyplot

_global_counter = 0

def visualize_predictions(images, labels, predictions, label_colors, visualize_path=None):
    global _global_counter
    preds = torch.argmax(predictions, dim=1).cpu().numpy()
    imgs = images.permute(0, 2, 3, 1).cpu().numpy()
    if visualize_path:
        os.makedirs(visualize_path, exist_ok=True)
    for i in range(len(imgs)):
        img = (imgs[i][:, :, :3] * 255).astype(numpy.uint8)
        gt_mask = labels[i].cpu().numpy()
        pred_mask = preds[i]
        h, w = img.shape[:2]
        overlay_pred = numpy.zeros((h, w, 3), dtype=numpy.uint8)
        overlay_gt = numpy.zeros((h, w, 3), dtype=numpy.uint8)
        for c, color in label_colors.items():
            overlay_pred[pred_mask == c] = color
            overlay_gt[gt_mask == c] = color
        blended_pred = cv2.addWeighted(img, 0.6, overlay_pred, 0.4, 0)
        blended_gt = cv2.addWeighted(img, 0.6, overlay_gt, 0.4, 0)
        fig, axes = matplotlib.pyplot.subplots(1, 2, figsize=(12, 6))
        axes[0].imshow(blended_gt)
        axes[0].set_title("Ground Truth")
        axes[0].axis("off")
        axes[1].imshow(blended_pred)
        axes[1].set_title("Prediction")
        axes[1].axis("off")
        if visualize_path:
            out_file = os.path.join(visualize_path, f"vis_{_global_counter}.png")
            matplotlib.pyplot.savefig(out_file, bbox_inches="tight")
            print(f"[INFO] Saved {out_file}")
        matplotlib.pyplot.close()
        _global_counter += 1