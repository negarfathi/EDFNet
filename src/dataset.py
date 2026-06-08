import os
import csv
import cv2
import numpy
import torch
import albumentations
from glob import glob
from collections import Counter
from torch.utils.data import Dataset


# Update this list if the DDOS label order in your local copy differs.
CLASS_NAMES = [
    "Animals",
    "Vehicles",
    "Buildings",
    "Trees",
    "Large Mesh",
    "Small Mesh",
    "Thin Structures",
    "Ultra-thin",
    "Other",
    "Background",
]


class DDOSDataset(Dataset):
    def __init__(self, dataset_path, modality, edge_method, size=256, augment=True, return_metadata=False):
        self.dataset_path = dataset_path
        self.modality = modality
        self.edge_method = edge_method
        self.size = size
        self.augment = augment
        self.return_metadata = return_metadata

        rgb_images = sorted(glob(os.path.join(dataset_path, "**", "image", "*.png"), recursive=True))
        depth_images = sorted(glob(os.path.join(dataset_path, "**", "depth", "*.png"), recursive=True))
        mask_images = sorted(glob(os.path.join(dataset_path, "**", "segmentation", "*.png"), recursive=True))

        def make_key(path):
            rel = os.path.relpath(path, dataset_path)
            parts = rel.split(os.sep)
            if len(parts) < 3:
                stem = os.path.splitext(parts[-1])[0]
                return stem
            env = parts[0]
            flight = parts[1]
            frame = os.path.splitext(parts[-1])[0]
            return f"{env}/{flight}/{frame}"

        rgb_dict = {make_key(p): p for p in rgb_images}
        depth_dict = {make_key(p): p for p in depth_images}
        mask_dict = {make_key(p): p for p in mask_images}

        self.sample_keys = sorted(set(rgb_dict) & set(depth_dict) & set(mask_dict))
        if len(self.sample_keys) == 0:
            raise RuntimeError("NO MATCHING RGB/DEPTH/MASK TRIPLETS FOUND! Check dataset path and structure.")

        self.rgb_images = [rgb_dict[k] for k in self.sample_keys]
        self.depth_images = [depth_dict[k] for k in self.sample_keys]
        self.mask_images = [mask_dict[k] for k in self.sample_keys]

        print(f"[DDOSDataset] Loaded samples: {len(self.rgb_images)} from {dataset_path}")

        self.class_pixel_counts = Counter()
        all_class_ids = []
        for mask_path in self.mask_images:
            mask = cv2.imread(mask_path, cv2.IMREAD_UNCHANGED)
            if mask is None:
                continue
            unique_classes, pixels_per_class = numpy.unique(mask, return_counts=True)
            all_class_ids.extend(unique_classes)
            for class_id, pixel_count in zip(unique_classes, pixels_per_class):
                self.class_pixel_counts[int(class_id)] += int(pixel_count)

        self.class_ids = sorted(set(map(int, all_class_ids)))
        self.class_to_index = {class_id: idx for idx, class_id in enumerate(self.class_ids)}
        self.index_to_class = {idx: class_id for class_id, idx in self.class_to_index.items()}
        self.num_classes = len(self.class_ids)

        total_pixels = sum(self.class_pixel_counts[cid] for cid in self.class_ids)
        self.total_pixels = total_pixels
        self.class_weights = [
            (total_pixels / (self.num_classes * self.class_pixel_counts[cid]))
            if self.class_pixel_counts[cid] > 0 else 0
            for cid in self.class_ids
        ]

        # Split augmentations into two stages.
        # 1) Spatial transforms are applied jointly to RGB, depth, edge, and mask.
        # 2) Photometric transforms are applied only to RGB.
        # This preserves multimodal alignment for RGB-D/RGB-E/RGB-D-E fusion.
        self.spatial_transform = albumentations.Compose(
            [
                albumentations.HorizontalFlip(p=0.5),
                albumentations.ShiftScaleRotate(
                    shift_limit=0.05,
                    scale_limit=0.1,
                    rotate_limit=10,
                    border_mode=cv2.BORDER_REFLECT,
                    p=0.5,
                ),
                albumentations.RandomResizedCrop(
                    size=(self.size, self.size),
                    scale=(0.8, 1.0),
                    ratio=(0.9, 1.1),
                    p=0.5,
                ),
            ],
            additional_targets={
                "depth": "image",  # bilinear-like interpolation for continuous depth
                "edge": "image",   # keep edge map geometrically aligned
            },
        ) if augment else None

        self.color_transform = albumentations.Compose([
            albumentations.RandomBrightnessContrast(p=0.5),
            albumentations.CLAHE(clip_limit=2.0, tile_grid_size=(8, 8), p=0.3),
        ]) if augment else None

    def class_frequency_rows(self):
        rows = []
        for idx, class_id in enumerate(self.class_ids):
            count = self.class_pixel_counts[class_id]
            percent = 100.0 * count / max(1, self.total_pixels)
            class_name = CLASS_NAMES[idx] if idx < len(CLASS_NAMES) else f"class_{class_id}"
            rows.append({
                "mapped_index": idx,
                "original_class_id": class_id,
                "class_name": class_name,
                "pixel_count": count,
                "pixel_percent": percent,
                "class_weight": self.class_weights[idx],
            })
        return rows

    def save_class_frequency_csv(self, output_csv):
        os.makedirs(os.path.dirname(output_csv) or ".", exist_ok=True)
        rows = self.class_frequency_rows()
        with open(output_csv, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)
        print(f"[INFO] Saved class-frequency table: {output_csv}")

    def print_class_frequency(self):
        print("\n[Class frequency]")
        for row in self.class_frequency_rows():
            print(
                f"{row['mapped_index']:02d} | {row['class_name']:<16} | "
                f"pixels={row['pixel_count']:<12} | {row['pixel_percent']:.6f}% | "
                f"weight={row['class_weight']:.4f}"
            )

    def extract_edges(self, rgb_image):
        gray_image = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2GRAY)
        if self.edge_method == "canny":
            edge_map = cv2.Canny(gray_image, 100, 200)
        elif self.edge_method == "sobel":
            gx = cv2.Sobel(gray_image, cv2.CV_64F, 1, 0, ksize=3)
            gy = cv2.Sobel(gray_image, cv2.CV_64F, 0, 1, ksize=3)
            edge_map = cv2.magnitude(gx, gy)
            edge_map = (edge_map / (edge_map.max() + 1e-8) * 255).astype(numpy.uint8)
        else:
            raise ValueError(f"Unknown edge detection method: {self.edge_method}")
        return edge_map.astype(numpy.float32) / 255.0

    def __len__(self):
        return len(self.rgb_images)

    def __getitem__(self, index):
        rgb_path = self.rgb_images[index]
        depth_path = self.depth_images[index]
        mask_path = self.mask_images[index]
        sample_key = self.sample_keys[index]

        rgb_image_bgr = cv2.imread(rgb_path)
        depth_image = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED)
        mask_image = cv2.imread(mask_path, cv2.IMREAD_UNCHANGED)

        if rgb_image_bgr is None or depth_image is None or mask_image is None:
            raise RuntimeError(f"FAILED TO LOAD DATA:\n{rgb_path}\n{depth_path}\n{mask_path}")

        rgb_image = rgb_image_bgr[:, :, ::-1]

        rgb_image = cv2.resize(rgb_image, (self.size, self.size))
        depth_image = cv2.resize(depth_image, (self.size, self.size))
        mask_image = cv2.resize(mask_image, (self.size, self.size), interpolation=cv2.INTER_NEAREST)

        # CLAHE enhancement used consistently before edge extraction.
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        for c in range(3):
            rgb_image[:, :, c] = clahe.apply(rgb_image[:, :, c])

        rgb_image = rgb_image.astype(numpy.float32) / 255.0

        depth_image = depth_image.astype(numpy.float32)
        if depth_image.max() > 0:
            depth_image = depth_image / depth_image.max()

        edge_image = self.extract_edges((rgb_image * 255).astype(numpy.uint8))

        if self.augment and self.spatial_transform:
            # Apply the same geometric transform to every aligned modality.
            # RGB, depth, edge, and mask stay pixel-aligned after augmentation.
            augmented = self.spatial_transform(
                image=(rgb_image * 255).astype(numpy.uint8),
                depth=depth_image.astype(numpy.float32),
                edge=edge_image.astype(numpy.float32),
                mask=mask_image,
            )
            rgb_image = augmented["image"].astype(numpy.float32) / 255.0
            depth_image = augmented["depth"].astype(numpy.float32)
            edge_image = augmented["edge"].astype(numpy.float32)
            mask_image = augmented["mask"]

            # Apply color-only augmentation to RGB after geometry. This does not affect depth/mask.
            if self.color_transform:
                color_augmented = self.color_transform(image=(rgb_image * 255).astype(numpy.uint8))
                rgb_image = color_augmented["image"].astype(numpy.float32) / 255.0
                # Recompute edge from the final RGB so the edge channel matches the actual RGB input.
                edge_image = self.extract_edges((rgb_image * 255).astype(numpy.uint8))

            # Guard numeric ranges after interpolation/augmentation.
            depth_image = numpy.clip(depth_image, 0.0, 1.0)
            edge_image = numpy.clip(edge_image, 0.0, 1.0)

        if self.modality == "rgb":
            input_image = rgb_image
        elif self.modality == "rgbd":
            input_image = numpy.dstack([rgb_image, depth_image])
        elif self.modality == "rgbe":
            input_image = numpy.dstack([rgb_image, edge_image])
        elif self.modality == "rgbde":
            input_image = numpy.dstack([rgb_image, depth_image, edge_image])
        else:
            raise ValueError(f"Unknown modality: {self.modality}")

        mask_mapper = numpy.vectorize(lambda v: self.class_to_index.get(int(v), 0))
        mask_remapped = mask_mapper(mask_image).astype(numpy.int64)

        input_tensor = torch.tensor(input_image, dtype=torch.float32).permute(2, 0, 1)
        target_tensor = torch.tensor(mask_remapped, dtype=torch.long)

        if self.return_metadata:
            metadata = {
                "sample_key": sample_key,
                "rgb_path": rgb_path,
                "depth_path": depth_path,
                "mask_path": mask_path,
            }
            return input_tensor, target_tensor, metadata

        return input_tensor, target_tensor


def export_dataset_class_statistics(dataset_path, output_csv, modality="rgb", edge_method="sobel"):
    dataset = DDOSDataset(dataset_path=dataset_path, modality=modality, edge_method=edge_method, augment=False)
    dataset.print_class_frequency()
    dataset.save_class_frequency_csv(output_csv)
    return output_csv
