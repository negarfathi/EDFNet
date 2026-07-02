import os
import csv
import cv2
import numpy
import torch
import albumentations
from glob import glob
from collections import Counter
from torch.utils.data import Dataset


# Canonical DDOS segmentation labels.
# Masks keep their original pixel IDs, then every split maps them to these fixed indices.
CANONICAL_CLASS_IDS = [0, 80, 100, 140, 160, 180, 200, 220, 240, 255]

CLASS_ID_TO_NAME = {
    0: "Background",
    80: "Other",
    100: "Animals",
    140: "Vehicles",
    160: "Buildings",
    180: "Trees",
    200: "Large Mesh",
    220: "Small Mesh",
    240: "Thin Structures",
    255: "Ultra-thin",
}

CLASS_NAMES = [CLASS_ID_TO_NAME[cid] for cid in CANONICAL_CLASS_IDS]
CLASS_ID_TO_INDEX = {class_id: idx for idx, class_id in enumerate(CANONICAL_CLASS_IDS)}
INDEX_TO_CLASS_ID = {idx: class_id for class_id, idx in CLASS_ID_TO_INDEX.items()}

THIN_CLASS_IDS = [240, 255]
THIN_CLASS_INDICES = [CLASS_ID_TO_INDEX[cid] for cid in THIN_CLASS_IDS]

IMAGENET_MEAN = numpy.array([0.485, 0.456, 0.406], dtype=numpy.float32)
IMAGENET_STD = numpy.array([0.229, 0.224, 0.225], dtype=numpy.float32)


def get_thin_class_indices():
    return list(THIN_CLASS_INDICES)


def is_pretrained_model(model_name):
    model_name = (model_name or "").lower()
    return model_name.endswith("_pretrained") or model_name == "segformer_pretrained"


def normalization_for_model(model_name):
    return "imagenet" if is_pretrained_model(model_name) else "none"


def validate_known_labels(mask, allowed_ids, mask_path=None):
    unique_ids = set(map(int, numpy.unique(mask)))
    unknown = sorted(unique_ids - set(allowed_ids))
    if unknown:
        where = f" in {mask_path}" if mask_path else ""
        raise ValueError(
            f"Unknown segmentation label(s){where}: {unknown}. "
            f"Expected only canonical DDOS labels: {list(allowed_ids)}"
        )


class DDOSDataset(Dataset):
    def __init__(self, dataset_path, modality, edge_method, size=256, augment=True,
                 return_metadata=False, normalization="none", unknown_label_policy="error",
                 ignore_index=-100, class_weight_max=10.0, depth_normalization_value=65535.0):
        self.dataset_path = dataset_path
        self.modality = modality
        self.edge_method = edge_method
        self.size = size
        self.augment = augment
        self.return_metadata = return_metadata
        self.normalization = normalization
        self.unknown_label_policy = unknown_label_policy
        self.ignore_index = ignore_index
        self.class_weight_max = class_weight_max
        self.depth_normalization_value = float(depth_normalization_value)

        if self.unknown_label_policy not in {"error", "ignore"}:
            raise ValueError("unknown_label_policy must be either 'error' or 'ignore'.")

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

        # Fixed class mapping for every split.
        self.class_ids = list(CANONICAL_CLASS_IDS)
        self.class_to_index = dict(CLASS_ID_TO_INDEX)
        self.index_to_class = dict(INDEX_TO_CLASS_ID)
        self.class_names = list(CLASS_NAMES)
        self.num_classes = len(self.class_ids)

        self.class_pixel_counts = Counter({cid: 0 for cid in self.class_ids})
        self.observed_unknown_class_counts = Counter()
        for mask_path in self.mask_images:
            mask = cv2.imread(mask_path, cv2.IMREAD_UNCHANGED)
            if mask is None:
                continue
            if mask.ndim == 3:
                mask = mask[:, :, 0]
            unique_classes, pixels_per_class = numpy.unique(mask, return_counts=True)
            for class_id, pixel_count in zip(unique_classes, pixels_per_class):
                class_id = int(class_id)
                if class_id in self.class_to_index:
                    self.class_pixel_counts[class_id] += int(pixel_count)
                else:
                    self.observed_unknown_class_counts[class_id] += int(pixel_count)

        if self.observed_unknown_class_counts and self.unknown_label_policy == "error":
            raise ValueError(
                f"Unknown segmentation labels found in {dataset_path}: "
                f"{dict(self.observed_unknown_class_counts)}. "
                f"Expected canonical DDOS labels: {self.class_ids}"
            )

        total_pixels = sum(self.class_pixel_counts[cid] for cid in self.class_ids)
        self.total_pixels = total_pixels
        raw_weights = []
        for cid in self.class_ids:
            count = self.class_pixel_counts[cid]
            if count > 0 and total_pixels > 0:
                raw_weights.append(total_pixels / (self.num_classes * count))
            else:
                raw_weights.append(0.0)
        self.raw_class_weights = raw_weights
        self.class_weights = self._clip_and_normalize_class_weights(raw_weights)

        # Split augmentations into two stages.
        # 1) Spatial transforms are applied jointly to RGB, depth, edge, and mask.
        # 2) Photometric transforms are applied only to RGB.
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
                "depth": "image",
                "edge": "image",
            },
        ) if augment else None

        self.color_transform = albumentations.Compose([
            albumentations.RandomBrightnessContrast(p=0.5),
            albumentations.CLAHE(clip_limit=2.0, tile_grid_size=(8, 8), p=0.3),
        ]) if augment else None

    def _clip_and_normalize_class_weights(self, raw_weights):
        weights = numpy.array(raw_weights, dtype=numpy.float32)
        positive = weights > 0
        if not positive.any():
            return [1.0 for _ in raw_weights]

        if self.class_weight_max is not None and self.class_weight_max > 0:
            weights[positive] = numpy.clip(weights[positive], 0.0, float(self.class_weight_max))

        # Keep the average positive weight around 1.0 so the loss scale remains stable.
        mean_positive = float(weights[positive].mean())
        if mean_positive > 0:
            weights[positive] = weights[positive] / mean_positive

        # If a class is absent in the training split, it will not appear in the loss;
        # keep its weight at 0 to avoid inventing an arbitrary training signal.
        weights[~positive] = 0.0
        return [float(w) for w in weights]

    def class_frequency_rows(self):
        rows = []
        for idx, class_id in enumerate(self.class_ids):
            count = self.class_pixel_counts[class_id]
            percent = 100.0 * count / max(1, self.total_pixels)
            rows.append({
                "mapped_index": idx,
                "original_class_id": class_id,
                "class_name": self.class_names[idx],
                "pixel_count": count,
                "pixel_percent": percent,
                "raw_class_weight": self.raw_class_weights[idx],
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
                f"orig_id={row['original_class_id']:<3} | pixels={row['pixel_count']:<12} | "
                f"{row['pixel_percent']:.6f}% | raw_w={row['raw_class_weight']:.4f} | "
                f"w={row['class_weight']:.4f}"
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

    def normalize_depth(self, depth_image):
        depth_image = depth_image.astype(numpy.float32)
        if depth_image.ndim == 3:
            depth_image = depth_image[:, :, 0]
        if depth_image.max() <= 0:
            return depth_image

        # DDOS depth PNGs are uint16. Use a fixed scale instead of per-image max
        # so absolute distance cues are preserved across images.
        divisor = self.depth_normalization_value if depth_image.max() > 255 else 255.0
        return numpy.clip(depth_image / divisor, 0.0, 1.0)

    def normalize_input_image(self, input_image):
        if self.normalization == "none":
            return input_image.astype(numpy.float32)

        if self.normalization != "imagenet":
            raise ValueError(f"Unknown normalization mode: {self.normalization}")

        input_image = input_image.astype(numpy.float32).copy()
        input_image[:, :, :3] = (input_image[:, :, :3] - IMAGENET_MEAN) / IMAGENET_STD

        # Extra modalities are already in [0,1]. Center them to roughly [-1,1].
        if input_image.shape[2] > 3:
            input_image[:, :, 3:] = (input_image[:, :, 3:] - 0.5) / 0.5
        return input_image

    def remap_mask(self, mask_image, mask_path=None):
        if mask_image.ndim == 3:
            mask_image = mask_image[:, :, 0]

        known = numpy.isin(mask_image, self.class_ids)
        if not numpy.all(known):
            unknown = sorted(set(map(int, numpy.unique(mask_image[~known]))))
            if self.unknown_label_policy == "error":
                where = f" in {mask_path}" if mask_path else ""
                raise ValueError(
                    f"Unknown segmentation label(s){where}: {unknown}. "
                    f"Expected only canonical DDOS labels: {self.class_ids}"
                )

        remapped = numpy.full(mask_image.shape, self.ignore_index, dtype=numpy.int64)
        for original_id, mapped_idx in self.class_to_index.items():
            remapped[mask_image == original_id] = mapped_idx
        return remapped

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
        depth_image = self.normalize_depth(depth_image)
        edge_image = self.extract_edges((rgb_image * 255).astype(numpy.uint8))

        if self.augment and self.spatial_transform:
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

            if self.color_transform:
                color_augmented = self.color_transform(image=(rgb_image * 255).astype(numpy.uint8))
                rgb_image = color_augmented["image"].astype(numpy.float32) / 255.0
                edge_image = self.extract_edges((rgb_image * 255).astype(numpy.uint8))

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

        input_image = self.normalize_input_image(input_image)
        mask_remapped = self.remap_mask(mask_image, mask_path=mask_path)

        input_tensor = torch.tensor(input_image, dtype=torch.float32).permute(2, 0, 1)
        target_tensor = torch.tensor(mask_remapped, dtype=torch.long)

        if self.return_metadata:
            metadata = {
                "sample_key": sample_key,
                "rgb_path": rgb_path,
                "depth_path": depth_path,
                "mask_path": mask_path,
                "normalization": self.normalization,
            }
            return input_tensor, target_tensor, metadata

        return input_tensor, target_tensor


def export_dataset_class_statistics(dataset_path, output_csv, modality="rgb", edge_method="sobel"):
    dataset = DDOSDataset(dataset_path=dataset_path, modality=modality, edge_method=edge_method, augment=False)
    dataset.print_class_frequency()
    dataset.save_class_frequency_csv(output_csv)
    return output_csv
