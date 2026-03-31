import os
import cv2
import numpy
import torch
import albumentations
from glob import glob
from collections import Counter
from torch.utils.data import Dataset

class DDOSDataset(Dataset):
    def __init__(self, dataset_path, modality, edge_method, size=256, augment=True):
        self.dataset_path = dataset_path
        self.modality = modality
        self.edge_method = edge_method
        self.size = size
        self.augment = augment

        # --- LOAD RAW PATH LISTS ---
        rgb_images  = sorted(glob(os.path.join(dataset_path, "**", "image", "*.png"), recursive=True))
        depth_images = sorted(glob(os.path.join(dataset_path, "**", "depth", "*.png"), recursive=True))
        mask_images  = sorted(glob(os.path.join(dataset_path, "**", "segmentation", "*.png"), recursive=True))

        # ---- build unique keys based on env / flight / frame ----
        def make_key(path):
            """
            dataset_path = .../data/train
            rel example: 'neighbourhood/0/image/12.png'
            parts = ['neighbourhood', '0', 'image', '12.png']
            We want key: 'neighbourhood/0/12'
            """
            rel = os.path.relpath(path, dataset_path)
            parts = rel.split(os.sep)
            if len(parts) < 3:
                # very defensive fallback
                stem = os.path.splitext(parts[-1])[0]
                return stem
            env = parts[0]          # 'neighbourhood' or similar
            flight = parts[1]       # '0', '1', ...
            frame = os.path.splitext(parts[-1])[0]  # '0'..'99'
            return f"{env}/{flight}/{frame}"

        rgb_dict   = {make_key(p): p for p in rgb_images}
        depth_dict = {make_key(p): p for p in depth_images}
        mask_dict  = {make_key(p): p for p in mask_images}

        common_keys = sorted(set(rgb_dict) & set(depth_dict) & set(mask_dict))

        if len(common_keys) == 0:
            raise RuntimeError("NO MATCHING RGB/DEPTH/MASK TRIPLETS FOUND! Check dataset path and structure.")

        self.rgb_images   = [rgb_dict[k]   for k in common_keys]
        self.depth_images = [depth_dict[k] for k in common_keys]
        self.mask_images  = [mask_dict[k]  for k in common_keys]

        print(f"[DDOSDataset] Loaded samples: {len(self.rgb_images)} from {dataset_path}")

        # --- CLASS STATISTICS ---
        all_class_ids = []
        class_pixel_counts = Counter()

        for mask_path in self.mask_images:
            mask = cv2.imread(mask_path, cv2.IMREAD_UNCHANGED)
            if mask is None:
                continue
            unique_classes, pixels_per_class = numpy.unique(mask, return_counts=True)
            all_class_ids.extend(unique_classes)
            for class_id, pixel_count in zip(unique_classes, pixels_per_class):
                class_pixel_counts[int(class_id)] += int(pixel_count)

        self.class_ids = sorted(set(map(int, all_class_ids)))
        self.class_to_index = {class_id: idx for idx, class_id in enumerate(self.class_ids)}
        self.num_classes = len(self.class_ids)

        total_pixels = sum(class_pixel_counts[cid] for cid in self.class_ids)
        self.class_weights = [
            (total_pixels / (self.num_classes * class_pixel_counts[cid]))
            if class_pixel_counts[cid] > 0 else 0
            for cid in self.class_ids
        ]

        # augmentations
        self.transform = albumentations.Compose([
            albumentations.HorizontalFlip(p=0.5),
            albumentations.ShiftScaleRotate(
                shift_limit=0.05, scale_limit=0.1, rotate_limit=10,
                border_mode=cv2.BORDER_REFLECT, p=0.5
            ),
            albumentations.RandomResizedCrop(
                size=(self.size, self.size), scale=(0.8, 1.0),
                ratio=(0.9, 1.1), p=0.5
            ),
            albumentations.RandomBrightnessContrast(p=0.5),
            albumentations.CLAHE(clip_limit=2.0, tile_grid_size=(8, 8), p=0.3)
        ]) if augment else None

    # --- EDGE EXTRACTION ---
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
        rgb_path   = self.rgb_images[index]
        depth_path = self.depth_images[index]
        mask_path  = self.mask_images[index]

        rgb_image = cv2.imread(rgb_path)[:, :, ::-1]
        depth_image = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED)
        mask_image = cv2.imread(mask_path, cv2.IMREAD_UNCHANGED)

        if rgb_image is None or depth_image is None or mask_image is None:
            raise RuntimeError(f"FAILED TO LOAD DATA:\n{rgb_path}\n{depth_path}\n{mask_path}")

        # resizing
        rgb_image = cv2.resize(rgb_image, (self.size, self.size))
        depth_image = cv2.resize(depth_image, (self.size, self.size))
        mask_image = cv2.resize(mask_image, (self.size, self.size), interpolation=cv2.INTER_NEAREST)

        # CLAHE
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        for c in range(3):
            rgb_image[:, :, c] = clahe.apply(rgb_image[:, :, c])

        rgb_image = rgb_image.astype(numpy.float32) / 255.0

        depth_image = depth_image.astype(numpy.float32)
        if depth_image.max() > 0:
            depth_image /= depth_image.max()

        edge_image = self.extract_edges((rgb_image * 255).astype(numpy.uint8))

        # augmentations
        if self.augment and self.transform:
            augmented = self.transform(
                image=(rgb_image * 255).astype(numpy.uint8),
                mask=mask_image
            )
            rgb_image = augmented["image"].astype(numpy.float32) / 255.0
            mask_image = augmented["mask"]

        # modality fusion
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

        # label remapping
        mask_mapper = numpy.vectorize(lambda v: self.class_to_index.get(int(v), 0))
        mask_remapped = mask_mapper(mask_image).astype(numpy.int64)

        input_tensor = torch.tensor(input_image, dtype=torch.float32).permute(2, 0, 1)
        target_tensor = torch.tensor(mask_remapped, dtype=torch.long)

        return input_tensor, target_tensor
