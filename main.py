# REQUIREMENTS:
# pip install -r requirements.txt

# DATASET:
# git lfs install
# git clone https://huggingface.co/datasets/benediktkol/DDOS ./data/DDOS
# OR
# pip install --upgrade huggingface_hub
# hf download benediktkol/DDOS \
#   --repo-type dataset \
#   --local-dir ./data/DDOS \
#   --include "data/*/neighbourhood/0/*" \
#   --exclude "data/*/neighbourhood/[1-9]*/*"


# EXECUTION:
# python main.py --modality <rgb/rgbd/rgbe/rgbde/all> --model <unet/unet_pretrained/deeplabv3/deeplabv3_pretrained/all> --device <cpu/cuda> --dataset <path/to/training_dataset> --train --test --visualize --edge_method <canny/sobel> --epochs <num_epochs> --batch_size <batch_size> --learning_rate <learning_rate>
# python main.py --modality all --model all --device cuda --dataset ./data/DDOS --train --test --visualize --edge_method sobel --epochs 50 --batch_size 16 --learning_rate 5e-4
# OR
# python main.py --modality all --model all --device cpu --dataset ./data/DDOS --train --test --visualize --edge_method sobel --epochs 25 --batch_size 8 --learning_rate 1e-4

import os
import argparse
from src.train import train_model
from src.test import test_model

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="EDFNet: Early Fusion for Thin-Obstacle Segmentation")
    parser.add_argument("--modality", type=str, choices=["rgb", "rgbd", "rgbe", "rgbde", "all"],
                        help="Modality configuration: RGB, RGB+D, RGB+E, RGB+D+E, or all.")
    parser.add_argument("--model", type=str, choices=["unet", "unet_pretrained", "deeplabv3", "deeplabv3_pretrained", "all"],
                        help="Model architecture: U-Net, DeepLabV3, pretrained variants, or all.")
    parser.add_argument("--device", type=str, choices=["cpu", "cuda"],
                        help="Computation device: CPU or CUDA.")
    parser.add_argument("--dataset", type=str,
                        help="Path to dataset root directory.")
    parser.add_argument("--train", action="store_true",
                        help="Run training stage.")
    parser.add_argument("--test", action="store_true",
                        help="Run testing stage.")
    parser.add_argument("--visualize", action="store_true",
                        help="Save visual results.")
    parser.add_argument("--edge_method", type=str, choices=["canny", "sobel"],
                        help="Edge extraction method: canny or sobel.")
    parser.add_argument("--epochs", type=int,
                        help="Number of training epochs.")
    parser.add_argument("--batch_size", type=int,
                        help="Batch size for training and testing.")
    parser.add_argument("--learning_rate", type=float,
                        help="Learning rate for optimizer.")
    args = parser.parse_args()

    if args.modality == "all":
        modalities = ["rgb", "rgbd", "rgbe", "rgbde"]
    else:
        modalities = [args.modality]

    if args.model == "all":
        models = ["unet", "unet_pretrained", "deeplabv3", "deeplabv3_pretrained"]
    else:
        models = [args.model]

    dataset_path = os.path.join(args.dataset, "data")
    train_path = os.path.join(dataset_path, "train")
    validation_path = os.path.join(dataset_path, "validation")
    test_path = os.path.join(dataset_path, "test")

    for model in models:
        for modality in modalities:
            checkpoint_path = os.path.join("checkpoints", f"{model}_{modality}.pth")
            if args.train:
                train_model(train_path=train_path,
                            validation_path=validation_path,
                            modality=modality,
                            model_name=model,
                            device=args.device,
                            edge_method=args.edge_method,
                            epochs=args.epochs,
                            batch_size=args.batch_size,
                            learning_rate=args.learning_rate)
            if args.test:
                test_model(test_path=test_path,
                           modality=modality,
                           device=args.device,
                           edge_method = args.edge_method,
                           checkpoint_path=checkpoint_path,
                           batch_size=args.batch_size,
                           visualize=args.visualize)