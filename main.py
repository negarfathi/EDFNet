# REQUIREMENTS:
# pip install -r requirements.txt
# Optional stronger baselines require a recent segmentation_models_pytorch version:
# pip install -U segmentation-models-pytorch timm

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

# EXECUTION EXAMPLES:
# python main.py --modality all --model all --device cuda --dataset ./data/DDOS --train --test --visualize --edge_method sobel --epochs 50 --batch_size 16 --learning_rate 5e-4
# python main.py --modality all --model deeplabv3plus_pretrained --device cuda --dataset ./data/DDOS --train --test --visualize --edge_method sobel --epochs 50 --batch_size 16 --learning_rate 5e-4
# python main.py --modality all --model segformer_pretrained --device cuda --dataset ./data/DDOS --train --test --visualize --edge_method sobel --epochs 50 --batch_size 16 --learning_rate 5e-4

import os
import argparse
from src.train import train_model
from src.test import test_model, summarize_results
from src.dataset import export_dataset_class_statistics


SUPPORTED_MODELS = [
    "unet",
    "unet_pretrained",
    "deeplabv3",
    "deeplabv3_pretrained",
    "deeplabv3plus",
    "deeplabv3plus_pretrained",
    "segformer_pretrained",
    "all",
]

SUPPORTED_MODALITIES = ["rgb", "rgbd", "rgbe", "rgbde", "all"]


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="EDFNet: Early Fusion for Thin-Obstacle Segmentation")
    parser.add_argument("--modality", type=str, choices=SUPPORTED_MODALITIES, required=True,
                        help="Modality configuration: RGB, RGB+D, RGB+E, RGB+D+E, or all.")
    parser.add_argument("--model", type=str, choices=SUPPORTED_MODELS, required=True,
                        help="Model architecture, pretrained variants, stronger baselines, or all.")
    parser.add_argument("--device", type=str, choices=["cpu", "cuda"], required=True,
                        help="Computation device: CPU or CUDA.")
    parser.add_argument("--dataset", type=str, required=True,
                        help="Path to dataset root directory.")
    parser.add_argument("--train", action="store_true", help="Run training stage.")
    parser.add_argument("--test", action="store_true", help="Run testing stage.")
    parser.add_argument("--visualize", action="store_true", help="Save visual results.")
    parser.add_argument("--edge_method", type=str, choices=["canny", "sobel"], default="sobel",
                        help="Edge extraction method: canny or sobel.")
    parser.add_argument("--epochs", type=int, default=50, help="Number of training epochs.")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size for training and testing.")
    parser.add_argument("--learning_rate", type=float, default=5e-4, help="Learning rate for optimizer.")
    parser.add_argument("--results_dir", type=str, default="results", help="Directory for CSV results and analysis files.")
    parser.add_argument("--visualize_max", type=int, default=20,
                        help="Maximum number of visualization images per model/modality. Use -1 for all.")
    parser.add_argument("--export_class_stats", action="store_true",
                        help="Export class-frequency tables for train/validation/test splits.")
    args = parser.parse_args()

    if args.modality == "all":
        modalities = ["rgb", "rgbd", "rgbe", "rgbde"]
    else:
        modalities = [args.modality]

    if args.model == "all":
        # Keeps original 16 experiments and adds one stronger recent baseline.
        # You can manually run deeplabv3plus/segformer if time permits.
        models = ["unet", "unet_pretrained", "deeplabv3", "deeplabv3_pretrained", "deeplabv3plus_pretrained"]
    else:
        models = [args.model]

    dataset_path = os.path.join(args.dataset, "data")
    train_path = os.path.join(dataset_path, "train")
    validation_path = os.path.join(dataset_path, "validation")
    test_path = os.path.join(dataset_path, "test")

    os.makedirs(args.results_dir, exist_ok=True)
    os.makedirs("checkpoints", exist_ok=True)

    if args.export_class_stats:
        export_dataset_class_statistics(train_path, os.path.join(args.results_dir, "class_frequency_train.csv"))
        export_dataset_class_statistics(validation_path, os.path.join(args.results_dir, "class_frequency_validation.csv"))
        export_dataset_class_statistics(test_path, os.path.join(args.results_dir, "class_frequency_test.csv"))

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
                            learning_rate=args.learning_rate,
                            results_dir=args.results_dir)
            if args.test:
                test_model(test_path=test_path,
                           modality=modality,
                           device=args.device,
                           edge_method=args.edge_method,
                           checkpoint_path=checkpoint_path,
                           batch_size=args.batch_size,
                           visualize=args.visualize,
                           results_dir=args.results_dir,
                           visualize_max=args.visualize_max)

    if args.test:
        summarize_results(args.results_dir)
