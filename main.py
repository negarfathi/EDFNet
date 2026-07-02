# REQUIREMENTS:
# pip install -r requirements.txt
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

'''
Fast Experiment:
rm -rf checkpoints_check results_check visualizations_check
python main.py \
  --modality all \
  --model all \
  --device cuda \
  --dataset ./data/DDOS \
  --train \
  --test \
  --visualize \
  --edge_method sobel \
  --epochs 1 \
  --batch_size 2 \
  --learning_rate 5e-4 \
  --checkpoint_dir checkpoints_check \
  --results_dir results_check \
  --visualize_max 50 \
  --visualization_dir visualizations_check \
  --seed 42 \
  --runs 1
'''

'''
Full Experiment:
rm -rf checkpoints results visualizations
python main.py \
  --modality all \
  --model all \
  --device cuda \
  --dataset ./data/DDOS \
  --train \
  --test \
  --visualize \
  --edge_method sobel \
  --epochs 50 \
  --batch_size 16 \
  --learning_rate 5e-4 \
  --checkpoint_dir checkpoints \
  --results_dir results \
  --visualize_max all \
  --visualization_dir visualizations \
  --seed 42 \
  --runs 1
'''

import os
import argparse
import random
import csv
import numpy
import torch

from src.train import train_model
from src.test import test_model, summarize_results
from src.dataset import export_dataset_class_statistics
from src.model import validate_model_availability


SUPPORTED_MODELS = [
    "unet",
    "unet_pretrained",
    "deeplabv3",
    "deeplabv3_pretrained",
    "deeplabv3plus",
    "deeplabv3plus_pretrained",
    "segformer",
    "segformer_pretrained",
    "all",
]

SUPPORTED_MODALITIES = ["rgb", "rgbd", "rgbe", "rgbde", "all"]
ALL_MODELS = [
    "unet",
    "unet_pretrained",
    "deeplabv3",
    "deeplabv3_pretrained",
    "deeplabv3plus",
    "deeplabv3plus_pretrained",
    "segformer",
    "segformer_pretrained",
]
ALL_MODALITIES = ["rgb", "rgbd", "rgbe", "rgbde"]

# Internal fixed value. It is not a command-line option because this benchmark
# should use one consistent weighting policy across all runs.
CLASS_WEIGHT_MAX = 10.0


def parse_visualize_max(value):
    """Accept either 'all' or a non-negative integer."""
    text = str(value).strip().lower()
    if text == "all":
        return -1
    try:
        number = int(text)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("visualize_max must be 'all' or a non-negative integer.") from exc
    if number < 0:
        raise argparse.ArgumentTypeError("Use --visualize_max all instead of a negative number.")
    return number


def set_global_seed(seed):
    """Set seeds and deterministic PyTorch/cuDNN settings for reproducibility."""
    os.environ.setdefault("PYTHONHASHSEED", str(seed))
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

    random.seed(seed)
    numpy.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    try:
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
    except Exception:
        pass

    try:
        torch.use_deterministic_algorithms(True, warn_only=True)
    except Exception:
        pass


def write_run_metadata(path, row):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    exists = os.path.exists(path)
    with open(path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(row.keys()))
        if not exists:
            writer.writeheader()
        writer.writerow(row)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="EDFNet: Early Fusion for Thin-Obstacle Segmentation")
    parser.add_argument("--modality", type=str, choices=SUPPORTED_MODALITIES, required=True,
                        help="Modality configuration: rgb, rgbd, rgbe, rgbde, or all.")
    parser.add_argument("--model", type=str, choices=SUPPORTED_MODELS, required=True,
                        help="Model architecture or all. all includes all eight supported models.")
    parser.add_argument("--device", type=str, choices=["cpu", "cuda"], required=True,
                        help="Computation device: cpu or cuda.")
    parser.add_argument("--dataset", type=str, required=True,
                        help="Path to dataset root directory. The code appends /data internally.")
    parser.add_argument("--train", action="store_true", help="Run training stage.")
    parser.add_argument("--test", action="store_true", help="Run testing stage.")
    parser.add_argument("--visualize", action="store_true", help="Save visual results during testing.")
    parser.add_argument("--edge_method", type=str, choices=["canny", "sobel"], required=True,
                        help="Edge extraction method: canny or sobel.")
    parser.add_argument("--epochs", type=int, required=True,
                        help="Number of training epochs.")
    parser.add_argument("--batch_size", type=int, required=True,
                        help="Batch size for training, validation, and testing.")
    parser.add_argument("--learning_rate", type=float, required=True,
                        help="Learning rate for Adam optimizer.")
    parser.add_argument("--results_dir", type=str, required=True,
                        help="Directory for CSV results and analysis files.")
    parser.add_argument("--checkpoint_dir", type=str, required=True,
                        help="Directory for model checkpoints.")
    parser.add_argument("--visualization_dir", type=str, required=True,
                        help="Directory for visualization images.")
    parser.add_argument("--visualize_max", type=parse_visualize_max, required=True,
                        help="Maximum visualization images per model/modality: use 'all' or a non-negative integer.")
    parser.add_argument("--seed", type=int, required=True,
                        help="Base random seed for reproducibility.")
    parser.add_argument("--runs", type=int, required=True,
                        help="Number of repeated runs. Seeds are seed, seed+1, ...")
    parser.add_argument("--export_class_stats", action="store_true",
                        help="Export class-frequency tables for train/validation/test splits before experiments.")
    args = parser.parse_args()

    if not args.train and not args.test:
        parser.error("At least one of --train or --test must be specified.")
    if args.visualize and not args.test:
        parser.error("--visualize requires --test.")
    if args.epochs <= 0:
        parser.error("--epochs must be positive.")
    if args.batch_size <= 0:
        parser.error("--batch_size must be positive.")
    if args.learning_rate <= 0:
        parser.error("--learning_rate must be positive.")
    if args.runs <= 0:
        parser.error("--runs must be positive.")

    if args.modality == "all":
        modalities = list(ALL_MODALITIES)
    else:
        modalities = [args.modality]

    if args.model == "all":
        models = list(ALL_MODELS)
    else:
        models = [args.model]

    # Fail before any long training starts if an optional model is unavailable.
    validate_model_availability(models)

    dataset_path = os.path.join(args.dataset, "data")
    train_path = os.path.join(dataset_path, "train")
    validation_path = os.path.join(dataset_path, "validation")
    test_path = os.path.join(dataset_path, "test")

    os.makedirs(args.results_dir, exist_ok=True)
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    os.makedirs(args.visualization_dir, exist_ok=True)

    if args.export_class_stats:
        export_dataset_class_statistics(train_path, os.path.join(args.results_dir, "class_frequency_train.csv"))
        export_dataset_class_statistics(validation_path, os.path.join(args.results_dir, "class_frequency_validation.csv"))
        export_dataset_class_statistics(test_path, os.path.join(args.results_dir, "class_frequency_test.csv"))

    for run_idx in range(args.runs):
        run_seed = args.seed + run_idx
        set_global_seed(run_seed)

        if args.runs == 1:
            run_name = f"run_01_seed_{run_seed}"
            run_results_dir = args.results_dir
            run_checkpoint_dir = args.checkpoint_dir
            run_visualization_dir = args.visualization_dir
        else:
            run_name = f"run_{run_idx + 1:02d}_seed_{run_seed}"
            run_results_dir = os.path.join(args.results_dir, run_name)
            run_checkpoint_dir = os.path.join(args.checkpoint_dir, run_name)
            run_visualization_dir = os.path.join(args.visualization_dir, run_name)

        os.makedirs(run_results_dir, exist_ok=True)
        os.makedirs(run_checkpoint_dir, exist_ok=True)
        os.makedirs(run_visualization_dir, exist_ok=True)
        print(f"\n=== Run {run_idx + 1}/{args.runs} | seed={run_seed} | deterministic=1 ===")

        write_run_metadata(os.path.join(run_results_dir, "run_metadata.csv"), {
            "run_name": run_name,
            "run_index": run_idx + 1,
            "seed": run_seed,
            "deterministic": 1,
            "models": ";".join(models),
            "modalities": ";".join(modalities),
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "learning_rate": args.learning_rate,
            "class_weight_max": CLASS_WEIGHT_MAX,
            "edge_method": args.edge_method,
            "results_dir": run_results_dir,
            "checkpoint_dir": run_checkpoint_dir,
            "visualization_dir": run_visualization_dir,
            "visualize_max": "all" if args.visualize_max < 0 else args.visualize_max,
        })

        for model in models:
            for modality in modalities:
                checkpoint_path = os.path.join(run_checkpoint_dir, f"{model}_{modality}.pth")
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
                                results_dir=run_results_dir,
                                checkpoint_dir=run_checkpoint_dir,
                                class_weight_max=CLASS_WEIGHT_MAX,
                                seed=run_seed,
                                run_name=run_name)
                if args.test:
                    test_model(test_path=test_path,
                               modality=modality,
                               device=args.device,
                               edge_method=args.edge_method,
                               checkpoint_path=checkpoint_path,
                               batch_size=args.batch_size,
                               visualize=args.visualize,
                               results_dir=run_results_dir,
                               visualization_dir=run_visualization_dir,
                               visualize_max=args.visualize_max,
                               seed=run_seed,
                               run_name=run_name)

        if args.test:
            summarize_results(run_results_dir)

    if args.test and args.runs > 1:
        summarize_results(args.results_dir)
