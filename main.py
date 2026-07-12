'''
Requirements:
pip install -r requirements.txt
'''

'''
Dataset:
git lfs install
git clone https://huggingface.co/datasets/benediktkol/DDOS ./data/DDOS
OR
pip install --upgrade huggingface_hub
hf download benediktkol/DDOS \
   --repo-type dataset \
   --local-dir ./data/DDOS \
   --include "data/*/neighbourhood/0/*" \
   --exclude "data/*/neighbourhood/[1-9]*/*"
'''

'''
Fast Experiment:
rm -rf checkpoints_check outputs_check visualizations_check
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
  --outputs_dir outputs_check \
  --visualize_max 50 \
  --visualization_dir visualizations_check \
  --seed 42 \
  --runs 1
'''

'''
Full Experiment:
rm -rf checkpoints outputs visualizations
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
  --outputs_dir outputs \
  --visualize_max all \
  --visualization_dir visualizations \
  --seed 42 \
  --runs 1
'''

import os
import argparse
import random
import numpy
import torch

from src.train import train_model
from src.test import test_model, summarize_results
from src.dataset import export_dataset_class_distribution
from src.model import validate_model_availability
from src.output import append_csv_row, csv_path, ensure_outputs_structure


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


METADATA_FIELDS = [
    "run_name", "run_index", "seed", "deterministic", "device", "dataset_path",
    "train_path", "validation_path", "test_path", "models", "modalities",
    "epochs", "batch_size", "learning_rate", "class_weight_max", "edge_method",
    "outputs_dir", "checkpoint_dir", "visualization_dir", "visualize",
    "visualize_max", "checkpoint_selection_metric", "checkpoint_type",
]


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
    parser.add_argument("--checkpoint_dir", type=str, required=True,
                        help="Directory for best-validation checkpoints.")
    parser.add_argument("--outputs_dir", type=str, required=True,
                        help="Directory for structured CSV output.")
    parser.add_argument("--visualization_dir", type=str, required=True,
                        help="Directory for visualization images.")
    parser.add_argument("--visualize_max", type=parse_visualize_max, required=True,
                        help="Maximum visualization images per model/modality: use 'all' or a non-negative integer.")
    parser.add_argument("--seed", type=int, required=True,
                        help="Base random seed for reproducibility.")
    parser.add_argument("--runs", type=int, required=True,
                        help="Number of repeated runs. Seeds are seed, seed+1, ...")
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

    modalities = list(ALL_MODALITIES) if args.modality == "all" else [args.modality]
    models = list(ALL_MODELS) if args.model == "all" else [args.model]

    # Fail before any long training starts if an optional model is unavailable.
    validate_model_availability(models)

    dataset_path = os.path.join(args.dataset, "data")
    train_path = os.path.join(dataset_path, "train")
    validation_path = os.path.join(dataset_path, "validation")
    test_path = os.path.join(dataset_path, "test")

    os.makedirs(args.checkpoint_dir, exist_ok=True)
    os.makedirs(args.visualization_dir, exist_ok=True)
    ensure_outputs_structure(args.outputs_dir)

    # Generate one combined class-distribution CSV for paper/debugging.
    export_dataset_class_distribution(
        {"train": train_path, "validation": validation_path, "test": test_path},
        csv_path(args.outputs_dir, "results", "class_distribution.csv"),
        modality="rgb",
        edge_method=args.edge_method,
    )

    for run_idx in range(args.runs):
        run_seed = args.seed + run_idx
        set_global_seed(run_seed)

        if args.runs == 1:
            run_name = f"run_01_seed_{run_seed}"
            run_outputs_dir = args.outputs_dir
            run_checkpoint_dir = args.checkpoint_dir
            run_visualization_dir = args.visualization_dir
        else:
            run_name = f"run_{run_idx + 1:02d}_seed_{run_seed}"
            run_outputs_dir = os.path.join(args.outputs_dir, run_name)
            run_checkpoint_dir = os.path.join(args.checkpoint_dir, run_name)
            run_visualization_dir = os.path.join(args.visualization_dir, run_name)
            ensure_outputs_structure(run_outputs_dir)
            os.makedirs(run_checkpoint_dir, exist_ok=True)
            os.makedirs(run_visualization_dir, exist_ok=True)

        print(f"\n=== Run {run_idx + 1}/{args.runs} | seed={run_seed} | deterministic=1 ===")

        append_csv_row(csv_path(run_outputs_dir, "metadata", "metadata.csv"), {
            "run_name": run_name,
            "run_index": run_idx + 1,
            "seed": run_seed,
            "deterministic": 1,
            "device": args.device,
            "dataset_path": dataset_path,
            "train_path": train_path,
            "validation_path": validation_path,
            "test_path": test_path,
            "models": ";".join(models),
            "modalities": ";".join(modalities),
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "learning_rate": args.learning_rate,
            "class_weight_max": CLASS_WEIGHT_MAX,
            "edge_method": args.edge_method,
            "outputs_dir": run_outputs_dir,
            "checkpoint_dir": run_checkpoint_dir,
            "visualization_dir": run_visualization_dir,
            "visualize": int(args.visualize),
            "visualize_max": "all" if args.visualize_max < 0 else args.visualize_max,
            "checkpoint_selection_metric": "validation_TOCS",
            "checkpoint_type": "best_validation_checkpoint",
        }, fieldnames=METADATA_FIELDS)

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
                                outputs_dir=run_outputs_dir,
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
                               outputs_dir=run_outputs_dir,
                               visualization_dir=run_visualization_dir,
                               visualize_max=args.visualize_max,
                               seed=run_seed,
                               run_name=run_name)

        if args.test:
            summarize_results(run_outputs_dir)

    if args.test and args.runs > 1:
        summarize_results(args.outputs_dir)
