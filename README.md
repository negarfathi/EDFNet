# EDFNet

EDFNet is a modular research framework for the empirical analysis of **early-fusion cues** in UAV thin-obstacle semantic segmentation. It evaluates how appearance, depth, and edge cues affect segmentation performance under a shared training and evaluation pipeline.

The framework targets challenging thin structures, such as wires, branches, poles, fences, and mesh-like objects, in cluttered aerial scenes from the [DDOS](https://huggingface.co/datasets/benediktkol/DDOS) dataset. These objects often occupy very few pixels, have weak visual contrast, and are highly sensitive to boundary errors.

EDFNet evaluates four input-modality configurations:

* **RGB** – standard three-channel visual input.
* **RGB-D** – RGB with an additional normalized depth channel.
* **RGB-E** – RGB with an additional edge channel.
* **RGB-D-E** – RGB with both depth and edge channels.

The framework supports eight segmentation backbones:

* **U-Net**
* **U-Net (pretrained)**
* **DeepLabV3**
* **DeepLabV3 (pretrained)**
* **DeepLabV3+**
* **DeepLabV3+ (pretrained)**
* **SegFormer**
* **SegFormer (pretrained)**

Together, these settings define **32 model–modality configurations** on the DDOS dataset. EDFNet includes training, validation-based checkpoint selection, testing, structured CSV reporting, thin-obstacle analysis, runtime reporting, and paper-oriented visualization utilities.

## Requirements

The required Python packages are listed in `requirements.txt`.

The main dependencies are:

* PyTorch
* TorchVision
* NumPy
* OpenCV
* Albumentations
* Matplotlib
* segmentation-models-pytorch
* timm

For CUDA execution, install a PyTorch build that is compatible with your GPU and CUDA version.

## Installation

Clone the repository:

```bash
git clone https://github.com/negarfathi/EDFNet.git
cd EDFNet
```

Create and activate a Python virtual environment:

```bash
python -m venv .venv
source .venv/bin/activate
```

Install the required packages:

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

## Dataset

EDFNet expects the DDOS dataset to have the following structure:

```text
<data_root>/data/train
<data_root>/data/validation
<data_root>/data/test
```

You can download the DDOS dataset using Git LFS:

```bash
git lfs install
git clone https://huggingface.co/datasets/benediktkol/DDOS ./data/DDOS
```

Then pass the dataset root to `--dataset`:

```bash
--dataset ./data/DDOS
```

## Execution

EDFNet is executed through `main.py`, which supports training, testing, visualization, modality selection, backbone selection, device selection, edge-extraction method selection, and experiment configuration.

```bash
python main.py \
  --modality <modality_name/all> \
  --model <model_name/all> \
  --device <cpu/cuda> \
  --dataset <path/to/DDOS> \
  --train \
  --test \
  --visualize \
  --edge_method <canny/sobel> \
  --epochs <num_epochs> \
  --batch_size <batch_size> \
  --learning_rate <learning_rate> \
  --checkpoint_dir <checkpoint_dir> \
  --outputs_dir <outputs_dir> \
  --visualization_dir <visualization_dir> \
  --visualize_max <integer/all> \
  --seed <seed> \
  --runs <num_runs>
```

Arguments:

* `--modality`: Input modality. Options: `rgb`, `rgbd`, `rgbe`, `rgbde`, and `all`.
* `--model`: Segmentation model. Options: `unet`, `unet_pretrained`, `deeplabv3`, `deeplabv3_pretrained`, `deeplabv3plus`, `deeplabv3plus_pretrained`, `segformer`, `segformer_pretrained`, and `all`.
* `--device`: Computation device. Options: `cpu` and `cuda`.
* `--dataset`: Path to the DDOS dataset root.
* `--train`: Run training.
* `--test`: Run testing on the test split.
* `--visualize`: Save qualitative visualizations during testing.
* `--edge_method`: Edge-extraction method. Options: `canny` and `sobel`.
* `--epochs`: Number of training epochs.
* `--batch_size`: Batch size for training, validation, and testing.
* `--learning_rate`: Learning rate for the Adam optimizer.
* `--checkpoint_dir`: Directory in which the best validation checkpoints are stored.
* `--outputs_dir`: Directory in which structured CSV outputs are stored.
* `--visualization_dir`: Directory in which qualitative visualizations are stored.
* `--visualize_max`: Maximum number of visualizations saved for each model–modality configuration. Use an integer or `all`.
* `--seed`: Random seed.
* `--runs`: Number of repeated runs. Seeds are assigned as `seed`, `seed+1`, and so on.

To run all 32 model–modality configurations:

```bash
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
```

## Evaluation Metrics

EDFNet reports standard segmentation metrics, thin-class metrics, runtime metrics, and a task-oriented composite score.

### Standard metrics

* **mIoU** – mean intersection over union across classes.
* **bIoU** – boundary IoU for evaluating boundary-localization quality.
* **Recall** – macro-averaged recall across classes.
* **Precision** – macro-averaged precision across classes.
* **FPR** – macro-averaged false-positive rate.
* **FPS** – inference throughput.
* **Latency** – average inference latency per batch.

### Per-class metrics

For each semantic class, EDFNet reports:

* IoU
* Recall
* Precision
* FPR
* Ground-truth pixels
* Predicted pixels
* True-positive pixels
* False-positive pixels
* False-negative pixels

### Thin-obstacle metrics

EDFNet separately reports metrics for the following classes:

* **Thin Structures**
* **Ultra-thin**

The thin-obstacle summary includes:

* Thin Structures IoU
* Ultra-thin IoU
* Mean thin IoU
* Thin Structures recall
* Ultra-thin recall
* Mean thin recall
* Thin Structures precision
* Ultra-thin precision
* Mean thin precision
* Thin Structures FPR
* Ultra-thin FPR
* Mean thin FPR
* Thin Structures F2
* Ultra-thin F2
* Mean thin F2

### Composite score

* **TOCS** – Thin-Obstacle Composite Score, a task-oriented ranking metric that combines thin-class IoU, thin-class F2, boundary IoU, mIoU, and thin-class precision. TOCS is reported alongside the standard metrics and is used to compare model–modality configurations.

## Output Structure

After execution, EDFNet writes checkpoints, structured CSV files, and visualizations.

```text
checkpoints/
  <model>_<modality>.pth

outputs/
  results/
    overall.csv
    per_class.csv
    thin.csv
    modality.csv
    model.csv
    class_distribution.csv

  debug/
    validation_history.csv
    error_analysis.csv
    checks.csv

  metadata/
    metadata.csv

visualizations/
  generated/
    <model>_<modality>/
```

### `outputs/results/overall.csv`

This file contains the main configuration-level results. Each row corresponds to one model–modality configuration.

Important columns include:

* `rank_by_tocs`
* `model`
* `pretrained`
* `modality`
* `run_name`
* `seed`
* `checkpoint_epoch`
* `miou`
* `biou`
* `recall`
* `precision`
* `fpr`
* `fps`
* `latency_ms`
* `tocs`
* `best_val_tocs`

### `outputs/results/per_class.csv`

This file contains per-class results for every model–modality configuration.

### `outputs/results/thin.csv`

This file contains thin-obstacle- and ultra-thin-specific metrics, including IoU, recall, precision, FPR, F2, and mean thin metrics.

### `outputs/results/modality.csv`

This file contains results aggregated by modality. It is useful for comparing RGB, RGB-D, RGB-E, and RGB-D-E across model families.

### `outputs/results/model.csv`

This file contains results aggregated by model. It is useful for comparing backbone behavior across modalities.

### `outputs/debug/validation_history.csv`

This file contains epoch-level validation metrics and checkpoint information.

### `outputs/debug/error_analysis.csv`

This file contains sample-level error analyses used to identify successful cases, failure cases, and thin-obstacle error cases for qualitative inspection.

### `outputs/debug/checks.csv`

This file contains sanity checks for each training and testing stage, including checkpoint-existence checks, metric-validity checks, class-count validation, tensor-shape validation, and visualization counts.

## Visualization

EDFNet saves paper-oriented qualitative figures for selected samples.

Each visualization includes:

* RGB input
* Ground-truth segmentation
* Predicted segmentation
* Thin-error map

The thin-error map uses the following colors:

* **Green** – correctly detected thin or ultra-thin pixels
* **Red** – missed thin or ultra-thin pixels
* **Blue** – false-positive thin or ultra-thin pixels
* **Black** – non-thin or irrelevant regions

For thin-obstacle and failure cases, EDFNet can also save zoomed-in visualizations centered on the thin-error region.

## Reproducibility

EDFNet uses a fixed seed and deterministic PyTorch settings where possible. The `--runs` argument supports repeated experiments by incrementing the base seed.

For example:

```bash
--seed 42 --runs 3
```

This command runs experiments using the following seeds:

```text
42, 43, 44
```

## Related Papers

This repository is based on the following work:

N. Fathi, “EDFNet: Early Fusion of Edge and Depth for Thin-Obstacle Segmentation in UAV Navigation,” *arXiv preprint* [arXiv:2604.09694](https://doi.org/10.48550/arXiv.2604.09694), 2026.

Artifact archive:

* Zenodo DOI: [https://doi.org/10.5281/zenodo.19683169](https://doi.org/10.5281/zenodo.19683169)

The arXiv preprint and Zenodo archive correspond to an earlier public version of the project. An updated manuscript and artifact package that are compatible with the current repository are coming soon.
