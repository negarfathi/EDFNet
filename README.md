# EDFNet

EDFNet is a modular early-fusion semantic segmentation framework for thin-obstacle perception in UAV navigation. It combines complementary RGB, depth, and edge cues at the input level so that the segmentation backbone can learn appearance, geometric, and boundary information jointly from the first convolutional layer onward. EDFNet is designed for challenging thin structures such as wires, poles, branches, and fences in cluttered aerial scenes.

EDFNet evaluates the following modality configurations:

* **RGB** – standard three-channel visual input.
* **RGBD** – RGB with an additional normalized depth channel.
* **RGBE** – RGB with an additional edge channel.
* **RGBDE** – RGB with both depth and edge channels.

The framework supports the following segmentation backbones:

* **U-Net**
* **U-Net (pretrained)**
* **DeepLabV3**
* **DeepLabV3 (pretrained)**

Together, these settings allow EDFNet to run controlled multimodal experiments across sixteen modality–backbone combinations on the [DDOS dataset](https://huggingface.co/datasets/benediktkol/DDOS). The implementation includes training, testing, metric reporting, and visualization utilities.

## Requirements

Before installing EDFNet, ensure that you have a Python environment with the packages listed in `requirements.txt`. The repository depends on the following libraries:

* **PyTorch**, **TorchVision**, **TorchAudio**
* **Albumentations**
* **OpenCV**
* **Matplotlib**
* **NumPy**
* **Scikit-learn**
* **tqdm**
* **datasets**
* **segmentation-models-pytorch**

To download the DDOS dataset, you also need Git LFS.

## Installation

Clone the repository:

```bash
git clone https://github.com/negarfathi/EDFNet.git
cd EDFNet
```

Install the required Python packages:

```bash
pip install -r requirements.txt
```

Clone the DDOS dataset with Git LFS:

```bash
git lfs install
git clone https://huggingface.co/datasets/benediktkol/DDOS ./data/DDOS
```

The main script automatically resolves the following dataset directories from the path passed to `--dataset`:

```text
<dataset_root>/data/train
<dataset_root>/data/validation
<dataset_root>/data/test
```

## Execution

EDFNet is executed through `main.py`, which supports training, testing, visualization, modality selection, backbone selection, device selection, edge extraction method, and training hyperparameters.

Run EDFNet using:

```bash
python main.py \
    --modality <rgb/rgbd/rgbe/rgbde/all> \
    --model <unet/unet_pretrained/deeplabv3/deeplabv3_pretrained/all> \
    --device <cpu/cuda> \
    --dataset <path/to/DDOS> \
    --train \
    --test \
    --visualize \
    --edge_method <canny/sobel> \
    --epochs <num_epochs> \
    --batch_size <batch_size> \
    --learning_rate <learning_rate>
```

Where:

* `--modality` selects the input modality configuration: `rgb`, `rgbd`, `rgbe`, `rgbde`, or `all`.
* `--model` selects the segmentation backbone: `unet`, `unet_pretrained`, `deeplabv3`, `deeplabv3_pretrained`, or `all`.
* `--device` selects the computation device: `cpu` or `cuda`.
* `--dataset` is the path to the DDOS dataset root.
* `--train` runs training.
* `--test` runs evaluation on the test split.
* `--visualize` saves qualitative prediction overlays.
* `--edge_method` selects the edge extraction operator: `canny` or `sobel`.
* `--epochs` specifies the number of training epochs.
* `--batch_size` specifies the batch size.
* `--learning_rate` specifies the optimizer learning rate.

### Example: Run all modality and model combinations on CUDA

```bash
python main.py --modality all --model all --device cuda --dataset ./data/DDOS --train --test --visualize --edge_method sobel --epochs 50 --batch_size 16 --learning_rate 5e-4
```

## Output

After execution, EDFNet produces the following outputs:

* **Model checkpoints** saved in:

```text
checkpoints/
```

* **Visualization results** saved in:

```text
visualizations/<model_name>_<modality>/
```

* **Console-reported evaluation metrics**, including:

  * mean IoU
  * per-class IoU
  * boundary IoU
  * recall
  * false positive rate
  * FPS
  * latency



## Papers

This repository is based on the following work:

N. Fathi, “EDFNet: Early Fusion of Edge and Depth for Thin-Obstacle Segmentation in UAV Navigation,” *arXiv* preprint, [arXiv:XXXX.XXXXX](https://arxiv.org/abs/XXXX.XXXXX), 2026.
