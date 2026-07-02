import torch
import torch.nn
import torch.nn.functional
import segmentation_models_pytorch
import torchvision.models.segmentation
from torchvision.models.segmentation import DeepLabV3_ResNet50_Weights


class UNet(torch.nn.Module):
    def __init__(self, num_classes, in_channels, base_channels=64):
        super().__init__()

        self.encoder1 = self.encoder_block(in_channels, base_channels)
        self.encoder2 = self.encoder_block(base_channels, base_channels * 2)
        self.encoder3 = self.encoder_block(base_channels * 2, base_channels * 4)
        self.encoder4 = self.encoder_block(base_channels * 4, base_channels * 8)
        self.bottleneck = self.encoder_block(base_channels * 8, base_channels * 16)

        self.decoder4 = self.decoder_block(base_channels * 16 + base_channels * 8, base_channels * 8)
        self.decoder3 = self.decoder_block(base_channels * 8 + base_channels * 4, base_channels * 4)
        self.decoder2 = self.decoder_block(base_channels * 4 + base_channels * 2, base_channels * 2)
        self.decoder1 = self.decoder_block(base_channels * 2 + base_channels, base_channels)
        self.output_conv = torch.nn.Conv2d(base_channels, num_classes, kernel_size=1)

    def encoder_block(self, in_channels, out_channels, kernel_size=3):
        return torch.nn.Sequential(
            torch.nn.Conv2d(in_channels, out_channels, kernel_size, padding=1),
            torch.nn.BatchNorm2d(out_channels),
            torch.nn.ReLU(inplace=True),
            torch.nn.Conv2d(out_channels, out_channels, kernel_size, padding=1),
            torch.nn.BatchNorm2d(out_channels),
            torch.nn.ReLU(inplace=True)
        )

    def decoder_block(self, in_channels, out_channels, kernel_size=3):
        return torch.nn.Sequential(
            torch.nn.Conv2d(in_channels, out_channels, kernel_size, padding=1),
            torch.nn.BatchNorm2d(out_channels),
            torch.nn.ReLU(inplace=True),
            torch.nn.Conv2d(out_channels, out_channels, kernel_size, padding=1),
            torch.nn.BatchNorm2d(out_channels),
            torch.nn.ReLU(inplace=True)
        )

    def center_crop(self, encoder_feat, target_feat):
        _, _, h, w = target_feat.shape
        return encoder_feat[:, :, :h, :w]

    def forward(self, x):
        e1 = self.encoder1(x)
        e2 = self.encoder2(torch.nn.functional.max_pool2d(e1, 2))
        e3 = self.encoder3(torch.nn.functional.max_pool2d(e2, 2))
        e4 = self.encoder4(torch.nn.functional.max_pool2d(e3, 2))
        bottleneck = self.bottleneck(torch.nn.functional.max_pool2d(e4, 2))

        d4_up = torch.nn.functional.interpolate(bottleneck, scale_factor=2, mode="bilinear", align_corners=True)
        d4 = self.decoder4(torch.cat([self.center_crop(e4, d4_up), d4_up], dim=1))

        d3_up = torch.nn.functional.interpolate(d4, scale_factor=2, mode="bilinear", align_corners=True)
        d3 = self.decoder3(torch.cat([self.center_crop(e3, d3_up), d3_up], dim=1))

        d2_up = torch.nn.functional.interpolate(d3, scale_factor=2, mode="bilinear", align_corners=True)
        d2 = self.decoder2(torch.cat([self.center_crop(e2, d2_up), d2_up], dim=1))

        d1_up = torch.nn.functional.interpolate(d2, scale_factor=2, mode="bilinear", align_corners=True)
        d1 = self.decoder1(torch.cat([self.center_crop(e1, d1_up), d1_up], dim=1))

        return self.output_conv(d1)


def _deeplabv3_resnet50_no_pretraining():
    """Create a fully non-pretrained DeepLabV3-ResNet50 across torchvision versions."""
    try:
        return torchvision.models.segmentation.deeplabv3_resnet50(weights=None, weights_backbone=None)
    except TypeError:
        # Older torchvision used pretrained_backbone.
        return torchvision.models.segmentation.deeplabv3_resnet50(
            weights=None,
            pretrained_backbone=False,
        )


def _adapt_first_conv(conv, in_channels, keep_pretrained_rgb=True):
    if conv.in_channels == in_channels:
        return conv

    new_conv = torch.nn.Conv2d(
        in_channels,
        conv.out_channels,
        kernel_size=conv.kernel_size,
        stride=conv.stride,
        padding=conv.padding,
        dilation=conv.dilation,
        groups=conv.groups,
        bias=(conv.bias is not None),
        padding_mode=conv.padding_mode,
    )

    if keep_pretrained_rgb and conv.in_channels == 3 and in_channels >= 3:
        with torch.no_grad():
            new_conv.weight[:, :3, :, :] = conv.weight
            if in_channels > 3:
                extra = conv.weight.mean(dim=1, keepdim=True)
                for c in range(3, in_channels):
                    new_conv.weight[:, c:c + 1, :, :] = extra
            if conv.bias is not None and new_conv.bias is not None:
                new_conv.bias.copy_(conv.bias)

    return new_conv


class DeepLabV3ResNet50(torch.nn.Module):
    def __init__(self, num_classes, in_channels, pretrained=False):
        super().__init__()
        if pretrained:
            self.model = torchvision.models.segmentation.deeplabv3_resnet50(
                weights=DeepLabV3_ResNet50_Weights.DEFAULT
            )
        else:
            self.model = _deeplabv3_resnet50_no_pretraining()

        original_conv = self.model.backbone.conv1
        self.model.backbone.conv1 = _adapt_first_conv(
            original_conv,
            in_channels=in_channels,
            keep_pretrained_rgb=pretrained,
        )

        self.model.classifier[4] = torch.nn.Conv2d(256, num_classes, kernel_size=1)
        if getattr(self.model, "aux_classifier", None) is not None:
            try:
                self.model.aux_classifier[4] = torch.nn.Conv2d(256, num_classes, kernel_size=1)
            except Exception:
                pass

    def forward(self, x):
        return self.model(x)["out"]



def _get_segformer_class():
    """Return the SegFormer class across segmentation_models_pytorch naming variants."""
    if hasattr(segmentation_models_pytorch, "Segformer"):
        return segmentation_models_pytorch.Segformer
    if hasattr(segmentation_models_pytorch, "SegFormer"):
        return segmentation_models_pytorch.SegFormer
    return None


def validate_model_availability(model_names):
    """Fail early before long benchmark runs if a requested optional model is unavailable."""
    requested = {m.lower() for m in model_names}
    if ({"segformer", "segformer_pretrained"} & requested) and _get_segformer_class() is None:
        version = getattr(segmentation_models_pytorch, "__version__", "unknown")
        raise RuntimeError(
            "SegFormer was requested, but this segmentation_models_pytorch installation "
            f"does not expose Segformer/SegFormer (version={version}). "
            "Install a recent version with `pip install -U segmentation-models-pytorch timm`, "
            "then rerun. The fixed benchmark intentionally includes SegFormer in --model all."
        )


def build_model(model_name, num_classes, in_channels):
    model_name = model_name.lower()

    if model_name == "unet":
        return UNet(num_classes=num_classes, in_channels=in_channels)

    if model_name == "unet_pretrained":
        return segmentation_models_pytorch.Unet(
            encoder_name="resnet34",
            encoder_weights="imagenet",
            in_channels=in_channels,
            classes=num_classes,
        )

    if model_name == "deeplabv3":
        return DeepLabV3ResNet50(num_classes=num_classes, in_channels=in_channels, pretrained=False)

    if model_name == "deeplabv3_pretrained":
        return DeepLabV3ResNet50(num_classes=num_classes, in_channels=in_channels, pretrained=True)

    if model_name == "deeplabv3plus":
        return segmentation_models_pytorch.DeepLabV3Plus(
            encoder_name="resnet34",
            encoder_weights=None,
            in_channels=in_channels,
            classes=num_classes,
        )

    if model_name == "deeplabv3plus_pretrained":
        return segmentation_models_pytorch.DeepLabV3Plus(
            encoder_name="resnet34",
            encoder_weights="imagenet",
            in_channels=in_channels,
            classes=num_classes,
        )

    # Modern transformer baseline. Both scratch and ImageNet-pretrained variants are
    # included so SegFormer is symmetric with UNet/DeepLabV3/DeepLabV3Plus.
    if model_name in ["segformer", "segformer_pretrained"]:
        segformer_class = _get_segformer_class()
        if segformer_class is None:
            version = getattr(segmentation_models_pytorch, "__version__", "unknown")
            raise RuntimeError(
                f"{model_name} was requested, but segmentation_models_pytorch does not provide "
                f"Segformer/SegFormer (version={version}). Run `pip install -U segmentation-models-pytorch timm`."
            )
        return segformer_class(
            encoder_name="mit_b0",
            encoder_weights="imagenet" if model_name == "segformer_pretrained" else None,
            in_channels=in_channels,
            classes=num_classes,
        )

    raise ValueError(f"Unknown model architecture: {model_name}")
