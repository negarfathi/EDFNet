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
        _, _, H, W = target_feat.shape
        return encoder_feat[:, :, :H, :W]


    def forward(self, x):
        e1 = self.encoder1(x)
        e2 = self.encoder2(torch.nn.functional.max_pool2d(e1, 2))
        e3 = self.encoder3(torch.nn.functional.max_pool2d(e2, 2))
        e4 = self.encoder4(torch.nn.functional.max_pool2d(e3, 2))

        bottleneck = self.bottleneck(torch.nn.functional.max_pool2d(e4, 2))

        d4_up = torch.nn.functional.interpolate(bottleneck, scale_factor=2, mode="bilinear", align_corners=True)
        e4 = self.center_crop(e4, d4_up)
        d4 = self.decoder4(torch.cat([e4, d4_up], dim=1))

        d3_up = torch.nn.functional.interpolate(d4, scale_factor=2, mode="bilinear", align_corners=True)
        e3 = self.center_crop(e3, d3_up)
        d3 = self.decoder3(torch.cat([e3, d3_up], dim=1))

        d2_up = torch.nn.functional.interpolate(d3, scale_factor=2, mode="bilinear", align_corners=True)
        e2 = self.center_crop(e2, d2_up)
        d2 = self.decoder2(torch.cat([e2, d2_up], dim=1))

        d1_up = torch.nn.functional.interpolate(d2, scale_factor=2, mode="bilinear", align_corners=True)
        e1 = self.center_crop(e1, d1_up)
        d1 = self.decoder1(torch.cat([e1, d1_up], dim=1))

        return self.output_conv(d1)



class DeepLabV3ResNet50(torch.nn.Module):
    def __init__(self, num_classes, in_channels, pretrained=False):
        super().__init__()

        if pretrained:
            self.model = torchvision.models.segmentation.deeplabv3_resnet50(weights=DeepLabV3_ResNet50_Weights.DEFAULT)
        else:
            self.model = torchvision.models.segmentation.deeplabv3_resnet50(weights=None)

        original_conv = self.model.backbone.conv1
        self.model.backbone.conv1 = torch.nn.Conv2d(
            in_channels,
            original_conv.out_channels,
            kernel_size=7,
            stride=2,
            padding=3,
            bias=False
        )

        self.model.classifier[4] = torch.nn.Conv2d(256, num_classes, kernel_size=1)


    def forward(self, x):
        return self.model(x)["out"]



def build_model(model_name, num_classes, in_channels):
    model_name = model_name.lower()

    if model_name == "unet":
        return UNet(num_classes=num_classes, in_channels=in_channels)
    elif model_name == "unet_pretrained":
        return segmentation_models_pytorch.Unet(
            encoder_name="resnet34",
            encoder_weights="imagenet",
            in_channels=in_channels,
            classes=num_classes,
        )
    elif model_name == "deeplabv3":
        return DeepLabV3ResNet50(num_classes=num_classes, in_channels=in_channels, pretrained=False)
    elif model_name == "deeplabv3_pretrained":
        return DeepLabV3ResNet50(num_classes=num_classes, in_channels=in_channels, pretrained=True)
    else:
        raise ValueError(f"Unknown model architecture: {model_name}")