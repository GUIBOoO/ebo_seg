import copy
from typing import Sequence

import numpy as np
import torch
import torch.nn as nn
from torchvision.transforms import functional as TF

from transunet.vit_seg_modeling import CONFIGS as TRANSUNET_CONFIGS
from transunet.vit_seg_modeling import VisionTransformer as TransUNet


class DoubleConv(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class UNet(nn.Module):
    def __init__(
        self,
        in_channels: int = 3,
        out_channels: int = 1,
        features: Sequence[int] = (64, 128, 256, 512),
    ):
        super().__init__()
        self.downs = nn.ModuleList()
        self.ups = nn.ModuleList()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        channels = in_channels
        for feature in features:
            self.downs.append(DoubleConv(channels, feature))
            channels = feature

        self.bottleneck = DoubleConv(features[-1], features[-1] * 2)

        up_channels = features[-1] * 2
        for feature in reversed(features):
            self.ups.append(
                nn.ConvTranspose2d(up_channels, feature, kernel_size=2, stride=2)
            )
            self.ups.append(DoubleConv(feature * 2, feature))
            up_channels = feature

        self.head = nn.Conv2d(features[0], out_channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        skips = []
        for down in self.downs:
            x = down(x)
            skips.append(x)
            x = self.pool(x)

        x = self.bottleneck(x)
        skips = skips[::-1]

        for idx in range(0, len(self.ups), 2):
            x = self.ups[idx](x)
            skip = skips[idx // 2]

            if x.shape[-2:] != skip.shape[-2:]:
                x = TF.resize(x, list(skip.shape[-2:]), antialias=False)

            x = torch.cat([skip, x], dim=1)
            x = self.ups[idx + 1](x)

        return self.head(x)


def build_transunet(
    num_classes: int,
    img_size: int = 224,
    in_channels: int = 1,
    pretrained_path: str | None = None,
    vit_patch_size: int = 16,
) -> nn.Module:
    """R50-ViT-B_16 TransUNet (Chen et al.), see transunet/vit_seg_modeling.py.

    `img_size` must be a multiple of `vit_patch_size`; the ResNet-50 stem
    downsamples by 16x before the ViT patch grid, so img_size=224 reproduces
    the original paper's 14x14 (196) token grid exactly. Other multiples of
    16 also work: mismatched position embeddings are bicubic-resized inside
    `load_from`.
    """
    if in_channels not in (1, 3):
        raise ValueError(
            "TransUNet (vendored R50-ViT-B_16) only supports 1-channel "
            f"(auto-replicated to 3) or native 3-channel input, got in_channels={in_channels}."
        )

    config = copy.deepcopy(TRANSUNET_CONFIGS["R50-ViT-B_16"])
    config.n_classes = 1 if num_classes == 1 else num_classes
    config.n_skip = 3
    config.patches.grid = (img_size // vit_patch_size, img_size // vit_patch_size)

    model = TransUNet(config, img_size=img_size, num_classes=config.n_classes)
    if pretrained_path:
        model.load_from(weights=np.load(pretrained_path))
    return model


def build_model_acdc(
    model_name: str,
    num_classes: int,
    img_size: int = 256,
    pretrained_path: str | None = None,
) -> nn.Module:
    model_name = model_name.lower()
    out_channels = 1 if num_classes == 1 else num_classes

    if model_name == "unet":
        return UNet(in_channels=1, out_channels=out_channels)
    if model_name == "transunet":
        return build_transunet(num_classes, img_size=img_size, in_channels=1, pretrained_path=pretrained_path)

    raise ValueError(f"Unsupported model: {model_name}")


def build_model_brats(
    model_name: str,
    num_classes: int,
    img_size: int = 256,
    pretrained_path: str | None = None,
    in_channels: int = 4,
) -> nn.Module:
    """BraTS models. `in_channels` is 4 for the raw dataset (extract_data.py) or
    3 for the T1ce/T2/FLAIR dataset (extract_data_brats3.py), which is what
    TransUNet's ImageNet-pretrained 3-channel stem requires.
    """
    model_name = model_name.lower()
    out_channels = 1 if num_classes == 1 else num_classes

    if model_name == "unet":
        return UNet(in_channels=in_channels, out_channels=out_channels)
    if model_name == "transunet":
        if in_channels != 3:
            raise ValueError(
                "TransUNet (vendored R50-ViT-B_16) expects 3-channel input on BraTS, got "
                f"in_channels={in_channels}. Build the 3-channel dataset with "
                "extract_data_brats3.py and pass --in-channels 3."
            )
        return build_transunet(
            num_classes, img_size=img_size, in_channels=3, pretrained_path=pretrained_path
        )

    raise ValueError(f"Unsupported model: {model_name}")
