import torch
import torch.nn as nn
from typing import List, Tuple, Any
from src.models import YOLO_CONFIG


class CNNBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        **kwargs,
    ):
        super(CNNBlock, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, bias=False, **kwargs)
        self.batch_norm = nn.BatchNorm2d(out_channels)
        self.leaky_relu = nn.LeakyReLU(0.1)

    def forward(self, x):
        return self.leaky_relu(self.batch_norm(self.conv(x)))


class YoloV1(nn.Module):
    def __init__(
        self,
        in_channels: int = 3,
        split_size: int = 7,
        num_boxes: int = 2,
        num_classes: int = 80,
        architecture: List = YOLO_CONFIG,
    ):
        super(YoloV1, self).__init__()
        self.architecture, self.in_channels = architecture, in_channels
        self.darknet = self._generate_layers(self.architecture, self.in_channels)

        S, B, C = split_size, num_boxes, num_classes
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(1024 * S * S, 496),
            nn.Dropout(0.0),
            nn.LeakyReLU(0.1),
            nn.Linear(496, S * S * (C + B * 5)),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc(torch.flatten(self.darknet(x), start_dim=1))

    def _generate_layers(
        self, architecture: List, in_channels: int = 3
    ) -> nn.Sequential:
        layers = []
        for block in architecture:
            block_parsed, in_channels = self.parse_block(in_channels, block)
            if isinstance(block_parsed, list):
                layers.extend(block_parsed)
            else:
                layers.append(block_parsed)
        return nn.Sequential(*layers)

    @staticmethod
    def parse_block(in_channels: int, block: List | Tuple | str) -> Tuple[Any, int]:
        match block:
            case tuple((kernel_size, out_channels, stride, padding)):
                return CNNBlock(
                    in_channels,
                    out_channels,
                    kernel_size=kernel_size,
                    stride=stride,
                    padding=padding,
                ), out_channels
            case list((conv1, conv2, repeats)):
                for _ in range(repeats):
                    layers = []
                    for conv in [conv1, conv2]:
                        kernel_size, out_channels, stride, padding = conv
                        layers.append(
                            CNNBlock(
                                in_channels,
                                out_channels,
                                kernel_size=kernel_size,
                                stride=stride,
                                padding=padding,
                            )
                        )
                        in_channels = out_channels
                    return layers, out_channels
            case str(_):
                return nn.MaxPool2d(kernel_size=2, stride=2), in_channels


if __name__ == "__main__":
    yolo = YoloV1()
    print(yolo)
