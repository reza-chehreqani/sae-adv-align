"""WideResNet, the architecture used by TRADES, AWP, DM-AT and most
RobustBench leaderboard entries. Slower than PreActResNet18 but higher
capacity -- use once the small/fast ablations on PreActResNet18 have picked
a winning recipe and you want to chase bigger numbers.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class WideBasic(nn.Module):
    def __init__(self, in_planes, planes, stride=1, drop_rate=0.0):
        super().__init__()
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.drop_rate = drop_rate

        self.shortcut = None
        if stride != 1 or in_planes != planes:
            self.shortcut = nn.Conv2d(in_planes, planes, kernel_size=1, stride=stride, bias=False)

    def forward(self, x):
        out = F.relu(self.bn1(x))
        shortcut = self.shortcut(out) if self.shortcut is not None else x
        out = self.conv1(out)
        if self.drop_rate > 0:
            out = F.dropout(out, p=self.drop_rate, training=self.training)
        out = self.conv2(F.relu(self.bn2(out)))
        return out + shortcut


class WideResNet(nn.Module):
    def __init__(self, depth=28, widen_factor=10, num_classes=10, drop_rate=0.0):
        super().__init__()
        assert (depth - 4) % 6 == 0, "WideResNet depth must be 6n+4"
        n = (depth - 4) // 6
        widths = [16, 16 * widen_factor, 32 * widen_factor, 64 * widen_factor]

        self.conv1 = nn.Conv2d(3, widths[0], kernel_size=3, padding=1, bias=False)
        self.in_planes = widths[0]
        self.layer1 = self._make_layer(widths[1], n, stride=1, drop_rate=drop_rate)
        self.layer2 = self._make_layer(widths[2], n, stride=2, drop_rate=drop_rate)
        self.layer3 = self._make_layer(widths[3], n, stride=2, drop_rate=drop_rate)
        self.bn_final = nn.BatchNorm2d(widths[3])
        self.linear = nn.Linear(widths[3], num_classes)

        self.layer_widths = {"layer1": widths[1], "layer2": widths[2], "layer3": widths[3]}
        self.default_feature_layer = "layer2"

    def _make_layer(self, planes, num_blocks, stride, drop_rate):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for s in strides:
            layers.append(WideBasic(self.in_planes, planes, s, drop_rate))
            self.in_planes = planes
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.conv1(x)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = F.relu(self.bn_final(out))
        out = F.adaptive_avg_pool2d(out, 1).flatten(1)
        return self.linear(out)


def WideResNet28_10(num_classes=10):
    return WideResNet(depth=28, widen_factor=10, num_classes=num_classes)


def WideResNet34_10(num_classes=10):
    return WideResNet(depth=34, widen_factor=10, num_classes=num_classes)
