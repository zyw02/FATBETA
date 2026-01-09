"""
ResNet implementation for CIFAR datasets (32x32 input)
- Supports the official ResNet-CIFAR family (3 stages: ResNet20, 32, 44, 56, 110)
- Supports the standard ResNet family adapted for CIFAR (4 stages: ResNet18, 34, 50)
"""
import torch
import torch.nn as nn
import torch.nn.init as init
from torchvision.models.resnet import BasicBlock, Bottleneck


def _weights_init(m):
    if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d):
        init.kaiming_normal_(m.weight)


class ResNet_CIFAR_Official(nn.Module):
    """
    Official ResNet for CIFAR (3 stages)
    Follows depth = 6n + 2 formula.
    Widths: 16, 32, 64
    """
    def __init__(self, block, num_layers, num_classes=10):
        super(ResNet_CIFAR_Official, self).__init__()
        self.in_planes = 16

        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(16)
        self.relu = nn.ReLU(inplace=True)
        
        self.layer1 = self._make_layer(block, 16, num_layers, stride=1)
        self.layer2 = self._make_layer(block, 32, num_layers, stride=2)
        self.layer3 = self._make_layer(block, 64, num_layers, stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(64, num_classes)

        self.apply(_weights_init)

    def _make_layer(self, block, planes, num_layers, stride):
        downsample = None
        if stride != 1 or self.in_planes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.in_planes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.in_planes, planes, stride, downsample))
        self.in_planes = planes * block.expansion
        for _ in range(1, num_layers):
            layers.append(block(self.in_planes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.avgpool(out)
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        return out


class ResNet_Std_CIFAR(nn.Module):
    """
    Standard ResNet adapted for CIFAR (4 stages)
    Modified first layer: 3x3 conv, stride 1, no maxpool.
    Widths: 64, 128, 256, 512
    """
    def __init__(self, block, num_blocks, num_classes=100):
        super(ResNet_Std_CIFAR, self).__init__()
        self.in_planes = 64

        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        self.apply(_weights_init)

    def _make_layer(self, block, planes, num_blocks, stride):
        downsample = None
        if stride != 1 or self.in_planes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.in_planes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.in_planes, planes, stride, downsample))
        self.in_planes = planes * block.expansion
        for _ in range(1, num_blocks):
            layers.append(block(self.in_planes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.avgpool(out)
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        return out


def resnet_cifar(depth, num_classes=10, dataset='cifar10'):
    """
    Flexible factory function for ResNet CIFAR models.
    Identifies the model family based on depth:
    - Depths 20, 32, 44, 56, 110 -> Official CIFAR ResNet (3 stages, 16-32-64 width)
    - Depths 18, 34, 50, 101, 152 -> Standard ResNet adapted for CIFAR (4 stages, 64-128-256-512 width)
    """
    official_depths = [20, 32, 44, 56, 110]
    standard_depths = [18, 34, 50, 101, 152]

    if depth in official_depths:
        n = (depth - 2) // 6
        print(f"[Model] Creating Official ResNet-{depth} for {dataset} (3 stages, n={n})")
        return ResNet_CIFAR_Official(BasicBlock, n, num_classes=num_classes)
    
    elif depth in standard_depths:
        print(f"[Model] Creating Standard ResNet-{depth} adapted for {dataset} (4 stages)")
        if depth == 18:
            return ResNet_Std_CIFAR(BasicBlock, [2, 2, 2, 2], num_classes=num_classes)
        elif depth == 34:
            return ResNet_Std_CIFAR(BasicBlock, [3, 4, 6, 3], num_classes=num_classes)
        elif depth == 50:
            return ResNet_Std_CIFAR(Bottleneck, [3, 4, 6, 3], num_classes=num_classes)
        elif depth == 101:
            return ResNet_Std_CIFAR(Bottleneck, [3, 4, 23, 3], num_classes=num_classes)
        elif depth == 152:
            return ResNet_Std_CIFAR(Bottleneck, [3, 8, 36, 3], num_classes=num_classes)
    
    else:
        # Fallback: if it's not a common depth, check if it fits the 6n+2 formula
        if (depth - 2) % 6 == 0:
            n = (depth - 2) // 6
            print(f"[Model] Unknown depth {depth}, but fits 6n+2. Creating Official ResNet (3 stages, n={n})")
            return ResNet_CIFAR_Official(BasicBlock, n, num_classes=num_classes)
        else:
            raise ValueError(f"Unsupported ResNet depth for CIFAR: {depth}. "
                             f"Common depths: {official_depths} (3-stage) or {standard_depths} (4-stage)")

def resnet18_cifar(num_classes=10, **kwargs):
    return ResNet_Std_CIFAR(BasicBlock, [2, 2, 2, 2], num_classes=num_classes)