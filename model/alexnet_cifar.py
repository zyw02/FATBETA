"""
AlexNet implementation for CIFAR datasets (32x32 input)
Adapted from original ImageNet AlexNet to work with CIFAR10/100
"""
import math
import torch
import torch.nn as nn
import torch.nn.init as init


class AlexNetCIFAR(nn.Module):
    """AlexNet for CIFAR datasets with 32x32 input"""
    
    def __init__(self, num_classes=10, **kwargs):
        super(AlexNetCIFAR, self).__init__()
        
        # For CIFAR-10/100 (32x32), we adapt the original ImageNet AlexNet
        # First conv layer: kernel_size=5, stride=1, padding=2 (instead of 11,4,2)
        # This keeps feature map size at 32x32
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=5, stride=1, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),  # 32 -> 16
            
            nn.Conv2d(64, 192, kernel_size=5, stride=1, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),  # 16 -> 8
            
            nn.Conv2d(192, 384, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            
            nn.Conv2d(384, 256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),  # 8 -> 4
        )
        
        # After 3 pooling layers: 32 -> 16 -> 8 -> 4
        # Feature map size: 256 * 4 * 4 = 4096
        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(256 * 4 * 4, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, num_classes),
        )
        
        self._initialize_weights()
    
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2.0 / n))
                if m.bias is not None:
                    init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.weight.data.normal_(0, 0.01)
                if m.bias is not None:
                    init.constant_(m.bias, 0)
    
    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x


def alexnet_cifar(num_classes=10, **kwargs):
    """AlexNet for CIFAR datasets"""
    return AlexNetCIFAR(num_classes=num_classes, **kwargs)



