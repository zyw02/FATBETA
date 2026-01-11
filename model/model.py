import logging
from .mobilenetv2 import mobilenet_v2
from .mobilenetv2_cifar import mobilenet_v2_cifar
from .mobilenetv2_c10 import mobilenet_v2 as mobilenet_v2_c10
from .resnet_cifar import resnet_cifar
from .alexnet_cifar import alexnet_cifar
import timm
import torch
import torchvision
from torchvision.models import resnet101, resnet18

def create_model(arch, dataset='imagenet', pre_trained=True):
    logger = logging.getLogger()

    model = None
    num_classes = 1000  # ImageNet default
    if dataset == 'cifar10':
        num_classes = 10
    elif dataset == 'cifar100':
        num_classes = 100
    
    if dataset == 'imagenet':
        if arch == 'resnet18':
            model = timm.create_model('gluon_resnet18_v1b', pretrained=True)
        elif arch == 'mobilenetv2':
            model = mobilenet_v2(pretrained=True)
        elif arch == 'resnet101':
            model = resnet101(torchvision.models.ResNet101_Weights)
        elif arch == 'efficientnet_lite':
            model = timm.create_model('efficientnet_lite0', pretrained=True)
    elif dataset in ['cifar10', 'cifar100']:
        if arch.startswith('resnet'):
            # Parse depth from arch name (e.g., resnet20 -> 20)
            try:
                depth = int(arch.replace('resnet', ''))
            except ValueError:
                depth = 18 # default
            
            model = resnet_cifar(depth=depth, num_classes=num_classes, dataset=dataset)
            if pre_trained:
                logger.warning(f'Pre-trained weights for {arch} ({dataset}) are not available, using random initialization')
        
        elif arch == 'mobilenetv2':
            # Use the new mobilenetv2_c10 for CIFAR
            model = mobilenet_v2_c10(pretrained=False, num_classes=num_classes, input_size=32)
            if pre_trained:
                logger.warning('Pre-trained weights for CIFAR MobileNetV2 (c10) are not available, using random initialization')
        elif arch == 'mobilenet_v2_cifar':
            # Use the new mobilenetv2_c10 for CIFAR version as well
            model = mobilenet_v2_c10(pretrained=False, num_classes=num_classes, input_size=32)
            if pre_trained:
                logger.warning('Pre-trained weights for CIFAR MobileNetV2 (c10) are not available, using random initialization')
        elif arch == 'alexnet':
            # Use dedicated AlexNet for CIFAR (properly designed for 32x32 input)
            model = alexnet_cifar(num_classes=num_classes)
            if pre_trained:
                logger.warning('Pre-trained weights for CIFAR AlexNet are not available, using random initialization')
        else:
            logger.error('Model architecture `%s` for `%s` dataset is not supported' % (arch, dataset))
            exit(-1)

    if model is None:
        logger.error('Model architecture `%s` for `%s` dataset is not supported' % (arch, dataset))
        exit(-1)

    msg = 'Created `%s` model for `%s` dataset (num_classes=%d)' % (arch, dataset, num_classes)
    msg += '\n          Use pre-trained model = %s' % pre_trained
    logger.info(msg)

    return model
