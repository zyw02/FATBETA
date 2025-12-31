import logging
from .mobilenetv2 import mobilenet_v2
from .resnet_cifar import resnet18_cifar
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
        if arch == 'resnet18':
            # Use dedicated ResNet18 for CIFAR (properly designed for 32x32 input)
            model = resnet18_cifar(num_classes=num_classes)
            if pre_trained:
                logger.warning('Pre-trained weights for CIFAR ResNet18 are not available, using random initialization')
        elif arch == 'mobilenetv2':
            model = mobilenet_v2(pretrained=False, num_classes=num_classes)
            if pre_trained:
                logger.warning('Pre-trained weights for CIFAR MobileNetV2 are not available, using random initialization')
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
