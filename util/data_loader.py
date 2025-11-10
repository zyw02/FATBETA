import os

import numpy as np
import torch
import torch.utils.data
import torchvision as tv
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torchvision.datasets.folder import ImageFolder, default_loader


def init_dataloader(cfg, arch):
    normalize_imagenet = tv.transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                  std=[0.229, 0.224, 0.225])
    normalize_cifar = tv.transforms.Normalize(mean=[0.4914, 0.4822, 0.4465],
                                               std=[0.2023, 0.1994, 0.2010])
    
    if cfg.dataset == 'imagenet':
        traindir = os.path.join(cfg.path, 'train')
        testdir = os.path.join(cfg.path, 'val')
        valdir = os.path.join(cfg.path, 'subImageNet') 

        if not os.path.exists(valdir): #using training set for searching?
            valdir = traindir

        train_set = datasets.ImageFolder(traindir, transforms.Compose([
                transforms.RandomResizedCrop(224),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                normalize_imagenet,
        ]))

        test_set = datasets.ImageFolder(testdir, transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            normalize_imagenet,
        ]))

        val_set = datasets.ImageFolder(valdir, transforms.Compose([
                transforms.RandomResizedCrop(224),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                normalize_imagenet,
        ]))
    elif cfg.dataset == 'cifar10':
        # download=False if tar.gz already exists, True otherwise
        import os
        tar_path = os.path.join(cfg.path, 'cifar-10-python.tar.gz')
        need_download = not os.path.exists(tar_path)
        train_set = datasets.CIFAR10(
            root=cfg.path, train=True, download=need_download,
            transform=transforms.Compose([
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                normalize_cifar,
            ]))
        test_set = datasets.CIFAR10(
            root=cfg.path, train=False, download=need_download,
            transform=transforms.Compose([
                transforms.ToTensor(),
                normalize_cifar,
            ]))
        val_set = datasets.CIFAR10(
            root=cfg.path, train=True, download=need_download,
            transform=transforms.Compose([
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                normalize_cifar,
            ]))
    elif cfg.dataset == 'cifar100':
        # download=False if tar.gz already exists, True otherwise
        import os
        tar_path = os.path.join(cfg.path, 'cifar-100-python.tar.gz')
        need_download = not os.path.exists(tar_path)
        train_set = datasets.CIFAR100(
            root=cfg.path, train=True, download=need_download,
            transform=transforms.Compose([
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                normalize_cifar,
            ]))
        test_set = datasets.CIFAR100(
            root=cfg.path, train=False, download=need_download,
            transform=transforms.Compose([
                transforms.ToTensor(),
                normalize_cifar,
            ]))
        val_set = datasets.CIFAR100(
            root=cfg.path, train=True, download=need_download,
            transform=transforms.Compose([
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                normalize_cifar,
            ]))
    else:
        raise NotImplementedError(f"Dataset {cfg.dataset} is not supported")
    
    # Only use DistributedSampler if distributed training is enabled
    if 'WORLD_SIZE' in os.environ:
        train_sampler = torch.utils.data.distributed.DistributedSampler(train_set)
        val_sampler = torch.utils.data.distributed.DistributedSampler(val_set)
        test_sampler = torch.utils.data.distributed.DistributedSampler(test_set)
    else:
        train_sampler = None
        val_sampler = None
        test_sampler = None

    print(f"[DEBUG] Creating train_loader: batch_size={cfg.batch_size}, workers={cfg.workers}, sampler={train_sampler}")
    train_loader = torch.utils.data.DataLoader(
        train_set, batch_size=cfg.batch_size, shuffle=(train_sampler is None), drop_last=True,
        num_workers=cfg.workers, pin_memory=True, sampler=train_sampler)
    print(f"[DEBUG] Train loader created, length: {len(train_loader)}")
    
    test_loader = torch.utils.data.DataLoader(
        test_set,
        batch_size=cfg.batch_size, shuffle=False, sampler=test_sampler, drop_last=False,
        num_workers=cfg.workers, pin_memory=True)
    print(f"[DEBUG] Test loader created, length: {len(test_loader)}")
    
    val_loader = torch.utils.data.DataLoader(
        val_set, batch_size=cfg.batch_size, shuffle=(val_sampler is None),drop_last=True,
        num_workers=cfg.workers, pin_memory=True, sampler=val_sampler)
    print(f"[DEBUG] Val loader created, length: {len(val_loader)}")
    
    return train_loader, val_loader, test_loader, train_sampler, val_sampler