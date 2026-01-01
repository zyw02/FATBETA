import os

import numpy as np
import torch
import torch.utils.data
import torchvision as tv
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torchvision.datasets.folder import ImageFolder, default_loader
import tarfile
import torch.distributed as dist


def init_dataloader(cfg, arch):
    normalize_imagenet = tv.transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                  std=[0.229, 0.224, 0.225])
    normalize_cifar = tv.transforms.Normalize(mean=[0.4914, 0.4822, 0.4465],
                                               std=[0.2023, 0.1994, 0.2010])
    
    def _ensure_cifar_ready(root: str, kind: str):
        if kind == 'cifar10':
            extracted_dir = os.path.join(root, 'cifar-10-batches-py')
            tar_path = os.path.join(root, 'cifar-10-python.tar.gz')
        else:
            extracted_dir = os.path.join(root, 'cifar-100-python')
            tar_path = os.path.join(root, 'cifar-100-python.tar.gz')
        if os.path.isdir(extracted_dir):
            return
        is_dist = dist.is_available() and dist.is_initialized()
        rank = dist.get_rank() if is_dist else 0
        if rank == 0 and os.path.exists(tar_path):
            with tarfile.open(tar_path, 'r:gz') as tar:
                tar.extractall(path=root)
        if is_dist:
            dist.barrier()

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
        extracted_dir = os.path.join(cfg.path, 'cifar-10-batches-py')
        _ensure_cifar_ready(cfg.path, 'cifar10')
        need_download = not os.path.isdir(extracted_dir)
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
        extracted_dir = os.path.join(cfg.path, 'cifar-100-python')
        _ensure_cifar_ready(cfg.path, 'cifar100')
        need_download = not os.path.isdir(extracted_dir)
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
    
    # Use torch.distributed status to decide samplers
    is_dist = dist.is_available() and dist.is_initialized()
    world_size = dist.get_world_size() if is_dist else 1
    if is_dist and world_size > 1:
        train_sampler = torch.utils.data.distributed.DistributedSampler(train_set)
        val_sampler = torch.utils.data.distributed.DistributedSampler(val_set, shuffle=False)
        test_sampler = torch.utils.data.distributed.DistributedSampler(test_set, shuffle=False)
    else:
        train_sampler = None
        val_sampler = None
        test_sampler = None
    print(f"[DEBUG] Creating train_loader: batch_size={cfg.batch_size}, workers={cfg.workers}, sampler={train_sampler}")
    train_loader = torch.utils.data.DataLoader(
        train_set, batch_size=cfg.batch_size, shuffle=(train_sampler is None), drop_last=True,
        num_workers=cfg.workers, pin_memory=True, sampler=train_sampler,
        persistent_workers=bool(cfg.workers and cfg.workers > 0))
    print(f"[DEBUG] Train loader created, length: {len(train_loader)}")
    
    test_loader = torch.utils.data.DataLoader(
        test_set,
        batch_size=cfg.batch_size, shuffle=False, sampler=test_sampler, drop_last=False,
        num_workers=cfg.workers, pin_memory=True,
        persistent_workers=bool(cfg.workers and cfg.workers > 0))
    print(f"[DEBUG] Test loader created, length: {len(test_loader)}")
    
    val_loader = torch.utils.data.DataLoader(
        val_set, batch_size=cfg.batch_size, shuffle=(val_sampler is None),drop_last=True,
        num_workers=cfg.workers, pin_memory=True, sampler=val_sampler,
        persistent_workers=bool(cfg.workers and cfg.workers > 0))
    print(f"[DEBUG] Val loader created, length: {len(val_loader)}")
    
    return train_loader, val_loader, test_loader, train_sampler, val_sampler
