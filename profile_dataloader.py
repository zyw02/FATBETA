"""
Profile data loading overhead.
"""
import time
import torch
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import torchvision.datasets as datasets

# CIFAR-10 setup (same as training)
transform = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

dataset = datasets.CIFAR10(root='./data/cifar10', train=True, download=True, transform=transform)
loader = DataLoader(dataset, batch_size=128, shuffle=True, num_workers=16, pin_memory=True)

# Test data loading speed
print("Testing data loading speed...")
times = []

# Warmup
for i, (x, y) in enumerate(loader):
    x = x.cuda(non_blocking=True)
    y = y.cuda(non_blocking=True)
    if i >= 10:
        break

# Measure
start_epoch = time.time()
for i, (x, y) in enumerate(loader):
    batch_start = time.time()
    x = x.cuda(non_blocking=True)
    y = y.cuda(non_blocking=True)
    torch.cuda.synchronize()
    times.append(time.time() - batch_start)
    
epoch_time = time.time() - start_epoch

print(f"\nTotal batches: {len(times)}")
print(f"Epoch data loading time: {epoch_time:.2f} s")
print(f"Average per batch: {sum(times)/len(times)*1000:.2f} ms")
print(f"Max batch load time: {max(times)*1000:.2f} ms")
print(f"Min batch load time: {min(times)*1000:.2f} ms")

# Also test with workers=0
print("\n\nTesting with workers=0 (no multiprocessing)...")
loader_single = DataLoader(dataset, batch_size=128, shuffle=True, num_workers=0, pin_memory=True)

start = time.time()
for i, (x, y) in enumerate(loader_single):
    x = x.cuda(non_blocking=True)
    y = y.cuda(non_blocking=True)
    if i >= 50:
        break
torch.cuda.synchronize()
print(f"50 batches with workers=0: {time.time() - start:.2f} s ({(time.time()-start)/50*1000:.2f} ms/batch)")
