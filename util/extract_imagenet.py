
import os
import tarfile
import shutil
import urllib.request
import sys

TRAIN_SRC_DIR = '/workspace/FATBETA/data/ILSVRC2012_img_train.tar'
TRAIN_DEST_DIR = '/workspace/FATBETA/data/imagenet/train'
VAL_SRC_DIR = '/workspace/FATBETA/data/ILSVRC2012_img_val.tar'
VAL_DEST_DIR = '/workspace/FATBETA/data/imagenet/val'
DEV_KIT_DIR = '/workspace/FATBETA/data/ILSVRC2012_devkit_t12'

# URL for the correct validation label mapping (Image ID -> WNID)
LABEL_URL = 'https://raw.githubusercontent.com/tensorflow/models/master/research/slim/datasets/imagenet_2012_validation_synset_labels.txt'


def extract_train():
    if not os.path.exists(TRAIN_SRC_DIR):
        print(f"Train tar not found at {TRAIN_SRC_DIR}, skipping extraction.")
        return

    print(f"Extracting training data to {TRAIN_DEST_DIR}...")
    with open(TRAIN_SRC_DIR, 'rb') as f:
        tar = tarfile.open(fileobj=f, mode='r:')
        for i, item in enumerate(tar):
            cls_name = item.name.strip(".tar")
            # Create a reader for the member file without extracting it to disk first
            a = tar.extractfile(item)
            if a is None:
                continue
            b = tarfile.open(fileobj=a, mode="r:")
            e_path = "{}/{}/".format(TRAIN_DEST_DIR, cls_name)
            if not os.path.isdir(e_path):
                os.makedirs(e_path)
            if i % 100 == 0:
                print("#", i, "extract train dateset to >>>", e_path)
            names = b.getnames()
            for name in names:
                b.extract(name, e_path)


def extract_val():
    if not os.path.exists(VAL_SRC_DIR):
        print(f"Val tar not found at {VAL_SRC_DIR}, skipping extraction.")
        return
        
    print(f"Extracting validation data to {VAL_DEST_DIR}...")
    with open(VAL_SRC_DIR, 'rb') as f:
        tar = tarfile.open(fileobj=f, mode='r:')
        if not os.path.isdir(VAL_DEST_DIR):
            os.makedirs(VAL_DEST_DIR)
        
        names = tar.getnames()
        for name in names:
            tar.extract(name, VAL_DEST_DIR)
    print("Validation extraction complete.")



def process_val():
    """
    Moves validation images into subfolders based on the official ImageNet mapping.
    Fixes the issue where using simple sorting leads to incorrect labels.
    """
    print("Processing validation data...")
    
    # 1. Ensure validation directory exists
    if not os.path.exists(VAL_DEST_DIR):
        print(f"Validation directory {VAL_DEST_DIR} does not exist. Please run extract_val() first.")
        return

    # 2. Get the label mapping file
    # We prefer the one from tensorflow/models which maps directly to WNIDs
    label_file = os.path.join(os.path.dirname(VAL_DEST_DIR), 'imagenet_2012_validation_synset_labels.txt')
    
    if not os.path.exists(label_file):
        print(f"Error: Label file not found at {label_file}")
        print("Please provide 'imagenet_2012_validation_synset_labels.txt' manually.")
        return
            
    with open(label_file, 'r') as f:
        labels = [l.strip() for l in f.readlines()]
        
    if len(labels) != 50000:
        print(f"Error: Expected 50,000 labels in {label_file}, found {len(labels)}")
        return

    print("Moving validation images to class folders...")
    count = 0
    
    # 3. Iterate and move
    for i, wnid in enumerate(labels):
        # Image filename is 1-based: ILSVRC2012_val_00000001.JPEG
        img_filename = f"ILSVRC2012_val_{i+1:08d}.JPEG"
        img_path = os.path.join(VAL_DEST_DIR, img_filename)
        
        target_dir = os.path.join(VAL_DEST_DIR, wnid)
        if not os.path.isdir(target_dir):
            os.makedirs(target_dir)
            
        target_path = os.path.join(target_dir, img_filename)
        
        # Move logic
        if os.path.exists(img_path):
            shutil.move(img_path, target_path)
            count += 1
        elif os.path.exists(target_path):
            # Already in place
            pass
        else:
            # Handle case where images might be in subfolders (e.g. from a previous failed run/sort)
            found = False
            # Optional: deep search if not found in root (slow, but robust)
            # For this script we'll keep it simple: warn if missing.
            # Assuming the user either has a fresh extract (images in root)
            # or a partially processed one.
            pass
            
    print(f"Organized {count} images into class folders.")
    
    # cleanup: remove any remaining empty folders or handle re-runs?
    # For now, this is sufficient.


if __name__ == '__main__':
    # Uncomment these if you need to re-extract from tar
    # extract_train() 
    # extract_val()
    
    # Always run processing logic
    process_val()
