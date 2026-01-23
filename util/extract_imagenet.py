import os
import tarfile
import shutil

TRAIN_SRC_DIR = '/workspace/FATBETA/data/ILSVRC2012_img_train.tar'
TRAIN_DEST_DIR = '/workspace/FATBETA/data/imagenet/train'
VAL_SRC_DIR = '/workspace/FATBETA/data/ILSVRC2012_img_val.tar'
VAL_DEST_DIR = '/workspace/FATBETA/data/imagenet/val'
DEV_KIT_DIR = '/workspace/FATBETA/data/ILSVRC2012_devkit_t12'


def extract_train():
    if not os.path.exists(TRAIN_SRC_DIR):
        print(f"Train tar not found at {TRAIN_SRC_DIR}, skipping extraction.")
        return

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
                # Check if already extracted to avoid re-work if possible, or just overwrite
                # tarfile.extract will overwrite
                b.extract(name, e_path)


def extract_val():
    if not os.path.exists(VAL_SRC_DIR):
        print(f"Val tar not found at {VAL_SRC_DIR}, skipping extraction.")
        return
        
    with open(VAL_SRC_DIR, 'rb') as f:
        tar = tarfile.open(fileobj=f, mode='r:')
        if not os.path.isdir(VAL_DEST_DIR):
            os.makedirs(VAL_DEST_DIR)
        print("extract val dateset to >>>", VAL_DEST_DIR)
        names = tar.getnames()
        for name in names:
            tar.extract(name, VAL_DEST_DIR)


def process_val():
    print("Processing validation data...")
    ground_truth_file = os.path.join(DEV_KIT_DIR, 'data/ILSVRC2012_validation_ground_truth.txt')
    
    if not os.path.exists(ground_truth_file):
        print(f"Error: Ground truth file not found at {ground_truth_file}")
        return

    # Get sorted list of synsets from train directory
    if not os.path.exists(TRAIN_DEST_DIR):
         print(f"Error: Train directory {TRAIN_DEST_DIR} does not exist.")
         return

    synsets = sorted([d for d in os.listdir(TRAIN_DEST_DIR) if os.path.isdir(os.path.join(TRAIN_DEST_DIR, d))])
    
    if len(synsets) != 1000:
         print(f"Error: Expected 1000 synsets in {TRAIN_DEST_DIR}, found {len(synsets)}. Ensure training data is extracted.")
         return

    with open(ground_truth_file, 'r') as f:
        gt_ids = [int(line.strip()) for line in f.readlines()]
        
    if len(gt_ids) != 50000:
        print(f"Error: Expected 50000 ground truth entries, found {len(gt_ids)}")
        return

    print("Moving validation images to class folders...")
    count = 0
    for i, gt_id in enumerate(gt_ids):
        # Image index is 1-based in filename: ILSVRC2012_val_00000001.JPEG
        img_filename = f"ILSVRC2012_val_{i+1:08d}.JPEG"
        img_path = os.path.join(VAL_DEST_DIR, img_filename)
        
        # Ground truth ID is 1-based, list index is 0-based
        synset = synsets[gt_id - 1]
        
        target_dir = os.path.join(VAL_DEST_DIR, synset)
        if not os.path.isdir(target_dir):
            os.makedirs(target_dir)
            
        target_path = os.path.join(target_dir, img_filename)
        
        if os.path.exists(img_path):
            shutil.move(img_path, target_path)
            count += 1
        elif not os.path.exists(target_path):
            # If it's not at source and not at target, it's missing
             print(f"Warning: Image {img_filename} not found in {VAL_DEST_DIR}")
        
    print(f"Moved {count} images to {len(synsets)} class folders.")


if __name__ == '__main__':
    # extract_train() # User said this is already done
    # extract_val() # User said this is already done, and we verified files are there.
    # However, running them again might be safe if checking for existence, but process_val is the main goal.
    # I will comment them out or leave them if they are idempotent-ish. 
    # The original script blindly extracts. I should probably just run process_val since user asked "labels processing".
    # But to be robust, I will keep the functions available but only call process_val in this run, 
    # or make them check existence. 
    # I'll just call process_val() for now as that's the specific request.
    
    # Actually, to make the script reusable for the future, I will leave the calls but maybe comment them 
    # or rely on the user to uncomment if they need re-extraction. 
    # BUT, the user's prompt implies they just want to handle labels.
    # I'll modify the main block to only do process_val for this specific run, or make the extraction check.
    # I added checks in the functions above.
    
    # extract_train() 
    # extract_val()
    process_val()
