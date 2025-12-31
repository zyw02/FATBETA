import logging
import torch
import yaml
import os
import json
from pathlib import Path
from timm.loss import LabelSmoothingCrossEntropy
import torch.distributed as dist
from model import create_model
from util import (ProgressMonitor, TensorBoardMonitor, 
                  get_config, init_logger, set_global_seed, setup_print, load_checkpoint, save_checkpoint, preprocess_model, init_dataloader)
from util.utils import copy_code, create_optimizer_and_lr_scheduler
from util.model_ema import ModelEma
from util.dist import is_master
from quan import find_modules_to_quantize, replace_module_by_names

# Import sensitive analysis tools
from util.sensitive_features import collect_gradient_sensitivity, compute_activation_baseline
from util.sensitive_restorer import SensitiveActivationCollector
from util.sensitive_restorer_transformer import LayerwiseRestorer
from util.fault_injector import FaultInjector
import argparse
import sys
from timm.scheduler import create_scheduler
from util.config import SchedulerConfig

# We need the train function from process.py
from process import train as train_process

def main():
    """
    Main function for Stage 2: Training the Sensitive Channel Restorer.
    This script follows the structure of main.py for consistency.
    """
    parser = argparse.ArgumentParser(description="Stage 2 Sensitive Restorer Training")
    parser.add_argument("--config", required=True, type=str, help="Path to the Stage 2 YAML configuration file.")
    parser.add_argument("--stage1_ckpt", required=True, type=str, help="Path to the Stage 1 model checkpoint.")
    args = parser.parse_args()

    # --- 1. CONFIGURATION and INITIALIZATION ---
    # Load the config file exactly once. This object will be passed everywhere.
    script_dir = Path.cwd()
    # Temporarily modify sys.argv for get_config, as it expects positional args
    original_argv = sys.argv
    sys.argv = [original_argv[0], args.config]
    configs = get_config(default_file=script_dir / 'template.yaml')
    sys.argv = original_argv

    # Initialize logger, monitors, and set seed
    if is_master():
        output_dir = script_dir / configs.output_dir
        output_dir.mkdir(exist_ok=True)
        log_dir = init_logger(configs.name, output_dir, script_dir / 'logging.conf')
        logger = logging.getLogger()
        with open(log_dir / "configs.yaml", "w") as yaml_file:
            yaml.safe_dump(configs, yaml_file)
        pymonitor = ProgressMonitor(logger)
        tbmonitor = TensorBoardMonitor(logger, log_dir)
        monitors = [pymonitor, tbmonitor]
    else:
        logger, log_dir, monitors = None, None, None
    
    setup_print(is_master=is_master())
    set_global_seed(seed=getattr(configs, 'seed', 42))

    # --- 2. DATASET and STAGE 1 MODEL LOADING ---
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    train_loader, val_loader, _, _, _ = init_dataloader(configs.dataloader, arch=configs.arch)

    # Load the Stage 1 model weights *without* loading its optimizer state.
    logger.info("Loading Stage 1 checkpoint for analysis...")
    training_model = create_model(configs.arch, dataset=configs.dataloader.dataset, pre_trained=configs.pre_trained)
    training_model = preprocess_model(training_model, configs)
    training_model = replace_module_by_names(training_model, find_modules_to_quantize(training_model, configs))
    training_model.to(device)
    load_checkpoint(training_model, args.stage1_ckpt)
    logger.info("Stage 1 model loaded successfully.")

    # --- 3. SENSITIVITY ANALYSIS ---
    logger.info("Collecting gradient sensitivity...")
    sensitive_channels_path = os.path.join(log_dir, "sensitive_channels.pth")
    sensitive_channels = collect_gradient_sensitivity(
        model=training_model, data_loader=train_loader, criterion=torch.nn.CrossEntropyLoss().to(device),
        device=device, topk_per_layer=configs.sensitive_restorer.get("topk_per_layer", 8),
        max_batches=configs.sensitive_restorer.get("sensitivity_batches", 50), output_path=sensitive_channels_path,
    )
    if not sensitive_channels:
        raise RuntimeError("No sensitive channels identified for Stage2 training. Aborting.")
    logger.info(f"Identified {sum(len(v['indices']) for v in sensitive_channels.values())} sensitive channels.")

    logger.info("Computing activation baseline...")
    baseline_path = os.path.join(log_dir, "sensitive_baseline.pth")
    baseline_stats = compute_activation_baseline(
        model=training_model, data_loader=train_loader, sensitive_channels=sensitive_channels,
        device=device, max_batches=configs.sensitive_restorer.get("baseline_batches", 50), output_path=baseline_path,
    )
    logger.info("Activation baseline computed.")

    # --- 4. RESTORER INITIALIZATION ---
    # Dummy pass to get feature dimensions
    logger.info("Running dummy forward pass for Restorer initialization...")
    collector_for_dim = SensitiveActivationCollector(training_model, sensitive_channels, baseline_stats)
    collector_for_dim.register_hooks()
    dummy_input = torch.randn(2, 3, 32, 32).to(device)
    training_model(dummy_input)
    _, feature_dims_per_layer = collector_for_dim.build_layer_features(dummy_input)
    collector_for_dim.remove_hooks()
    logger.info(f"Determined feature dims for restorer: {feature_dims_per_layer}")

    restorer = LayerwiseRestorer(
        num_layers=len(sensitive_channels), feature_dims_per_layer=feature_dims_per_layer,
        num_classes=configs.dataloader.num_classes, expert_hidden_dim=configs.sensitive_restorer.get("expert_hidden_dim", 128)
    )
    restorer.to(device)
    total_params = sum(p.numel() for p in restorer.parameters() if p.requires_grad)
    logger.info(f"LayerwiseRestorer initialized. Trainable params: {total_params / 1e3:.1f}K")

    # --- 5. OPTIMIZER and SCHEDULER for RESTORER ---
    # This optimizer ONLY manages the restorer's parameters.
    logger.info("Creating optimizer and scheduler for the Restorer...")
    restorer_optimizer, _, restorer_lr_scheduler, _ = create_optimizer_and_lr_scheduler(
        restorer, configs, lr=configs.sensitive_restorer.restorer_lr, epochs=configs.sensitive_restorer.stage2_epochs
    )
    
    # --- 6. FAULT INJECTOR for STAGE 2 ---
    fault_injector = FaultInjector(
        model=training_model, mode="ber", ber=configs.sensitive_restorer.stage2_ber,
        enable_in_training=True, seed=configs.sensitive_restorer.stage2_seed
    )
    logger.info("Fault injector for Stage 2 created.")

    # --- 7. TRAINING LOOP ---
    # We now call the existing `train_process` function from process.py, 
    # passing all necessary components.
    logger.info("Starting Stage 2 training...")
    criterion = torch.nn.CrossEntropyLoss().cuda()
    
    for epoch in range(configs.sensitive_restorer.stage2_epochs):
        if is_master():
            logger.info('>>>>>>>> Epoch %3d' % epoch)

        # Call the training process function
        train_process(
            train_loader=train_loader,
            model=training_model,
            criterion=criterion,
            optimizer=None, # Pass None for main model optimizer
            epoch=epoch,
            monitors=monitors,
            configs=configs,
            model_ema=None,
            device=device,
            # Pass restorer components
            fault_injector=fault_injector,
            output_corrector=restorer,
            corrector_optimizer=restorer_optimizer
        )

        if restorer_lr_scheduler is not None:
            restorer_lr_scheduler.step(epoch + 1)
        
        # Save checkpoint for the restorer
        if is_master():
            save_checkpoint(
                epoch, configs.arch, model=restorer, optimizer=restorer_optimizer, 
                lr_scheduler=restorer_lr_scheduler, is_best=False, 
                name=f"{configs.name}_restorer", save_path=log_dir
            )

    if is_master():
        tbmonitor.writer.close()

if __name__ == "__main__":
    main()
