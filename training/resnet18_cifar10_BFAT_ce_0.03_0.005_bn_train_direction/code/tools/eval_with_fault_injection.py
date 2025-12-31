#!/usr/bin/env python
"""
评估混合精度模型在故障注入下的鲁棒性

此脚本展示了如何使用故障注入工具评估混合精度模型的单粒子翻转（SEU）鲁棒性。

用法:
    python tools/eval_with_fault_injection.py \
        --config configs/eval/eval_resnet18_cifar10_single_gpu.yaml \
        --bit_width_config search/resnet18_cifar10_single_gpu_search_bit_width_config.json \
        --ber_list "1e-8,1e-7,1e-6,1e-5,1e-4"
"""

import argparse
import sys
import os
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import torch
from util.config import get_config, init_logger
from model import create_model
from util.data_loader import init_dataloader
from util.checkpoint import load_checkpoint
from util.dist import logger_info
from util.fault_injector import FaultInjector, setup_model_with_bit_width_config
# No output corrector needed for main model evaluation
# from util.output_corrector import create_output_corrector
from process import validate
from util.monitor import ProgressMonitor
from timm.loss import LabelSmoothingCrossEntropy
from quan import find_modules_to_quantize, replace_module_by_names
from util.utils import preprocess_model
import logging


def parse_ber_list(ber_str: str):
    """Parse comma-separated BER values."""
    if not ber_str:
        return []
    return [float(x.strip()) for x in ber_str.split(',') if x.strip()]


def main():
    parser = argparse.ArgumentParser(description='Evaluate model with fault injection')
    parser.add_argument('--config', type=str, required=True,
                        help='Path to evaluation config YAML file')
    parser.add_argument('--bit_width_config', type=str, required=False, default="",
                        help='Path to bit-width configuration JSON file from search (optional, for dynamic bit training models)')
    parser.add_argument('--resume_path', type=str, default=None,
                        help='Path to checkpoint file (overrides YAML config resume.path)')
    parser.add_argument('--ber_list', type=str, default="1e-8,1e-7,1e-6,1e-5,1e-4",
                        help='Comma-separated list of BER values to test (e.g., "1e-8,1e-7,1e-6")')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed for reproducibility')
    parser.add_argument('--config_index', type=int, default=0,
                        help='Index of configuration to use from JSON file (default: 0)')
    parser.add_argument('--num_trials', type=int, default=5,
                        help='Number of trials for each BER value (default: 5). Each trial uses a different random seed to sample different fault patterns.')
    parser.add_argument('--use_position_based_mask', action='store_true', default=False,
                        help='If True, use position-based fixed mask (same weight position always gets same mask). If False, use random mask each time (default: False).')
    parser.add_argument('--seed_list', type=str, default=None,
                        help='Comma-separated list of seeds to use (e.g., "42,100,200,300"). If provided, trials will sample from this list instead of generating seeds. This ensures the same fault patterns as training.')
    # No calibration needed for main model evaluation
    # parser.add_argument('--calibration_samples', type=int, default=500,
    #                     help='Number of samples from test set for corrector calibration (default: 500, ~5%% of CIFAR10 test set). Set to 0 to disable calibration.')
    
    args, unknown = parser.parse_known_args()
    
    # Load config using get_config to properly handle __delete__ markers and template merging
    # We need to temporarily override sys.argv to avoid argparse conflicts
    import sys
    import yaml
    import munch
    from util.config import merge_nested_dict
    
    # Save original argv
    original_argv = sys.argv.copy()
    # Temporarily replace argv with just the config file to let get_config parse it
    sys.argv = ['eval_with_fault_injection.py', args.config]
    
    try:
        # Use get_config to properly merge with template.yaml and handle __delete__
        from util.config import get_config
        template_path = Path(__file__).parent.parent / 'template.yaml'
        if template_path.exists():
            configs = get_config(str(template_path))
        else:
            # Fallback: load directly if template doesn't exist
            with open(args.config, 'r') as f:
                cfg = yaml.safe_load(f)
            configs = munch.munchify(cfg)
    finally:
        # Restore original argv
        sys.argv = original_argv
    
    # Set some defaults that might be missing
    if not hasattr(configs, 'local_rank'):
        configs.local_rank = 0
    if not hasattr(configs, 'enable_dynamic_bit_training'):
        configs.enable_dynamic_bit_training = True
    if not hasattr(configs, 'split_aw_cands'):
        configs.split_aw_cands = False
    if not hasattr(configs, 'smoothing'):
        configs.smoothing = 0.0
    if not hasattr(configs, 'world_size'):
        configs.world_size = 1
    if not hasattr(configs, 'rank'):
        configs.rank = 0
    
    # Initialize logger
    script_dir = Path.cwd()
    output_dir = script_dir / configs.output_dir
    # Find logging config file (if exists)
    log_cfg_file = script_dir / "logging.conf"
    if not log_cfg_file.exists():
        log_cfg_file = None
    log_dir = init_logger(configs.name, output_dir, cfg_file=log_cfg_file)
    logger = logging.getLogger()
    
    # Setup device
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    configs.device = device
    
    logger_info(logger, f"Using device: {device}")
    logger_info(logger, f"Bit-width config: {args.bit_width_config}")
    
    # Set global random seed for reproducibility (CRITICAL for deterministic results)
    import random
    import numpy as np
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)
    # Set deterministic mode for CUDA operations (but keep benchmark=True for performance)
    # Note: benchmark=True may cause slight non-determinism, but is much faster
    # For fully deterministic results, set deterministic=True and benchmark=False (but slower)
    torch.backends.cudnn.deterministic = False  # Set to False for better performance
    torch.backends.cudnn.benchmark = True  # Enable benchmark for better performance
    logger_info(logger, f"Global random seed set to {args.seed} for reproducibility")
    
    # Parse BER values
    ber_values = parse_ber_list(args.ber_list)
    if not ber_values:
        logger.error("No BER values specified")
        return
    
    # Parse seed_list if provided
    seed_list = None
    if args.seed_list is not None:
        try:
            seed_list = [int(s.strip()) for s in args.seed_list.split(',')]
            if len(seed_list) == 1 and args.num_trials == 1:
                logger_info(logger, f"Using seed_list: {seed_list} (will use fixed seed {seed_list[0]} deterministically)")
            else:
                logger_info(logger, f"Using seed_list: {seed_list} (will randomly sample {args.num_trials} seeds from this list for each BER)")
        except ValueError:
            logger.warning(f"Invalid seed_list format: {args.seed_list}, ignoring...")
            seed_list = None
    
    logger_info(logger, f"Testing BER values: {ber_values}")
    
    # Create model
    dataset = getattr(configs.dataloader, 'dataset', 'cifar10')
    num_classes = getattr(configs.dataloader, 'num_classes', 10)
    model = create_model(configs.arch, dataset=dataset, pre_trained=configs.pre_trained)
    
    # Preprocess model (if needed)
    model = preprocess_model(model, configs)
    
    # Insert quantizers into the model
    logger_info(logger, 'Inserting quantizers into the model')
    model = replace_module_by_names(model, find_modules_to_quantize(model, configs))
    
    model = model.to(device)
    model.eval()
    
    # Load checkpoint
    # Priority: command line argument (--resume_path) > YAML config (resume.path)
    checkpoint_path = None
    if args.resume_path is not None:
        # Use command line argument if provided
        checkpoint_path = args.resume_path
        logger_info(logger, f"Using checkpoint from command line argument: {checkpoint_path}")
    elif hasattr(configs, 'resume') and hasattr(configs.resume, 'path') and configs.resume.path:
        # Fall back to YAML config
        checkpoint_path = configs.resume.path
        logger_info(logger, f"Using checkpoint from YAML config: {checkpoint_path}")
    
    # This script is for evaluating the main model with fault injection only
    # No output corrector is used here
    output_corrector = None
    
    if checkpoint_path:
        if os.path.exists(checkpoint_path):
            logger_info(logger, f"Loading checkpoint from {checkpoint_path}")
            # load_checkpoint returns (model, start_epoch, ...)
            # Use strict=False to allow missing quantizer parameters (they will be auto-initialized)
            result = load_checkpoint(model, checkpoint_path, device, strict=False, lean=getattr(configs.resume, 'lean', False))
            if isinstance(result, tuple):
                model = result[0]
            else:
                model = result
        else:
            logger.error(f"Checkpoint not found: {checkpoint_path}")
            return
    else:
        logger.error("No checkpoint path specified. Please provide --resume_path or set resume.path in YAML config.")
        return
    
    # Load and set bit-width configuration FIRST (CRITICAL for mixed-precision models)
    # This must be done BEFORE initializing output_size, as the bit width affects quantization
    # For dynamic bit training models (like stage1), bit_width_config may be empty
    if args.bit_width_config and args.bit_width_config.strip():
        logger_info(logger, f"Loading bit-width configuration from: {args.bit_width_config}")
        try:
            weight_bits, act_bits = setup_model_with_bit_width_config(
                model,
                args.bit_width_config,
                config_index=args.config_index,
                verbose=True
            )
            logger_info(logger, f"✓ Bit-width configuration loaded: {len(weight_bits)} layers")
        except Exception as e:
            logger.error(f"Failed to load bit-width configuration: {e}")
            import traceback
            traceback.print_exc()
            return
    else:
        # For dynamic bit training models, switch to max bit width
        logger_info(logger, "No bit-width config provided, using max target bit width (for dynamic bit training models)")
        from util.mpq import switch_bit_width
        target_bits = configs.target_bits if hasattr(configs, 'target_bits') else [6, 5, 4, 3, 2]
        max_bit = max(target_bits)
        logger_info(logger, f"Switching model to {max_bit}-bit quantization (max target-bit)...")
        logger_info(logger, f"Note: Layers in excepts (features.0, classifier.6) will remain 8-bit")
        switch_bit_width(model, quan_scheduler=configs.quan, wbit=max_bit, abits=max_bit)
        logger_info(logger, f"✓ Model switched to {max_bit}-bit successfully.")
    
    # Initialize output_size by doing a forward pass (needed for model_profiling)
    # This sets output_size for each QuanConv2d layer
    # IMPORTANT: Do this AFTER setting bit width, so quantization uses correct bit width
    logger_info(logger, "Initializing model output_size with a dummy forward pass")
    model.eval()
    with torch.no_grad():
        # Determine input size based on dataset
        input_size = 32 if dataset in ['cifar10', 'cifar100'] else 224
        dummy_input = torch.randn(1, 3, input_size, input_size).to(device)
        try:
            _ = model(dummy_input)
            logger_info(logger, "✓ Model output_size initialized")
        except Exception as e:
            logger.warning(f"Forward pass failed: {e}")
            import traceback
            traceback.print_exc()
    
    # Initialize data loaders
    train_loader, val_loader, test_loader, _, _ = init_dataloader(configs.dataloader, arch=configs.arch)
    
    # Create criterion
    criterion = LabelSmoothingCrossEntropy(configs.smoothing).cuda() if hasattr(configs, 'smoothing') and configs.smoothing > 0 else \
        torch.nn.CrossEntropyLoss().cuda()
    
    # Create monitors
    monitors = [ProgressMonitor(logger) for _ in range(len(ber_values) + 1)]
    
    # No output corrector calibration needed for main model evaluation
    
    # Baseline evaluation (no fault injection)
    logger_info(logger, "\n" + "="*60)
    logger_info(logger, "Baseline Evaluation (No Fault Injection)")
    logger_info(logger, "="*60)
    # Evaluate baseline (no fault injection)
    model.eval()
    baseline_result = validate(
        test_loader, model, criterion, -1, monitors[0], configs,
        train_loader=train_loader,
        eval_predefined_arch=[(32, None, None)]  # Preserve current bit-width (avoid MIN_POLICY)
    )
    baseline_acc = baseline_result[0] if isinstance(baseline_result, list) else baseline_result
    logger_info(logger, f"Baseline Top-1 Accuracy: {baseline_acc:.2f}%")
    
    # Evaluate with different BER values
    results = []
    for idx, ber in enumerate(ber_values, 1):
        logger_info(logger, "\n" + "="*60)
        logger_info(logger, f"Fault Injection Test {idx}/{len(ber_values)}: BER = {ber:.1e}")
        logger_info(logger, f"Performing {args.num_trials} trials with different random seeds...")
        logger_info(logger, "="*60)
        
        # Multiple trials with different random seeds (sampling different fault patterns)
        # If seed_list is provided, sample num_trials seeds from seed_list
        # Otherwise, generate seeds based on args.seed
        # Note: If seed_list has only one element and num_trials=1, use it directly (deterministic)
        if seed_list is not None:
            # If seed_list has only one element and we only need one trial, use it directly (deterministic)
            if len(seed_list) == 1 and args.num_trials == 1:
                sampled_seeds = seed_list  # Use the single seed directly, no random sampling
                logger_info(logger, f"  Using fixed seed from seed_list: {sampled_seeds[0]} (deterministic)")
            else:
                # Randomly sample num_trials seeds from seed_list (without replacement if possible)
                import random
                # Use a different random seed for each BER to ensure different samples
                random.seed(args.seed + idx * 1000)  # Different seed for each BER
                if len(seed_list) >= args.num_trials:
                    # Sample without replacement if we have enough seeds
                    sampled_seeds = random.sample(seed_list, args.num_trials)
                else:
                    # If not enough seeds, sample with replacement
                    sampled_seeds = [random.choice(seed_list) for _ in range(args.num_trials)]
                logger_info(logger, f"  Randomly sampled {args.num_trials} seeds from seed_list: {sampled_seeds}")
        else:
            # Generate seeds based on args.seed
            sampled_seeds = [args.seed + trial_idx * 1000 for trial_idx in range(args.num_trials)]
        
        trial_accs = []
        for trial_idx in range(args.num_trials):
            # Use the pre-sampled seed for this trial
            selected_seed = sampled_seeds[trial_idx]
            logger_info(logger, f"  Trial {trial_idx + 1}/{args.num_trials} (seed={selected_seed})...")
            
            # Create fault injector with selected seed
            # IMPORTANT: In eval mode, we want to use the explicit seed directly, not seed_list logic
            # So we don't pass seed_list to ensure deterministic behavior
            # Use position_based_mask based on args (default False for performance, True for full determinism)
            # If use_position_based_mask=True, same weight position always gets same mask (slower, fully deterministic)
            # If use_position_based_mask=False, uses torch.Generator with fixed seed (faster, deterministic if seed fixed)
            injector = FaultInjector(
                model=model,
                mode="ber",
                ber=ber,
                device=device,
                enable_in_training=False,
                enable_in_inference=True,
                seed=selected_seed,
                use_position_based_mask=args.use_position_based_mask,  # Use args setting (default False for performance)
                seed_list=None,  # Don't pass seed_list in eval mode to ensure explicit seed is used
                skip_first_last=False,  # Inject faults in all layers including first and last
            )
            
            # Enable fault injection
            injector.enable()
            
            # CRITICAL: validate() by default calls sample_min_cands() which sets model to minimum bit-width (2-bit)
            # This would break our fixed bit-width configuration (6-bit from JSON or max bit)!
            # Solution: Pass eval_predefined_arch=[(32, None, None)] to skip bit-width modification
            # When arch[0] == 32, validate() does 'pass' and doesn't change bit-width
            model.eval()
            if output_corrector is not None:
                output_corrector.set_runtime_context(ber=ber, stage='eval_fault')
            # Note: Root process.py validate() doesn't support output_corrector parameter
            # Use eval_predefined_arch=[(32, None, None)] to preserve current bit-width configuration
            result = validate(
                test_loader, model, criterion, -1, monitors[idx], configs,
                train_loader=train_loader,
                eval_predefined_arch=[(32, None, None)]  # Skip bit-width modification (arch[0] == 32 means 'pass')
            )
            # Handle both old format (single list) and new format (tuple of two lists)
            if isinstance(result, tuple):
                # New format: (model_accs, corrected_accs)
                model_acc, corrected_acc = result
                model_acc = model_acc[0] if isinstance(model_acc, list) else model_acc
                corrected_acc = corrected_acc[0] if isinstance(corrected_acc, list) else corrected_acc
                # 如果有corrector，保存两个值；否则只保存model acc
                if output_corrector is not None:
                    trial_accs.append((model_acc, corrected_acc))
                else:
                    trial_accs.append(model_acc)
            elif isinstance(result, list):
                # Old format: single list
                trial_accs.append(result[0])
            else:
                # Single value
                trial_accs.append(result)
            
            # Disable fault injection
            injector.disable()
            
            # Print flip statistics for this trial (only for the last trial to avoid too much output)
            if trial_idx == len(seed_list) - 1:
                logger_info(logger, f"\n故障注入统计 (BER={ber:.2e}, Trial {trial_idx + 1}):")
                injector.print_flip_statistics(verbose=False)  # Use verbose=False to reduce output
            
            # 打印trial结果
            if isinstance(trial_accs[-1], tuple):
                model_acc_trial, corrected_acc_trial = trial_accs[-1]
                logger_info(logger, f"    Trial {trial_idx + 1}: Model={model_acc_trial:.2f}%, With Corrector={corrected_acc_trial:.2f}%, Gain={corrected_acc_trial - model_acc_trial:+.2f}%")
            else:
                logger_info(logger, f"    Trial {trial_idx + 1}: Accuracy={trial_accs[-1]:.2f}%")
        
        # Calculate average accuracy across trials
        if output_corrector is not None and isinstance(trial_accs[0], tuple):
            # 有corrector：分别计算model和corrected的平均值
            model_accs = [acc[0] for acc in trial_accs]
            corrected_accs = [acc[1] for acc in trial_accs]
            avg_model_acc = sum(model_accs) / len(model_accs)
            avg_corrected_acc = sum(corrected_accs) / len(corrected_accs)
            std_model_acc = (sum((x - avg_model_acc) ** 2 for x in model_accs) / len(model_accs)) ** 0.5 if len(model_accs) > 1 else 0.0
            std_corrected_acc = (sum((x - avg_corrected_acc) ** 2 for x in corrected_accs) / len(corrected_accs)) ** 0.5 if len(corrected_accs) > 1 else 0.0
            acc_drop_model = baseline_acc - avg_model_acc
            acc_drop_corrected = baseline_acc_corrected - avg_corrected_acc
            results.append((ber, avg_model_acc, avg_corrected_acc, acc_drop_model, acc_drop_corrected, std_model_acc, std_corrected_acc, trial_accs))
            
            logger_info(logger, f"\nBER={ber:.1e}:")
            logger_info(logger, f"  Model Accuracy:     {avg_model_acc:.2f}% ± {std_model_acc:.2f}% (drop: {acc_drop_model:.2f}%)")
            logger_info(logger, f"  With Corrector:     {avg_corrected_acc:.2f}% ± {std_corrected_acc:.2f}% (drop: {acc_drop_corrected:.2f}%)")
            logger_info(logger, f"  Corrector Gain:     +{avg_corrected_acc - avg_model_acc:.2f}%")
        else:
            # 没有corrector：原有逻辑
            avg_acc = sum(trial_accs) / len(trial_accs)
            std_acc = (sum((x - avg_acc) ** 2 for x in trial_accs) / len(trial_accs)) ** 0.5 if len(trial_accs) > 1 else 0.0
            acc_drop = baseline_acc - avg_acc
            results.append((ber, avg_acc, acc_drop, std_acc, trial_accs))
            
            logger_info(logger, f"\nBER={ber:.1e}: Average Accuracy={avg_acc:.2f}% ± {std_acc:.2f}% (over {args.num_trials} trials)")
            logger_info(logger, f"  Individual trials: {[f'{acc:.2f}' for acc in trial_accs]}")
            logger_info(logger, f"  Drop from baseline: {acc_drop:.2f}%")
    
    # Extract checkpoint name from path
    checkpoint_path = args.resume_path if args.resume_path else getattr(configs.resume, 'path', 'Unknown')
    if checkpoint_path and checkpoint_path != 'Unknown':
        # Extract checkpoint name: get the directory name or filename
        checkpoint_path_obj = Path(checkpoint_path)
        # If it's a .pth.tar file, use the parent directory name; otherwise use filename
        if checkpoint_path_obj.suffixes == ['.pth', '.tar']:
            checkpoint_name = checkpoint_path_obj.parent.name
        else:
            checkpoint_name = checkpoint_path_obj.stem
    else:
        checkpoint_name = "Unknown"
    
    # Print summary
    logger_info(logger, "\n" + "="*80)
    logger_info(logger, "Summary")
    logger_info(logger, "="*80)
    logger_info(logger, f"Checkpoint: {checkpoint_name}")
    logger_info(logger, f"Bit Config: {Path(args.bit_width_config).name if args.bit_width_config else 'N/A'}")
    logger_info(logger, "-"*80)
    logger_info(logger, f"{'BER':<12} {'Accuracy (Avg)':<18} {'Std':<10} {'Drop':<12}")
    logger_info(logger, "-"*80)
    logger_info(logger, f"{'Baseline':<12} {baseline_acc:<18.2f} {'-':<10} {'0.00':<12}")
    for result in results:
        if len(result) == 8:  # With corrector: (ber, avg_model, avg_corrected, drop_model, drop_corrected, std_model, std_corrected, trials)
            ber, avg_model, avg_corrected, drop_model, drop_corrected, std_model, std_corrected, trial_accs = result
            logger_info(logger, f"{ber:<12.1e} {avg_model:<18.2f} {std_model:<10.2f} {drop_model:<12.2f} (Model)")
            logger_info(logger, f"{'':12} {avg_corrected:<18.2f} {std_corrected:<10.2f} {drop_corrected:<12.2f} (With Corrector, Gain: +{avg_corrected - avg_model:.2f}%)")
        elif len(result) == 5:  # Without corrector: (ber, avg_acc, drop, std_acc, trial_accs)
            ber, avg_acc, drop, std_acc, trial_accs = result
            logger_info(logger, f"{ber:<12.1e} {avg_acc:<18.2f} {std_acc:<10.2f} {drop:<12.2f}")
        else:  # Old format (backward compatibility)
            ber, acc, drop = result
            logger_info(logger, f"{ber:<12.1e} {acc:<18.2f} {'-':<10} {drop:<12.2f}")
    
    logger_info(logger, f"\nNote: Each BER value was evaluated {args.num_trials} times with different random seeds")
    logger_info(logger, "      to sample different fault patterns (simulating training distribution).")
    logger_info(logger, "\nEvaluation complete!")


if __name__ == "__main__":
    main()

