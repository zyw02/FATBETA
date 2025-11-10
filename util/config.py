import argparse
import logging
import logging.config
import os
import time
from pathlib import Path
import munch
import yaml


def merge_nested_dict(d, other):
    new = dict(d)
    for k, v in other.items():
        # Special marker to delete a key: use "__delete__" as value
        if v == "__delete__":
            if k in new:
                del new[k]
            continue
        if d.get(k, None) is not None and type(v) is dict:
            merged = merge_nested_dict(d[k], v)
            # If merged dict is empty or contains only "__delete__" markers, remove the key
            if merged and not all(v == "__delete__" for v in merged.values()):
                new[k] = merged
            else:
                # Remove keys that were marked for deletion
                new[k] = {k2: v2 for k2, v2 in merged.items() if v2 != "__delete__"}
        else:
            new[k] = v
    return new


def get_config(default_file):
    p = argparse.ArgumentParser(description='Learned Step Size Quantization')
    p.add_argument('config_file', metavar='PATH', nargs='+',
                   help='path to a configuration file')
    p.add_argument("--local_rank", default=0, type=int)
    p.add_argument("--enable_dynamic_bit_training", type=bool, default=True)
    p.add_argument("--split_aw_cands", type=bool, default=False)
    # Support for command line arguments that override YAML config
    # Use dot notation for nested keys, e.g., --dataloader.path /path/to/data
    p.add_argument("--bit_width_config_path", type=str, default=None,
                   help='path to bit width configuration JSON file from search')
    arg = p.parse_args()

    with open(default_file) as yaml_file:
        cfg = yaml.safe_load(yaml_file)

    for f in arg.config_file:
        if not os.path.isfile(f):
            raise FileNotFoundError('Cannot find a configuration file at', f)
        with open(f) as yaml_file:
            c = yaml.safe_load(yaml_file)
            cfg = merge_nested_dict(cfg, c)
    args = munch.munchify(cfg)
    args.local_rank = arg.local_rank
    if 'enable_dynamic_bit_training' not in args:
        args.enable_dynamic_bit_training = arg.enable_dynamic_bit_training
    if 'split_aw_cands' not in args:
        args.split_aw_cands = arg.split_aw_cands
    # Override with command line argument if provided
    if arg.bit_width_config_path is not None:
        args.bit_width_config_path = arg.bit_width_config_path

    print(args)

    return args


def init_logger(experiment_name, output_dir, cfg_file=None):
    time_str = time.strftime("%Y%m%d-%H%M%S")
    exp_full_name = time_str if experiment_name is None else experiment_name + '_' + time_str
    exp_full_name = experiment_name
    log_dir = output_dir / exp_full_name
    log_dir.mkdir(exist_ok=True)
    log_file = log_dir / (exp_full_name + '.log')
    logging.config.fileConfig(cfg_file, defaults={'logfilename': str(log_file)})
    logger = logging.getLogger()
    logger.info('Log file for this run: ' + str(log_file))
    return log_dir
