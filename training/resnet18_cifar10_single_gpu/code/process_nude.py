import logging
import time
import torch
import torch.nn as nn

from util import AverageMeter
from util.utils import accuracy, update_meter, set_global_seed
from util.dist import master_only, logger_info
from util.mpq import switch_bit_width

__all__ = ["train", "validate", "PerformanceScoreboard"]

logger = logging.getLogger()


def _max_target_bit(configs) -> int:
    target_bits = getattr(configs, "target_bits", [6, 5, 4, 3, 2])
    if isinstance(target_bits, (list, tuple)) and len(target_bits) > 0:
        return int(max(target_bits))
    return int(target_bits) if target_bits is not None else 6


def _force_max_bitwidth(model: nn.Module, configs) -> int:
    """Force all dynamic quantized layers to use max(target_bits) for both W/A."""
    max_bit = _max_target_bit(configs)
    try:
        switch_bit_width(model, quan_scheduler=configs.quan, wbit=max_bit, abits=max_bit)
    except Exception:
        # Best-effort: training should still run even if BN switching differs
        pass
    return max_bit


def _get_meters(mode: str = "training"):
    return {
        "name": mode,
        "loss": AverageMeter(),
        # Keep key names consistent with util.utils.update_meter()
        "top1": AverageMeter(),
        "top5": AverageMeter(),
        "QE_loss": AverageMeter(),
        "dist_loss": AverageMeter(),
        "IDM_loss": AverageMeter(),
        "batch_time": AverageMeter(),
    }


@master_only
def _update_monitors(monitors, meters, epoch, batch_idx, steps_per_epoch, optimizer, mode="training"):
    if monitors is None:
        return
    lr = optimizer.param_groups[0]["lr"] if optimizer is not None else 0.0
    stats = {
        "lr": lr,
        "loss": meters["loss"].avg,
        "acc1": meters["top1"].avg,
        "acc5": meters["top5"].avg,
    }
    for m in monitors:
        try:
            m.update(epoch, batch_idx, steps_per_epoch, mode=mode, **stats)
        except Exception:
            pass


def train(
    train_loader,
    model,
    criterion,
    optimizer,
    epoch,
    monitors,
    configs,
    model_ema=None,
    nr_random_sample=0,  # ignored by design (NO multi-path)
    mode="training",
    soft_criterion=None,
    teacher_model=None,
    optimizer_q=None,
    annealing_schedule=None,
    freezing_annealing_schedule=None,
    IDM_weight=0.0,
    scaler=None,
    fault_injector=None,
    output_corrector=None,
    corrector_optimizer=None,
    device=None,
):
    """
    NUDE training:
    - No nr_random_sample / mixed-path sampling
    - Always train the max(target_bits) subnet
    - Loss = CrossEntropy only (no QE loss / distribution loss / SR-QAT / orthogonality / other aux)
    """
    # Keep reproducibility consistent with the rest of the repo
    num_updates = epoch * len(train_loader)
    set_global_seed(num_updates + 1)

    model.train()
    if model_ema is not None:
        model_ema.ema.train()

    max_bit = _force_max_bitwidth(model, configs)
    logger_info(logger, f"[NUDE] Epoch {epoch}: force max bit-width W/A = {max_bit}")

    meters = _get_meters(mode=mode)
    end = time.time()
    steps_per_epoch = len(train_loader)

    for batch_idx, (inputs, targets) in enumerate(train_loader):
        inputs = inputs.cuda(non_blocking=True)
        targets = targets.cuda(non_blocking=True)

        optimizer.zero_grad()
        if optimizer_q is not None:
            optimizer_q.zero_grad()

        # Ensure bits are kept at max for dynamic layers
        _force_max_bitwidth(model, configs)

        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()

        nn.utils.clip_grad_value_(model.parameters(), 1.0)

        optimizer.step()
        if optimizer_q is not None:
            optimizer_q.step()

        if model_ema is not None:
            model_ema.update(model)

        acc1, acc5 = accuracy(outputs.data, targets.data, topk=(1, 5))
        update_meter(
            meters,
            loss,
            None,
            None,
            None,
            acc1,
            acc5,
            inputs.size(0),
            time.time() - end,
            configs.world_size,
        )
        end = time.time()

        if (batch_idx + 1) % configs.log.print_freq == 0:
            try:
                logger_info(
                    logger,
                    f"[NUDE][TRAIN] Epoch {epoch}/{configs.epochs} "
                    f"Iter {batch_idx+1}/{steps_per_epoch} "
                    f"Loss {meters['loss'].avg:.4f} "
                    f"Top1 {meters['top1'].avg:.2f} "
                    f"Top5 {meters['top5'].avg:.2f}",
                )
            except Exception:
                pass
            _update_monitors(monitors, meters, epoch, batch_idx, steps_per_epoch, optimizer, mode=mode)

    return meters["top1"].avg, meters["top5"].avg, meters["loss"].avg


def validate(
    data_loader,
    model,
    criterion,
    epoch,
    monitors,
    configs,
    nr_random_sample=0,  # ignored
    alpha=1,
    train_loader=None,
    eval_predefined_arch=None,
    bops_limit=1e10,
    train_mode=False,
):
    """
    NUDE validation: evaluate only the max(target_bits) subnet.
    """
    model.eval()
    max_bit = _force_max_bitwidth(model, configs)
    logger_info(logger, f"[NUDE] Validate: force max bit-width W/A = {max_bit}")

    meters = _get_meters(mode="validation")
    steps = len(data_loader)
    end = time.time()

    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(data_loader):
            inputs = inputs.cuda(non_blocking=True)
            targets = targets.cuda(non_blocking=True)

            _force_max_bitwidth(model, configs)
            outputs = model(inputs)
            loss = criterion(outputs, targets)

            acc1, acc5 = accuracy(outputs.data, targets.data, topk=(1, 5))
            update_meter(
                meters,
                loss,
                None,
                None,
                None,
                acc1,
                acc5,
                inputs.size(0),
                time.time() - end,
                configs.world_size,
            )
            end = time.time()

    _update_monitors(monitors, meters, epoch, steps - 1, steps, optimizer=None, mode="validation")
    return meters["top1"].avg


class PerformanceScoreboard:
    def __init__(self, num_best_scores=3):
        self.num_best_scores = num_best_scores
        self.board = []

    def update(self, top1, top5, epoch):
        self.board.append((top1, top5, epoch))
        self.board = sorted(self.board, key=lambda x: x[0], reverse=True)[: self.num_best_scores]

    def is_best(self, top1):
        if len(self.board) == 0:
            return True
        return top1 >= max([x[0] for x in self.board])


