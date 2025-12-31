import logging
import torch
import yaml
import os
import time
from pathlib import Path
from timm.loss import LabelSmoothingCrossEntropy
from torch.nn.parallel import DistributedDataParallel
import torch.distributed as dist
from model import create_model
from util import (ProgressMonitor, TensorBoardMonitor, 
                  get_config, init_logger, set_global_seed, setup_print, load_checkpoint, save_checkpoint, preprocess_model, init_dataloader)
from util.mpq import sample_min_cands, switch_bit_width
from util.greedy_search import search, reset_bit_cands
from util.model_ema import ModelEma
from util.qat import get_quantized_layers
from util.loss_ops import DistributionLoss
from util.utils import create_optimizer_and_lr_scheduler
from util.dist import logger_info, is_master, init_dist_nccl_backend, tbmonitor_add_scalars
from util.weight_schd import CosineSched
from quan import find_modules_to_quantize, replace_module_by_names
from policy import BITS
from process import train, validate, PerformanceScoreboard, alexnet_train, alex_train_2, train_one_epoch_no_freezing_no_sample
from evolution_search import EvolutionSearcher
from data.cifar import Cifar100,Cifar10


def init_logger_and_monitor(configs, script_dir):
    # if is_master():
    dst_dir = Path('D:/BaiduNetdiskDownload/resnet18')
    output_dir = dst_dir / configs.output_dir
    output_dir.mkdir(exist_ok=True)
    log_dir = init_logger(configs.name, output_dir,
                          script_dir / 'logging.conf')
    logger = logging.getLogger()

    with open(log_dir / "configs.yaml", "w") as yaml_file:  # dump experiment config
        yaml.safe_dump(configs, yaml_file)

    pymonitor = ProgressMonitor(logger)
    tbmonitor = TensorBoardMonitor(logger, log_dir)

    return logger, log_dir, pymonitor, tbmonitor
    # else:
    #     return None, None, None, None

def main():

    script_dir = Path.cwd()
    configs = get_config(default_file=script_dir / 'template.yaml')

    assert configs.training_device == 'gpu', 'NOT SUPPORT CPU TRAINING NOW'

    init_dist_nccl_backend(configs)

    assert configs.rank >= 0, 'ERROR IN RANK'
    # assert configs.distributed

    logger, log_dir, pymonitor, tbmonitor = init_logger_and_monitor(
        configs, script_dir)
    monitors = [pymonitor, tbmonitor]

    setup_print(is_master=(configs.local_rank == 0))
    set_global_seed(seed=19)

    teacher_model = None
    using_distillation = configs.kd
    if using_distillation:
        teacher_model = create_model('resnet101')
        teacher_model.eval()

    model = create_model(configs.arch, pre_trained=configs.pre_trained)
    # model(torch.randn((1, 3, 224, 224)))
    model = preprocess_model(model, configs)

    logger_info(logger, 'Inserted quantizers into the original model')
    model = replace_module_by_names(model, find_modules_to_quantize(model, configs))

    model.eval()

    # 单卡训练，不需要DDP
    # wrap_the_model_with_ddp = lambda x: DistributedDataParallel(x.cuda(), device_ids=[configs.local_rank], find_unused_parameters=True)
    #
    # model = wrap_the_model_with_ddp(model)
    # if using_distillation:
    #     teacher_model = wrap_the_model_with_ddp(teacher_model)

    # ------------- data --------------
    dataset = Cifar10(32, 4)
    train_loader = dataset.train
    val_loader = dataset.test
    test_loader = dataset.test
    # train_loader, val_loader, test_loader, train_sampler, val_sampler = init_dataloader(configs.dataloader, arch=configs.arch)

    enable_linear_scaling_rule = False
    if enable_linear_scaling_rule:
        configs.lr = configs.lr * dist.get_world_size() * configs.dataloader.batch_size / 512
        configs.min_lr = configs.min_lr * \
            dist.get_world_size() * configs.dataloader.batch_size / 512
        configs.warmup_lr = configs.warmup_lr * \
            dist.get_world_size() * configs.dataloader.batch_size / 512

    optimizer, optimizer_q, lr_scheduler, lr_scheduler_q = create_optimizer_and_lr_scheduler(
        model, configs)

    start_epoch = 0
    print(model)
    model = model.cuda()
    # model(torch.randn((1, 3, 224, 224)).cuda())

    target_model = ModelEma(model, decay=configs.ema_decay)
    
    if configs.resume.path and os.path.exists(configs.resume.path):
        model, start_epoch, _ = load_checkpoint(model, configs.resume.path, 'cuda', lean=configs.resume.lean, optimizer=optimizer, override_optim=configs.eval,
                                                lr_scheduler=lr_scheduler, lr_scheduler_q=lr_scheduler_q, optimizer_q=optimizer_q)
        reset_bn_cands = not (getattr(configs, "eval", False) or getattr(configs, "search", False))
        
        w_cands, a_cands = target_model._load_checkpoint(configs.resume.path, )
        q_layers_ema, _ = get_quantized_layers(target_model.ema)
        for idx, layer in enumerate(q_layers_ema):
            layer.set_bit_cands(w_cands[idx], a_cands[idx])

    criterion = LabelSmoothingCrossEntropy(configs.smoothing).cuda() if configs.smoothing > 0. else \
        torch.nn.CrossEntropyLoss().cuda()

    soft_criterion = DistributionLoss() if teacher_model is not None else None

    mode = 'training' 
    target_bit_width = configs.target_bits
    max_bit_width_cand = max(target_bit_width)

    perf_scoreboard = PerformanceScoreboard(configs.log.num_best_scores)
    # print(model)
    print(configs.quan)
    switch_bit_width(model, quan_scheduler=configs.quan, 
                     wbit=target_bit_width, abits=target_bit_width)
    switch_bit_width(target_model.ema, quan_scheduler=configs.quan, 
                     wbit=target_bit_width, abits=target_bit_width)

    annealing_schedule = CosineSched(
        start_step=len(train_loader) * 5,
        max_step=len(train_loader) * configs.epochs,
        eta_start=0,
        eta_end=0.1
    )

    lr_scheduler.step(start_epoch)

    # freezing_annealing_schedule = None
    if configs.enable_dynamic_bit_training:
        logger_info(logger, 'Start dynamic bit-width training...')
        freezing_annealing_schedule = CosineSched(
            start_step=0,
            max_step=configs.epochs//2,
            eta_start=0.5,
            eta_end=0.2
        )

    if configs.eval:
        bitwidth_policies = BITS[configs.arch]

        bops_limit = []
        ret = validate(test_loader, target_model.ema, criterion, -1, monitors, configs, train_loader=train_loader,
                       eval_predefined_arch=bitwidth_policies, nr_random_sample=300, bops_limit=bops_limit)

        print(ret)

    elif configs.search:
        searcher = 'bid_search'

        assert searcher in ['bid_search', 'random_search', 'evolution_searcher']

        if searcher == 'evolution_searcher':
            q_layers, _ = get_quantized_layers(target_model.ema)
            searcher = EvolutionSearcher(configs, 'cuda', train_loader, target_model.ema, val_loader, test_loader, output_dir=f'./evolution_searcher/{configs.arch}/{configs.bops_limits}_bops', quantized_layers=q_layers)
            searcher.search()

        elif searcher == 'bid_search':
            reset_bit_cands(model=target_model.ema, reset=False)
            switch_bit_width(target_model.ema,
                            quan_scheduler=configs.quan, wbit=max_bit_width_cand-1, abit=max_bit_width_cand)
            
            conf = search(loader=train_loader, model=target_model.ema, criterion=criterion, metrics=('bitops', [configs.bops_limits]), epoch=0, cfgs=configs, start_bits=configs.start_bit_width,)
            
            acc = validate(test_loader, target_model.ema, criterion, -1, monitors,
                        configs, train_loader=train_loader, eval_predefined_arch=conf)
            print(conf)

        elif searcher == 'random_search':
            from util.random_search import do_random_search
            conf = do_random_search(train_loader, model, criterion=criterion, metrics=configs.bops_limits, quan_scheduler=configs.quan)
            print(conf)

    else:  # training
        if 0:
            alexnet_train(train_loader, model, 200, test_loader)
        if 1:
            v_top1, v_top5, v_loss = 0, 0, 0

            for epoch in range(start_epoch, configs.epochs):

                # alex_train_2(train_loader, test_loader, model, criterion, optimizer,
                #                                epoch, monitors, configs, model_ema=target_model, nr_random_sample=getattr(
                #                                    configs, 'num_random_path', 3),
                #                                soft_criterion=soft_criterion, teacher_model=teacher_model,
                #                                optimizer_q=optimizer_q, mode=mode,
                #                                annealing_schedule=annealing_schedule,
                #                                freezing_annealing_schedule=freezing_annealing_schedule
                #                                )
                train_one_epoch_no_freezing_no_sample(model, train_loader, test_loader, criterion, optimizer, epoch, configs,
                                          lr_scheduler=lr_scheduler, optimizer_q=optimizer_q,
                                          annealing_schedule=annealing_schedule,
                                          freezing_annealing_schedule=freezing_annealing_schedule)
                if lr_scheduler is not None:
                    lr_scheduler.step(epoch+1)

                if lr_scheduler_q is not None:
                    lr_scheduler_q.step()


                if (epoch+1) % 20 == 0 and epoch > 159:
                    torch.save(model.state_dict(),f'trained_models/alexnet_w6a6_no_freezing_no_sampling_{epoch}.pt')
    if configs.local_rank == 0:
        tbmonitor.writer.close()  # close the TensorBoard


if __name__ == "__main__":
    main()
