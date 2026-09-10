import argparse
import logging
import os
import time
from pathlib import Path

import torch

from src.configs import build_dataset, build_model, load_config
from src.models.model_wrapper import BatchModelWrapper
from src.trainer import TRAINER_REGISTRY
from src.trainer.base_trainer import BaseTrainer
from src.trainer.recon_trainer import ReconTrainer
from src.utils.dist import setup_env
from src.utils.running import RunnerInfo

logger = logging.getLogger('artico')

from torch.utils.data import DataLoader


# ---------------------------------------------------------------------------
# Args
# ---------------------------------------------------------------------------
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="ptscond", help="Config name or path (default: ptscond)")
    parser.add_argument(
        '--ckpt-path', '--ckpt_path',
        help='the path to the checkpoint',
        default=None)
    parser.add_argument(
        '--launcher',
        choices=['none', 'pytorch', 'slurm', 'mpi'],
        default='pytorch',
        help='job launcher')
    parser.add_argument('--local_rank', '--local-rank', type=int, default=0)
    parser.add_argument(
        '--run-name', '--run_name',
        default=None,
        help=(
            'Optional human-readable name for this eval run. When set, it '
            'prefixes the timestamp in the output directory name '
            '(<work-dir>/<run_name>_<YYYYMMDD_HHMMSS>/) so A/B sweep runs are '
            'easy to tell apart.'
        ))
    parser.add_argument(
        '--work-dir', '--work_dir',
        default='./work_dirs',
        help=(
            'Root directory for evaluation outputs. Each run writes to '
            '<work-dir>/<run-name>_<timestamp>/, or <work-dir>/<timestamp>/ '
            'when --run-name is omitted.'))
    parser.add_argument(
        '--override', '-o',
        nargs='*',
        metavar='KEY=VALUE',
        help='OmegaConf dotlist overrides, e.g. --override model.params.num_3d_tokens=512 data.params.dummy=true')
    args = parser.parse_args()
    if 'LOCAL_RANK' not in os.environ:
        os.environ['LOCAL_RANK'] = str(args.local_rank)
    return args

if __name__ == "__main__":
    args = parse_args()
    cfg = load_config(args.config, overrides=args.override)

    # save some information useful during the training
    runner_info = RunnerInfo()
    
    # ---------------------------------------------------------------------------
    # Distributed env
    # ---------------------------------------------------------------------------
    if args.launcher == 'none':
        distributed = False
        timestamp = torch.tensor(time.time(), dtype=torch.float64)
        timestamp = time.strftime('%Y%m%d_%H%M%S', time.localtime(timestamp.item()))
        rank = 0
        world_size = 1
    else:
        distributed = True
        env_cfg = cfg.get('env_cfg', dict(dist_cfg=dict(backend='nccl')))
        rank, world_size, timestamp = setup_env(env_cfg, distributed, args.launcher)
    runner_info.launcher = args.launcher
    runner_info.rank = rank
    runner_info.distributed = distributed
    runner_info.world_size = world_size

    # Every run gets its own timestamped directory so repeated evals never
    # overwrite each other and sort chronologically. --run-name prefixes the
    # timestamp to keep A/B sweep folders grep-friendly.
    eval_tag = f'{args.run_name}_{timestamp}' if args.run_name else timestamp
    # Artifacts *inside* the run directory are named by the bare timestamp — the
    # run name is already in the directory, so repeating it just makes the paths
    # long (the trainer builds val_pcd/<timestamp>/ and
    # per_sample_metrics_<timestamp>.json from this).
    runner_info.timestamp = timestamp

    # ---------------------------------------------------------------------------
    # Output dir
    # ---------------------------------------------------------------------------
    runner_info.trail_name = os.path.split(args.work_dir.rstrip('/'))[-1]
    runner_info.work_dir = os.path.join(args.work_dir, eval_tag)
    os.makedirs(runner_info.work_dir, exist_ok=True)

    # ---------------------------------------------------------------------------
    # Logging  (rank 0 only — other ranks get a NullHandler so logger calls are
    #           silently dropped without any per-call rank checks)
    # ---------------------------------------------------------------------------
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.INFO)

    if rank == 0:
        # Slashes in --run-name are intentional (group sweeps under one parent),
        # but a slash in the log filename would create an unintended subdir.
        log_tag = eval_tag.replace('/', '_')
        log_filename = 'eval_{}.log'.format(log_tag)
        log_file = os.path.join(runner_info.work_dir, log_filename)
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

        file_handler = logging.FileHandler(log_file, mode='a')
        file_handler.setLevel(logging.INFO)
        file_handler.setFormatter(formatter)

        stream_handler = logging.StreamHandler()
        stream_handler.setLevel(logging.INFO)
        stream_handler.setFormatter(formatter)

        root_logger.addHandler(file_handler)
        root_logger.addHandler(stream_handler)
    else:
        root_logger.addHandler(logging.NullHandler())

    logger.setLevel(logging.INFO)

    # log config
    logger.info(f"\n[1] Loading config: {args.config!r}")
    logger.info(cfg)


    # ---------------------------------------------------------------------------
    # Build model
    # ---------------------------------------------------------------------------
    logger.info(f"\n[2] Building model: {cfg.model.name}")
    model = build_model(cfg)

    if cfg.get('load_pretrain', False):
        logger.info(f"    -- will load pretrained weights from {cfg.ckpt_path} (for building a model with pretrained weights, not for evaluation)")
        ckpt_path = cfg.ckpt_path
        ckpt = torch.load(ckpt_path, weights_only=False)
        if "model" in ckpt:
            state_dict = ckpt["model"]
        else:
            state_dict = ckpt
        missing, unexpected = model.load_state_dict(state_dict, strict=False)
        logger.info(f"    Missing keys   : {len(missing)}")
        logger.info(f"    Unexpected keys: {len(unexpected)}")
        if missing:
            logger.info("    -- missing (first 20) --")
            for k in missing[:20]:
                logger.info(f"        {k}")
        if unexpected:
            logger.info("    -- unexpected (first 20) --")
            for k in unexpected[:20]:
                logger.info(f"        {k}")


    logger.info(f"    -- will load evaluation weights from {args.ckpt_path}")
    ckpt_path = args.ckpt_path
    ckpt = torch.load(ckpt_path, weights_only=False)
    if "model" in ckpt:
        state_dict = ckpt["model"]
    else:
        state_dict = ckpt
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    logger.info(f"    Missing keys   : {len(missing)}")
    logger.info(f"    Unexpected keys: {len(unexpected)}")
    if missing:
        logger.info("    -- missing (first 20) --")
        for k in missing[:20]:
            logger.info(f"        {k}")
    if unexpected:
        logger.info("    -- unexpected (first 20) --")
        for k in unexpected[:20]:
            logger.info(f"        {k}")

    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"    Parameters: {total:,} total  /  {trainable:,} trainable")

    if runner_info.distributed:
        torch.cuda.set_device(runner_info.rank)
        model.cuda(runner_info.rank)
        model = torch.nn.parallel.DistributedDataParallel(
            model, device_ids=[runner_info.rank], output_device=runner_info.rank,
            find_unused_parameters=cfg.get('find_unused_parameters', False))
    else:
        model.cuda()

    # ---------------------------------------------------------------------------
    # Build dataset
    # ---------------------------------------------------------------------------
    dataset = build_dataset(cfg.val_dataloader.dataset)
    if runner_info.distributed:
        val_sampler = torch.utils.data.distributed.DistributedSampler(dataset, shuffle=False)
    else:
        val_sampler = None
    
    val_dataloader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        num_workers=cfg.val_dataloader.num_workers,
        pin_memory=True,
        persistent_workers=True,
        sampler=val_sampler)
    
    # build trainer — honor ``trainer_class`` if set (matches train.py dispatch)
    trainer_class_name = cfg.get("trainer_class", None)
    if trainer_class_name is not None:
        if trainer_class_name not in TRAINER_REGISTRY:
            raise ValueError(
                f"Unknown trainer_class={trainer_class_name!r}. "
                f"Registered: {sorted(TRAINER_REGISTRY)}"
            )
        trainer_cls = TRAINER_REGISTRY[trainer_class_name]
    elif cfg.get("stage", 1) == 1:
        trainer_cls = BaseTrainer
    else:
        trainer_cls = ReconTrainer

    trainer = trainer_cls(
        config=cfg,
        train_sampler=val_sampler,
        train_dataloader=val_dataloader,
        val_dataloader=val_dataloader,
        model=model,
        wandb_run=None,
        runner_info=runner_info)

    trainer.val_epoch()

