# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the CC-by-NC license found in the
# LICENSE file in the root directory of this source tree.
import argparse
import gc
import logging
import math
import os
from typing import Iterable

import numpy as np
import torch
import torch.distributed as dist
from pytorch3d.loss import chamfer_distance
from tqdm import tqdm

from src.datasets.utils import denormalize_output, save_pointcloud, scale_shift_alignment_pointcloud
from src.flow_matching.path import AffineProbPath
from src.flow_matching.path.scheduler import CosineScheduler
from src.flow_matching.solver import ODESolver
from src.models.model_wrapper import BatchModelWrapper
from src.utils.dist import all_reduce_mean

logger = logging.getLogger(__name__)

MASK_TOKEN = 256
PRINT_FREQUENCY = 50

class BaseTrainer:
    def __init__(self,
                 config,
                 train_sampler,
                 train_dataloader,
                 val_dataloader,
                 model,
                 wandb_run=None,
                 runner_info=None,
                 ):
    
        self.config = config
        self.wandb_run = wandb_run
        self.runner_info = runner_info

        
        self.train_sampler = train_sampler
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        self.model = model

        # build optimizer and scheduler
        optim_cfg = config.trainer.optim
        head_lr = optim_cfg.lr
        backbone_lr = optim_cfg.get('backbone_lr', None)
        head_prefixes = tuple(
            optim_cfg.get('head_prefixes', ('seg_head', 'articulation_head'))
        )
        # Optional third LR group for modules that need slower fine-tuning
        # than the encoder backbone — e.g. an unfrozen FM decoder, where
        # the encoder's tuned LR (3e-5) is destabilising for the
        # ODE-integrated decoder.
        decoder_prefixes = tuple(
            optim_cfg.get('decoder_prefixes', ()) or ()
        )
        decoder_lr = optim_cfg.get('decoder_lr', None)

        # Inspect parameter names on the unwrapped module so DDP's
        # ``module.`` prefix doesn't have to be threaded through every
        # comparison. Param tensors are passed by reference, so the
        # optimizer ends up tracking exactly the same tensors regardless.
        inner_for_optim = (
            self.model.module if hasattr(self.model, 'module') else self.model
        )

        if backbone_lr is not None:
            # Two-group AdamW: heads (matched by ``head_prefixes``) at
            # ``head_lr`` and the rest of the trainable params (the
            # pretrained backbone) at ``backbone_lr``. Used by full-stack
            # fine-tune recipes (e.g. phase5b_se_pp_ft) where heads need
            # the standard 3e-4 to learn from scratch but the backbone
            # should only be nudged by ~10× less.
            head_params, backbone_params, decoder_params = [], [], []
            head_names, backbone_names, decoder_names = [], [], []
            for name, p in inner_for_optim.named_parameters():
                if not p.requires_grad:
                    continue
                # Priority: head_prefixes > decoder_prefixes > backbone.
                # head_prefixes win so seg_head/articulation_head can't be
                # accidentally captured by a coarser decoder prefix.
                if any(
                    name == pref or name.startswith(pref + '.')
                    for pref in head_prefixes
                ):
                    head_params.append(p)
                    head_names.append(name)
                elif decoder_prefixes and any(
                    name == pref or name.startswith(pref + '.')
                    for pref in decoder_prefixes
                ):
                    decoder_params.append(p)
                    decoder_names.append(name)
                else:
                    backbone_params.append(p)
                    backbone_names.append(name)

            # Sanity check: every trainable tensor must land in exactly
            # one group. If a param name is missed by the prefix list,
            # we'd silently drop it from the optimizer and it would
            # appear to train but never update.
            n_train_total = sum(
                1 for _, p in inner_for_optim.named_parameters() if p.requires_grad
            )
            assert len(head_params) + len(backbone_params) + len(decoder_params) == n_train_total, (
                f"param-group split missed tensors: "
                f"head={len(head_params)} + backbone={len(backbone_params)} "
                f"+ decoder={len(decoder_params)} != trainable={n_train_total}"
            )

            n_head_p = sum(p.numel() for p in head_params)
            n_back_p = sum(p.numel() for p in backbone_params)
            n_dec_p = sum(p.numel() for p in decoder_params)
            decoder_lr_eff = decoder_lr if decoder_lr is not None else backbone_lr
            logger.info(
                f"[optim] param groups: "
                f"head ({head_lr:.2e}) {len(head_params)} tensors / {n_head_p:,} params; "
                f"backbone ({backbone_lr:.2e}) {len(backbone_params)} tensors / "
                f"{n_back_p:,} params; "
                f"decoder ({decoder_lr_eff:.2e}) {len(decoder_params)} tensors / "
                f"{n_dec_p:,} params"
            )
            logger.info(
                f"[optim] head prefixes: {list(head_prefixes)}; "
                f"head sample: {head_names[:4]}{' ...' if len(head_names) > 4 else ''}"
            )
            logger.info(
                f"[optim] decoder prefixes: {list(decoder_prefixes)}; "
                f"decoder sample: {decoder_names[:4]}{' ...' if len(decoder_names) > 4 else ''}"
            )
            logger.info(
                f"[optim] backbone sample: "
                f"{backbone_names[:4]}{' ...' if len(backbone_names) > 4 else ''}"
            )

            param_groups = []
            if head_params:
                param_groups.append({'params': head_params, 'lr': head_lr})
            if backbone_params:
                param_groups.append({'params': backbone_params, 'lr': backbone_lr})
            if decoder_params:
                param_groups.append({'params': decoder_params, 'lr': decoder_lr_eff})
            self.optimizer = torch.optim.AdamW(
                param_groups,
                lr=head_lr,  # AdamW fallback; per-group lr already set above
                weight_decay=optim_cfg.get('weight_decay', 0.05),
                betas=tuple(optim_cfg.get('betas', (0.9, 0.999))),
            )
        else:
            self.optimizer = torch.optim.AdamW(
                self.model.parameters(),
                lr=head_lr,
                weight_decay=optim_cfg.get('weight_decay', 0.05),
                betas=tuple(optim_cfg.get('betas', (0.9, 0.999))),
            )
        total_steps  = self.config.trainer.train_cfg.max_epochs * len(self.train_dataloader)
        warmup_steps = optim_cfg.get('warmup_steps', 0)

        cosine = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=max(total_steps - warmup_steps, 1),
            eta_min=optim_cfg.get('min_lr', 0.0),
        )

        if warmup_steps > 0:
            warmup = torch.optim.lr_scheduler.LinearLR(
                self.optimizer,
                start_factor=1e-6,
                end_factor=1.0,
                total_iters=warmup_steps,
            )
            self.scheduler = torch.optim.lr_scheduler.SequentialLR(
                self.optimizer,
                schedulers=[warmup, cosine],
                milestones=[warmup_steps],
            )
        else:
            self.scheduler = cosine
    
        # I'd like use wandb log_name
        self.train_step = 0 # for training
        self.val_step = 0 # for validation

        self.iters_per_train_epoch = len(self.train_dataloader)
        self.iters_per_val_epoch = len(self.val_dataloader)
        
        self.device = 'cuda'
        self.train_cfg = config.trainer.train_cfg

        self.path = AffineProbPath(scheduler=CosineScheduler())
        logger.info('successfully init trainer')

    def run(self):
        if self.runner_info.debug_val is True:
            self.val_epoch() # do you want to debug val step?

        # self.save_checkpoint(0)
        is_rank0 = self.runner_info.rank == 0
        total_steps = self.train_cfg.max_epochs * self.iters_per_train_epoch
        train_bar = tqdm(
            total=total_steps,
            desc="Training",
            unit="step",
            disable=not is_rank0,
            dynamic_ncols=True,
        )
        for epoch_idx in range(self.train_cfg.max_epochs):
            if self.runner_info.distributed:
                self.train_sampler.set_epoch(epoch_idx)
            self.train_epoch(epoch_idx, train_bar=train_bar)
            if (epoch_idx + 1) % self.train_cfg.val_interval == 0 and (epoch_idx + 1) >= self.train_cfg.get('eval_start', 0) and self.train_cfg.val_type == 'epoch_base':
                self.val_epoch()
            if (epoch_idx + 1) % self.train_cfg.save_checkpoint_interval == 0:
                self.save_checkpoint(epoch_idx)
            if (epoch_idx + 1) % self.train_cfg.get('early_stop_epoch', 9999999) == 0: # Are you using 99999999+ epochs?
                logger.info('early stop at epoch: {}'.format(epoch_idx))
                break
        train_bar.close()

        if self.train_cfg.val_type == 'iter_base':
            self.val_epoch()

    def skewed_timestep_sample(self, num_samples: int, device: torch.device) -> torch.Tensor:
        P_mean = -1.2
        P_std = 1.2
        rnd_normal = torch.randn((num_samples,), device=device)
        sigma = (rnd_normal * P_std + P_mean).exp()
        time = 1 / (1 + sigma)
        time = torch.clip(time, min=0.0001, max=1.0)
        return time

    def val_epoch(self):
        self.model.eval()
        device = torch.device(self.device)

        step_size = self.config.get('fm_step_size', 0.04)
        method    = self.config.get('fm_sampling', 'euler')
        num_steps = round(1.0 / step_size)
        T         = torch.linspace(0, 1, num_steps + 1, device=device)

        num_queries = self.config.get('val_num_queries', 8192)
        fs_thres    = self.config.get('val_fs_thres', 0.05)

        align_cfg = self.config.get('alignment', None)
        align_kwargs = dict(align_cfg) if align_cfg is not None else {
            'num_sample': None,
            'filter_outliers': False,
        }

        wrapper = BatchModelWrapper(model=self.model)
        solver  = ODESolver(velocity_model=wrapper)

        all_cd = []
        all_f1 = []
        all_pred = []
        all_gt = []
        all_ids = []
        all_cats: list[str] = []
        save_pcd_interval = self.train_cfg.get('save_pcd_interval', 10)

        is_rank0 = self.runner_info.rank == 0
        val_bar = tqdm(
            self.val_dataloader,
            desc=f"Val (step {self.val_step})",
            unit="batch",
            disable=not is_rank0,
            dynamic_ncols=True,
        )
        for val_idx, data in enumerate(val_bar):
            samples = data["pts"]
            gt_pts = data["pts_gt"]  # (B, N, 3)

            samples = samples.to(self.device, non_blocking=True)   # (B, N, 3) — normalized GT
            gt_pts = gt_pts.to(self.device, non_blocking=True)
            B = samples.shape[0]
            images = torch.zeros(B, 1, 3, 1, 1, device=device)

            # Encode the conditioning point cloud
            with torch.inference_mode():
                with torch.cuda.amp.autocast(dtype=torch.bfloat16):
                    encoder_data = self.model._encode(pointmaps=samples, test=True) \
                        if not hasattr(self.model, 'module') \
                        else self.model.module._encode(pointmaps=samples, test=True)

                # ODE sampling: noise → predicted point cloud
                x_init = torch.rand(B, num_queries, 3, device=device) * 2.0 - 1.0

                with torch.cuda.amp.autocast(dtype=torch.bfloat16):
                    pred = solver.sample(
                        x_init=x_init,
                        time_grid=T,
                        method=method,
                        step_size=step_size,
                        return_intermediates=False,
                        images=images,
                        token_mask=None,
                        encoder_data=encoder_data,
                        pointmaps=samples,
                    )                              # (B, num_queries, 3)

            pred = pred.float()

            # Subsample GT to the same size for a fair comparison
            # gt = meta["gt"].to(device)  # (B, N, 3)
            if gt_pts.shape[1] > num_queries:
                idx = torch.randperm(gt_pts.shape[1], device=device)[:num_queries]
                gt_pts = gt_pts[:, idx, :].float()
            else:
                gt_pts = gt_pts.float()

            gt_valid = torch.ones((gt_pts.shape[:2]), dtype=torch.bool)
            pred, pred_mask = scale_shift_alignment_pointcloud(
                pred, gt_pts, gt_valid, **align_kwargs,
            )
            pred = pred[pred_mask].unsqueeze(0)
            
            # --- Chamfer Distance ---
            # pytorch3d returns squared distances; take sqrt for L2
            dist_tuple, _ = chamfer_distance(
                pred, gt_pts,
                batch_reduction=None, point_reduction=None, norm=2,
            )
            dist_pred, dist_gt = dist_tuple          # (B, num_queries) each, squared
            dist_pred = torch.sqrt(dist_pred)
            dist_gt   = torch.sqrt(dist_gt)
            cd = (dist_pred.mean(dim=1) + dist_gt.mean(dim=1)) / 2.0   # (B,)

            # --- F1 @ fs_thres ---
            precision = (dist_pred < fs_thres).float().mean(dim=1)      # (B,)
            recall    = (dist_gt   < fs_thres).float().mean(dim=1)      # (B,)
            f1 = 2.0 * precision * recall / (precision + recall + 1e-8) # (B,)

            all_cd.append(cd)
            all_f1.append(f1)

            batch_cats = data.get("category", None)
            if batch_cats is None:
                batch_cats = [""] * B
            # DataLoader default collate turns list[str] into list[str]; str stays str.
            if isinstance(batch_cats, str):
                batch_cats = [batch_cats]
            all_cats.extend([str(c) for c in batch_cats])

            if self.runner_info.rank == 0 and val_idx % save_pcd_interval == 0:
                all_pred.append(pred.detach().cpu())
                all_gt.append(gt_pts.detach().cpu())
                batch_ids = data.get("sample_id", None)
                if batch_ids is None:
                    batch_ids = [f"{val_idx:06d}_{i}" for i in range(pred.shape[0])]
                all_ids.append(list(batch_ids))
            if val_idx % 20 == 0:
                torch.cuda.empty_cache()

        cd_tensor = torch.cat(all_cd)   # (local_N,)
        f1_tensor = torch.cat(all_f1)   # (local_N,)

        # calculate mean cd and f1 on each device
        device_cd = cd_tensor.mean().item()
        device_f1 = f1_tensor.mean().item()
        # logger.info(
        #     f"[Val] Device {self.runner_info.rank}: CD={device_cd:.6f}  F1@{fs_thres}={device_f1:.4f}"
        # )
        all_cd = all_reduce_mean(cd_tensor.sum().item(), cd_tensor.shape[0])
        all_f1 = all_reduce_mean(f1_tensor.sum().item(), f1_tensor.shape[0])

        # ---- Per-category CD / F1 -------------------------------------------
        # Gather the union of categories seen across ranks so every rank
        # iterates the same (cat) list in the same order. Empty-string
        # categories (samples without an obj_id→cat match) are reported as
        # "unknown".
        is_dist = dist.is_available() and dist.is_initialized()
        local_cat_set = set(all_cats)
        if is_dist:
            gathered: list = [None] * dist.get_world_size()
            dist.all_gather_object(gathered, list(local_cat_set))
            global_cats = sorted({c for sub in gathered for c in sub})
        else:
            global_cats = sorted(local_cat_set)

        cats_np = np.asarray(all_cats, dtype=object)
        per_cat: dict[str, dict[str, float]] = {}
        for cat in global_cats:
            if cats_np.size and (cats_np == cat).any():
                mask = torch.as_tensor(cats_np == cat, device=cd_tensor.device)
                local_sum_cd = cd_tensor[mask].sum().item()
                local_sum_f1 = f1_tensor[mask].sum().item()
                local_cnt = int(mask.sum().item())
            else:
                local_sum_cd = 0.0
                local_sum_f1 = 0.0
                local_cnt = 0
            if is_dist:
                t = torch.tensor(
                    [local_sum_cd, local_sum_f1, float(local_cnt)],
                    dtype=torch.float64, device='cuda',
                )
                dist.all_reduce(t, op=dist.ReduceOp.SUM)
                total_cnt = int(t[2].item())
                cd_mean = (t[0] / t[2]).item() if total_cnt > 0 else float('nan')
                f1_mean = (t[1] / t[2]).item() if total_cnt > 0 else float('nan')
            else:
                total_cnt = local_cnt
                cd_mean = local_sum_cd / local_cnt if local_cnt > 0 else float('nan')
                f1_mean = local_sum_f1 / local_cnt if local_cnt > 0 else float('nan')
            label = cat if cat else "unknown"
            per_cat[label] = {"cd": cd_mean, "f1": f1_mean, "n": total_cnt}

        # Save point clouds to work_dir (only every save_pcd_interval val calls)
        if self.runner_info.rank == 0:
            work_dir = self.runner_info.work_dir
            timestamp = getattr(self.runner_info, 'timestamp', '')
            pcd_dir = os.path.join(work_dir,'val_pcd', timestamp, f'step_{self.val_step}')
            os.makedirs(pcd_dir, exist_ok=True)
            sample_idx = 0
            for batch_pred, batch_gt, batch_ids in zip(all_pred, all_gt, all_ids):
                for i in range(batch_pred.shape[0]):
                    name = str(batch_ids[i]) if i < len(batch_ids) else f'{sample_idx:06d}'
                    save_pointcloud(batch_pred[i], os.path.join(pcd_dir, f'{name}_pred.ply'))
                    save_pointcloud(batch_gt[i],   os.path.join(pcd_dir, f'{name}_gt.ply'))
                    sample_idx += 1
            logger.info(f"[Val] {sample_idx} point clouds saved -> {pcd_dir}")

        logger.info(
            f"[Val] CD={all_cd:.6f}  F1@{fs_thres}={all_f1:.4f}"
        )
        for label, m in per_cat.items():
            logger.info(
                f"[Val/{label}] CD={m['cd']:.6f}  "
                f"F1@{fs_thres}={m['f1']:.4f}  n={m['n']}"
            )

        if self.wandb_run is not None:
            log_dict = {
                "val/chamfer_distance": all_cd,
                f"val/f1@{fs_thres}": all_f1,
                "val/step": self.val_step,
            }
            for label, m in per_cat.items():
                log_dict[f"val/{label}/chamfer_distance"] = m["cd"]
                log_dict[f"val/{label}/f1@{fs_thres}"] = m["f1"]
                log_dict[f"val/{label}/n"] = m["n"]
            self.wandb_run.log(log_dict)
        self.val_step += 1

        del wrapper, solver
        torch.cuda.empty_cache()


        result = {'cd': all_cd, f'f1@{fs_thres}': all_f1}
        result['per_category'] = per_cat
        return result

    def save_checkpoint(self, epoch: int):
        model = self.model.module if hasattr(self.model, 'module') else self.model
        ckpt = {
            'epoch': epoch + 1,
            'model': model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'scheduler': self.scheduler.state_dict(),
        }
        if self.runner_info.rank == 0: # Only save checkpoint on rank 0 to avoid conflicts
            work_dir = self.runner_info.work_dir
            os.makedirs(work_dir, exist_ok=True)
            path = os.path.join(work_dir, f'epoch_{epoch + 1}.pth')
            torch.save(ckpt, path)
            logger.info(f"Checkpoint saved -> {path}")

    def train_epoch(
        self,
        epoch: int,
        train_bar: tqdm = None,
    ):
        gc.collect()
        self.model.train(True)

        for data_iter_step, data in enumerate(self.train_dataloader):
            self.optimizer.zero_grad()

            samples = data['pts']
            samples = samples.to(self.device, non_blocking=True)

            noise = torch.rand_like(samples) * 2.0 - 1.0
            if self.train_cfg.skewed_timesteps:
                t = self.skewed_timestep_sample(samples.shape[0], device=self.device)
            else:
                t = torch.rand(samples.shape[0], device=self.device)
            path_sample = self.path.sample(t=t, x_0=noise, x_1=samples)
            x_t = path_sample.x_t
            u_t = path_sample.dx_t

            B = samples.shape[0]
            images = torch.zeros(B, 1, 3, 1, 1, device=self.device)

            N = x_t.shape[1]
            t_unsqueezed = t.unsqueeze(1) # (B, 1)

            with torch.cuda.amp.autocast(dtype=torch.bfloat16):
                v_predict = self.model(
                    images=images,
                    pointmaps=samples,
                    token_mask=None,
                    query_points=x_t,
                    timestep=t_unsqueezed,)['pts3d_xyz']
                loss = torch.pow(v_predict - u_t, 2).mean()

            loss_value = loss.item()

            if not math.isfinite(loss_value):
                raise ValueError(f"Loss is {loss_value}, stopping training")

            loss.backward()
            max_norm = self.config.trainer.optim.get('grad_clip', 1.0)
            grad_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm)
            self.optimizer.step()

            lr = self.optimizer.param_groups[0]["lr"]
            if data_iter_step % self.train_cfg.logging_interval == 0:
                logger.info(
                    f"Epoch {epoch} [{data_iter_step}/{len(self.train_dataloader)}]: "
                    f"loss = {loss_value:.6f}, grad_norm = {grad_norm:.4f}, lr = {lr:.2e}"
                )

            if train_bar is not None and data_iter_step % self.train_cfg.logging_interval == 0: # update progress bar every epoch
                train_bar.set_postfix(
                    epoch=epoch,
                    loss=f"{loss_value:.4f}",
                    lr=f"{lr:.2e}",
                    refresh=False,
                )
                train_bar.update(self.train_cfg.logging_interval)

            if self.wandb_run is not None and data_iter_step % self.train_cfg.logging_interval == 0:
                self.wandb_run.log({
                    "train/loss": loss_value,
                    "train/grad_norm": grad_norm.item() if torch.is_tensor(grad_norm) else grad_norm,
                    "train/lr": lr,
                    "train/epoch": epoch,
                    "train/step": self.train_step,
                })
            self.train_step += 1

            self.scheduler.step()

            del data
        
        torch.cuda.empty_cache()