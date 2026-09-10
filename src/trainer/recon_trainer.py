import gc
import os, math,logging
from tqdm import tqdm
from pytorch3d.loss import chamfer_distance

import numpy as np
import torch
import torch.distributed as dist
from PIL import Image

from src.trainer.base_trainer import BaseTrainer
from src.datasets.utils import cam_to_world, save_pointcloud, scale_shift_alignment_pointcloud
from src.flow_matching.solver import ODESolver
from src.models.model_wrapper import BatchModelWrapper
from src.datasets.utils import denormalize_output
from src.utils.dist import all_reduce_mean

logger = logging.getLogger(__name__)

class ReconTrainer(BaseTrainer):
    def __init__(self,
                 config,
                 train_sampler,
                 train_dataloader,
                 val_dataloader,
                 model,
                 wandb_run=None,
                 runner_info=None):

        super().__init__(
            config,
            train_sampler,
            train_dataloader,
            val_dataloader,
            model,
            wandb_run=wandb_run,
            runner_info=runner_info,)

    def _ood_categories(self) -> set[str]:
        """Categories considered out-of-distribution at val time.

        Defined as the train-set ``exclude_categories`` list — those classes
        appear in val but the model never saw them during training, which is
        exactly the OOD generalization metric LARM-Table-8 reports for Oven.
        Returns an empty set if no categories are excluded from training.
        """
        try:
            ds_cfg = self.config.train_dataloader.dataset.params
        except Exception:
            return set()
        excl = ds_cfg.get('exclude_categories', None)
        if excl is None:
            return set()
        return {str(c) for c in excl}

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
        align_kwargs = dict(align_cfg) if align_cfg is not None else {}

        wrapper = BatchModelWrapper(model=self.model)
        solver  = ODESolver(velocity_model=wrapper)

        all_cd = []
        all_f1 = []
        num_items = 0
        all_pred = []
        all_gt = []
        all_images = []
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
            # samples = samples.to(device, non_blocking=True)   # (B, N, 3) — normalized GT
            ###############
            # Load data
            ###############
            images = data['image'] # B, V, C, H, W
            gt_pts = data["pts_gt"] # sampled from mesh surface.

            images = images.to(self.device, non_blocking=True)
            gt_pts = gt_pts.to(self.device, non_blocking=True)

            state_tag = data.get("state_tag", None)
            if state_tag is not None:
                state_tag = state_tag.to(self.device, non_blocking=True)

            input_images = images

            B = gt_pts.shape[0]

            # encoder_data = model.module._encode(images=images, pointmaps=pts3d_src)

            # Encode the conditioning point cloud. ``state_tag`` is consumed only
            # by models that opt into the per-view state embedding (articulated
            # subclass); the base ``Nova3rImgCond._encode`` swallows it via **kwargs.
            with torch.inference_mode():
                with torch.cuda.amp.autocast(dtype=torch.bfloat16):
                    inner = self.model.module if hasattr(self.model, 'module') else self.model
                    encoder_data = inner._encode(
                        images=images, pointmaps=None, test=True,
                        state_tag=state_tag,
                    )

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
                        pointmaps=None,
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
            if isinstance(batch_cats, str):
                batch_cats = [batch_cats]
            all_cats.extend([str(c) for c in batch_cats])

            if self.runner_info.rank == 0 and val_idx % save_pcd_interval == 0:
                all_pred.append(pred.detach().cpu())
                all_gt.append(gt_pts.detach().cpu())
                imgs_to_save = (input_images.detach() * 0.5 + 0.5).clamp(0, 1)
                imgs_to_save = (imgs_to_save * 255.0).to(torch.uint8)
                imgs_to_save = imgs_to_save.permute(0, 1, 3, 4, 2).contiguous().cpu()
                all_images.append(imgs_to_save)
                batch_ids = data.get("sample_id", None)
                if batch_ids is None:
                    batch_ids = [f"{val_idx:06d}_{i}" for i in range(pred.shape[0])]
                all_ids.append(list(batch_ids))
            if val_idx % 20 == 0:
                torch.cuda.empty_cache()
            
            del encoder_data, x_init, pred
            del dist_tuple, dist_pred, dist_gt, cd, precision, recall, f1
            del images, gt_pts, data

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
        # categories (samples without an obj_id->cat match) are reported as
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

        def _reduce_cd_f1(mask_np: np.ndarray) -> tuple[float, float, int]:
            if cats_np.size and mask_np.any():
                mask = torch.as_tensor(mask_np, device=cd_tensor.device)
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
            return cd_mean, f1_mean, total_cnt

        per_cat: dict[str, dict[str, float]] = {}
        for cat in global_cats:
            mask_np = (cats_np == cat) if cats_np.size else np.zeros(0, dtype=bool)
            cd_mean, f1_mean, total_cnt = _reduce_cd_f1(mask_np)
            label = cat if cat else "unknown"
            per_cat[label] = {"cd": cd_mean, "f1": f1_mean, "n": total_cnt}

        # OOD rollup: aggregate metrics over categories the train set
        # excluded (e.g. Oven). Always computed, even when the OOD set is
        # empty, so the wandb panel layout is stable across configs.
        ood_set = self._ood_categories()
        if ood_set and cats_np.size:
            ood_mask_np = np.isin(cats_np, list(ood_set))
        else:
            ood_mask_np = np.zeros(cats_np.shape if cats_np.size else 0, dtype=bool)
        ood_cd, ood_f1, ood_n = _reduce_cd_f1(ood_mask_np)

        # Save point clouds to work_dir (only every save_pcd_interval val calls)
        if self.runner_info.rank == 0:
            work_dir = self.runner_info.work_dir
            timestamp = getattr(self.runner_info, 'timestamp', '')
            pcd_dir = os.path.join(work_dir,'val_pcd', timestamp, f'step_{self.val_step}')
            os.makedirs(pcd_dir, exist_ok=True)
            sample_idx = 0
            
            for batch_pred, batch_gt, batch_imgs, batch_ids in zip(all_pred, all_gt, all_images, all_ids):
                for i in range(batch_pred.shape[0]):
                    name = str(batch_ids[i]) if i < len(batch_ids) else f'{sample_idx:06d}'
                    save_pointcloud(batch_pred[i], os.path.join(pcd_dir, f'{name}_pred.ply'))
                    save_pointcloud(batch_gt[i],   os.path.join(pcd_dir, f'{name}_gt.ply'))
                    for v in range(batch_imgs.shape[1]):
                        img_np = batch_imgs[i, v].numpy()
                        Image.fromarray(img_np).save(
                            os.path.join(pcd_dir, f'{name}_img_v{v}.png')
                        )
                    sample_idx += 1
            logger.info(f"[Val] {sample_idx} point clouds saved -> {pcd_dir}")

        logger.info(
            f"[Val/overall] CD={all_cd:.6f}  F1@{fs_thres}={all_f1:.4f}"
        )
        if ood_set:
            logger.info(
                f"[Val/ood] cats={sorted(ood_set)}  "
                f"CD={ood_cd:.6f}  F1@{fs_thres}={ood_f1:.4f}  n={ood_n}"
            )
        for label, m in per_cat.items():
            logger.info(
                f"[Val/per_class/{label}] CD={m['cd']:.6f}  "
                f"F1@{fs_thres}={m['f1']:.4f}  n={m['n']}"
            )

        if self.wandb_run is not None:
            # Wandb groups charts by the first ``/``-separated token, so we
            # use ``val-<group>/<metric>`` (hyphenated group at the top) to
            # make each group its own collapsible section on the run page.
            log_dict = {
                "val-overall/cd": all_cd,
                f"val-overall/f1@{fs_thres}": all_f1,
                "val/step": self.val_step,
            }
            if ood_set:
                log_dict["val-ood/cd"] = ood_cd
                log_dict[f"val-ood/f1@{fs_thres}"] = ood_f1
                log_dict["val-ood/n"] = ood_n
            for label, m in per_cat.items():
                log_dict[f"val-per_class/{label}/cd"] = m["cd"]
                log_dict[f"val-per_class/{label}/f1@{fs_thres}"] = m["f1"]
                log_dict[f"val-per_class/{label}/n"] = m["n"]
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

            ###############
            # Load data
            ###############
            images = data['image'] # B, V, C, H, W
            input_pts = data["pts"] # normed for flow-matching head

            images = images.to(self.device, non_blocking=True)
            input_pts = input_pts.to(self.device, non_blocking=True)

            ##################
            # Batch Forward
            #################
            ### for decoder's input
            noise = torch.rand_like(input_pts) * 2.0 - 1.0
            if self.train_cfg.skewed_timesteps:
                t = self.skewed_timestep_sample(input_pts.shape[0], device=self.device)
            else:
                t = torch.rand(input_pts.shape[0], device=self.device)
            path_sample = self.path.sample(t=t, x_0=noise, x_1=input_pts)
            x_t = path_sample.x_t
            u_t = path_sample.dx_t

            B = input_pts.shape[0]

            N = x_t.shape[1]
            t_unsqueezed = t.unsqueeze(1) # (B, 1)

            with torch.cuda.amp.autocast(dtype=torch.bfloat16):
                v_predict = self.model(
                    images=images,
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

            del images, input_pts, data
        
        torch.cuda.empty_cache()