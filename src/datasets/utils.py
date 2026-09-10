import torch
import open3d as o3d
import numpy as np
import trimesh
from pytorch3d.loss import chamfer_distance


def load_mesh_as_pointcloud(obj_path: str, num_points: int = 8192) -> np.ndarray:
    """Load OBJ mesh and sample point cloud from surface."""
    mesh = trimesh.load(obj_path, force="mesh", process=False)
    if isinstance(mesh, trimesh.Scene):
        mesh = mesh.dump(concatenate=True)
    points, _ = trimesh.sample.sample_surface(mesh, count=num_points)
    return np.array(points, dtype=np.float32)

def load_pointcloud(ply_path: str, max_points: int = 50000, verbose: bool = False) -> np.ndarray:
    pcd = o3d.io.read_point_cloud(ply_path)
    pts = np.asarray(pcd.points, dtype=np.float32)
    if len(pts) > max_points:
        idx = np.random.choice(len(pts), max_points, replace=False)
        pts = pts[idx]
        if verbose:
            print(f"  Subsampled {len(np.asarray(pcd.points))} -> {max_points} points")
    else:
        if verbose:
            print(f"  Loaded {len(pts)} points")
    return pts

def save_pointcloud(pts3d: torch.Tensor, out_path: str, verbose: bool = False):
    """Save point cloud as PLY."""
    combined_pts = pts3d.reshape(-1, 3).cpu().numpy()

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(combined_pts)
    o3d.io.write_point_cloud(out_path, pcd)
    if verbose:
        print(f"  Saved -> {out_path}")


def save_segmented_pointcloud(
    pts3d: "torch.Tensor | np.ndarray",
    labels: "torch.Tensor | np.ndarray",
    out_path: str,
    palette: "list[tuple[float, float, float]] | None" = None,
    verbose: bool = False,
) -> None:
    """Save a point cloud as PLY with per-point colors picked by integer label.

    ``palette`` defaults to a 2-class scheme — class 0 light grey (context),
    class 1 bright red (active part). Label values outside the palette range
    are clipped to the palette's last color.
    """
    if isinstance(pts3d, torch.Tensor):
        pts = pts3d.detach().cpu().reshape(-1, 3).numpy()
    else:
        pts = np.asarray(pts3d, dtype=np.float32).reshape(-1, 3)

    if isinstance(labels, torch.Tensor):
        lbl = labels.detach().cpu().reshape(-1).numpy().astype(np.int64)
    else:
        lbl = np.asarray(labels).reshape(-1).astype(np.int64)

    if palette is None:
        palette = [
            (0.70, 0.70, 0.70),  # 0 = context — light grey
            (0.90, 0.20, 0.20),  # 1 = active  — bright red
        ]
    pal = np.asarray(palette, dtype=np.float32)
    pal_idx = np.clip(lbl, 0, len(pal) - 1)
    colors = pal[pal_idx]

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pts)
    pcd.colors = o3d.utility.Vector3dVector(colors)
    o3d.io.write_point_cloud(out_path, pcd)
    if verbose:
        print(f"  Saved -> {out_path}")

def invalid_to_zeros(arr, valid_mask, ndim=999):
    if valid_mask is not None:
        arr = arr.clone()
        arr[~valid_mask] = 0
        nnz = valid_mask.view(len(valid_mask), -1).sum(1)
    else:
        nnz = arr.numel() // len(arr) if len(arr) else 0  # number of point per image
    if arr.ndim > ndim:
        arr = arr.flatten(-2 - (arr.ndim - ndim), -2)
    return arr, nnz


def normalize_input(pts3d_src, valid_src, pts3d_trg, valid_trg, mode='none'):
    """Normalize the input points.

    Returns:
        pts3d_src_new, pts3d_trg_new, norm_params
        norm_params is a dict that can be passed to denormalize_output() to
        map predictions back to the original GT coordinate space.
    """
    if mode == 'none':
        return pts3d_src, pts3d_trg, {}

    elif 'median' in mode: # median_3 by default
        if mode == 'median':
            target_median = 1.0
        else:
            target_median = float(mode.split('_')[-1])

        pts3d_src_new = []
        pts3d_trg_new = []
        norm_factors = []

        for b in range(pts3d_src.shape[0]):
            src_xyz = pts3d_src[b]
            trg_xyz = pts3d_trg[b]
            src_valid = valid_src[b]
            trg_valid = valid_trg[b]

            nan_pts, nnz = invalid_to_zeros(trg_xyz, trg_valid, ndim=3)

            all_dis = nan_pts.norm(dim=-1)

            mean_factor = all_dis.sum() / (nnz.sum() + 1e-8)

            valid_dis = all_dis[trg_valid]
            norm_factor = valid_dis.median() if valid_dis.numel() > 0 else torch.tensor(1.0, device=all_dis.device)

            norm_factor = norm_factor.clip(min=0.01, max=100.0)

            src_xyz_norm = src_xyz / norm_factor * target_median
            trg_xyz_norm = trg_xyz / norm_factor * target_median

            src_xyz_norm = torch.clamp(src_xyz_norm, min=-1000.0, max=1000.0)
            trg_xyz_norm = torch.clamp(trg_xyz_norm, min=-1000.0, max=1000.0)

            pts3d_src_new.append(src_xyz_norm)
            pts3d_trg_new.append(trg_xyz_norm)
            norm_factors.append(norm_factor)

        pts3d_src_new = torch.stack(pts3d_src_new, dim=0)  # B, N, 3
        pts3d_trg_new = torch.stack(pts3d_trg_new, dim=0)  # B, N, 3
        norm_params = {
            'norm_factor': torch.stack(norm_factors, dim=0),  # (B,)
            'target_median': torch.tensor(target_median).repeat(pts3d_src.shape[0]),  # (B,)
        }
        return pts3d_src_new, pts3d_trg_new, norm_params

    elif 'cube' in mode:
        # not tested yet
        if mode == 'cube':
            target_scale = 1.0
        else:
            target_scale = float(mode.split('_')[-1])

        pts3d_src_new = []
        pts3d_trg_new = []
        centers = []
        max_dists = []

        for b in range(pts3d_src.shape[0]):
            src_xyz = pts3d_src[b]
            trg_xyz = pts3d_trg[b]
            src_valid = valid_src[b]
            trg_valid = valid_trg[b]

            center_trg = trg_xyz[trg_valid].mean(dim=0)

            src_xyz_centered = src_xyz - center_trg
            trg_xyz_centered = trg_xyz - center_trg

            dist_trg = torch.norm(trg_xyz_centered[trg_valid], dim=1)
            max_dist_trg = torch.quantile(dist_trg, 0.9)

            src_xyz_norm = src_xyz_centered / max_dist_trg * target_scale
            trg_xyz_norm = trg_xyz_centered / max_dist_trg * target_scale

            pts3d_src_new.append(src_xyz_norm)
            pts3d_trg_new.append(trg_xyz_norm)
            centers.append(center_trg)
            max_dists.append(max_dist_trg)

        pts3d_src = torch.stack(pts3d_src_new, dim=0)
        pts3d_trg = torch.stack(pts3d_trg_new, dim=0)
        norm_params = {
            'center': torch.stack(centers, dim=0),    # (B, 3)
            'max_dist': torch.stack(max_dists, dim=0),  # (B,)
            'target_scale': target_scale,
        }
        return pts3d_src, pts3d_trg, norm_params


def denormalize_output(pts, norm_factor, target_median, mode="median"):
    """Rescale model predictions back to the original GT coordinate space.

    Args:
        pts: (B, N, 3) normalized predictions
        norm_params: dict returned by normalize_input()

    Returns:
        (B, N, 3) predictions in the original GT space
    """
    if mode == 'median':
        # forward: pts_norm = pts_orig / norm_factor * target_median
        # inverse: pts_orig = pts_norm / target_median * norm_factor
        return pts / target_median * norm_factor

    elif mode == 'cube':
        # not tested yet
        raise NotImplementedError("denormalize_output for 'cube' mode not implemented yet.")


def process_point_cloud(pts, target_pts, qpos_str, w2c_first, norm_mode='median_3'):

    # Transform to first-camera space before normalisation
    pts = world_to_cam(pts, w2c_first)
    target_pts_cam = world_to_cam(target_pts, w2c_first)

    valid = torch.ones(pts.shape[0], dtype=torch.bool)
    gt_valid = torch.ones(target_pts_cam.shape[0], dtype=torch.bool)

    pts_unsqueezed, valid_unsqueezed = pts.unsqueeze(dim=0), valid.unsqueeze(dim=0)  # add batch dim
    target_pts_unsqueezed, gt_valid_unsqueezed = target_pts_cam.unsqueeze(dim=0), gt_valid.unsqueeze(dim=0)  # add batch dim

    pts_normed, target_pts_normed, norm_params = normalize_input(
        pts_unsqueezed,
        valid_unsqueezed,
        target_pts_unsqueezed,
        gt_valid_unsqueezed,
        mode=norm_mode)

    pts_normed = pts_normed.squeeze(0)  # remove batch dim
    target_pts_normed = target_pts_normed.squeeze(0)  # remove batch dim

    return target_pts_normed, target_pts_normed, target_pts_cam, norm_params

def world_to_cam(pts, w2c_first):
    """Transform (N, 3) points from world space to first-camera space."""
    ones = torch.ones(pts.shape[0], 1, dtype=pts.dtype, device=pts.device)
    pts_h = torch.cat([pts, ones], dim=-1)  # [N, 4]
    return (w2c_first @ pts_h.T).T[:, :3]   # [N, 3]

def cam_to_world(pts, c2w_first):
    """Transform (N, 3) points from first-camera space to world space."""
    # print(pts.shape, c2w_first.shape) # B N 3, B 4 4
    ones = torch.ones(pts.shape[0], pts.shape[1], 1, dtype=pts.dtype, device=pts.device) # B N 1
    pts_h = torch.cat([pts, ones], dim=-1)  # [B, N, 4]
    return (c2w_first @ pts_h.transpose(1, 2)).transpose(1, 2)[:, :, :3]   # [B, N, 3]

def outlier_filtering(pred_xyz, nb_neighbors: int = 50, std_ratio: float = 4.0):
    """Remove statistical outliers from predicted point clouds and return inlier mask."""
    B, N, C = pred_xyz.shape
    outlier_mask = torch.zeros((B, N), dtype=torch.bool, device=pred_xyz.device)

    for b in range(B):
        pts_masked = pred_xyz[b].cpu().numpy()

        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(
            pts_masked.reshape(-1, 3)
        )
        cl, ind = pcd.remove_statistical_outlier(
            nb_neighbors=nb_neighbors, std_ratio=std_ratio
        )
        outlier_mask[b, ind] = True

    return outlier_mask

@torch.no_grad()
def get_joint_pointcloud_center_scale(pts, valid_masks=None, z_only=False, center=True):
    # pts: [B, N, 3], valid_masks: [B, N]
    
    if valid_masks is not None:
        # Set invalid points to NaN
        _pts = pts.clone()
        _pts[~valid_masks] = float('nan')
    else:
        _pts = pts

    # compute median center
    _center = torch.nanmedian(_pts, dim=1, keepdim=True).values  # (B, 1, 3)
    if z_only:
        _center[..., :2] = 0  # do not center X and Y

    # compute median norm
    _norm = ((_pts - _center) if center else _pts).norm(dim=-1)
    scale = torch.nanmedian(_norm, dim=1).values
    return _center, scale

def scale_shift_alignment_chamfer(pred_xyz, gt_xyz, gt_mask=None, max_iterations=100, lr=0.01, num_sample=None, return_transform=False, generator=None):
    """
    Align pred_xyz to gt_xyz using gradient descent to minimize chamfer distance.
    
    Args:
        pred_xyz: [B, N, 3] predicted point cloud
        gt_xyz: [B, N, 3] ground truth point cloud  
        gt_mask: [B, N] validity mask for ground truth points (optional)
        max_iterations: maximum number of optimization iterations
        lr: learning rate for optimization
    
    Returns:
        aligned_pred_xyz: [B, N, 3] aligned predicted point cloud
        final_scale: [B] final scale values
        final_shift: [B, 3] final shift values
    """
    B, N_pred, C = pred_xyz.shape
    B, N_gt, C = gt_xyz.shape
    device = pred_xyz.device
    
    # Initialize parameters to optimize
    scale = torch.ones(B, 1, 1, device=device, requires_grad=True)
    shift = torch.zeros(B, 1, 3, device=device, requires_grad=True)
    
    # Setup optimizer
    optimizer = torch.optim.Adam([scale, shift], lr=lr)
    
    best_loss = float('inf')
    best_scale = scale.clone()
    best_shift = shift.clone()
    
    target_xyz = gt_xyz.clone().detach()
    target_xyz.requires_grad = True
    source_xyz = pred_xyz.clone().detach()
    source_xyz.requires_grad = True
    if num_sample is None:
        num_sample = max(N_pred, N_gt) // 4

    for i in range(max_iterations):
        optimizer.zero_grad()
        
        # Apply transformation: aligned = scale * pred + shift
        aligned_pred = scale * source_xyz + shift

        # Compute chamfer distance
        # randomly down sample
        idx_pred = torch.randint(0, N_pred, (B, min(N_pred, num_sample)), device=device, generator=generator)
        idx_gt = torch.randint(0, N_gt, (B, min(N_gt, num_sample)), device=device, generator=generator)

        if num_sample < N_pred:
            aligned_pred_sampled = aligned_pred[:, idx_pred[0], :]
        else:
            aligned_pred_sampled = aligned_pred
        if num_sample < N_gt:
            target_xyz_sampled = target_xyz[:, idx_gt[0], :]
        else:
            target_xyz_sampled = target_xyz

        cd_loss, _ = chamfer_distance(aligned_pred_sampled, target_xyz_sampled, batch_reduction='mean')

        # Backpropagation
        cd_loss.backward()
        optimizer.step()
        
        # Clamp scale to reasonable bounds
        with torch.no_grad():
            scale.clamp_(0.01, 100.0)
        
        # Track best solution
        if cd_loss.item() < best_loss:
            best_loss = cd_loss.item()
            best_scale = scale.clone()
            best_shift = shift.clone()
        
    # Apply best transformation
    with torch.no_grad():
        aligned_pred_xyz = best_scale * pred_xyz + best_shift
    
    if return_transform:
        return aligned_pred_xyz, best_shift.detach(), best_scale.detach()
    else:
        return aligned_pred_xyz


def scale_shift_alignment_pointcloud(
    pred_xyz,
    gt_xyz,
    gt_mask,
    num_sample=None,
    filter_outliers=None,
    *,
    filter_before: bool = False,
    filter_after: bool = False,
    outlier_nb_neighbors: int = 50,
    outlier_std_ratio: float = 4.0,
    cham_iters: int = 100,
    cham_lr: float = 0.01,
    cham_num_sample=None,
    seed: int = 0,
    return_transform: bool = False,
):
    '''
    Scale-shift alignment of pred to gt: median-stat init → Adam-on-chamfer →
    optional statistical outlier filter (placement controlled by
    filter_before / filter_after).

    pred, gt: B N 3
    mask: B N

    Backward-compat: ``filter_outliers`` is a deprecated alias for
    ``filter_after``. ``num_sample`` still controls the chamfer Adam subset
    size when ``cham_num_sample`` is None.
    '''
    if filter_outliers is not None:
        filter_after = bool(filter_outliers)
    if cham_num_sample is None:
        cham_num_sample = num_sample

    if filter_before:
        pre_mask = outlier_filtering(
            pred_xyz, nb_neighbors=outlier_nb_neighbors, std_ratio=outlier_std_ratio,
        )
        # Loader uses B=1 at eval time; keep that contract.
        pred_xyz = pred_xyz[pre_mask].unsqueeze(0)

    gt_xyz_raw = gt_xyz.clone()

    gt_center, gt_scale = get_joint_pointcloud_center_scale(gt_xyz, gt_mask)

    pred_center, pred_scale = get_joint_pointcloud_center_scale(pred_xyz)

    pred_scale = pred_scale.clip(min=1e-3, max=1e3)

    # Median-stat init: pred_init = scale_init * pred + shift_init
    # ``pred_center`` and ``gt_center`` are already [B, 1, 3] from
    # ``get_joint_pointcloud_center_scale``; ``scale_init`` is [B, 1, 1]
    # and broadcasts cleanly against them.
    scale_init = (gt_scale / pred_scale).view(-1, 1, 1)            # [B, 1, 1]
    shift_init = -scale_init * pred_center + gt_center             # [B, 1, 3]

    pred_xyz = scale_init * pred_xyz + shift_init

    gen = torch.Generator(device=pred_xyz.device)
    gen.manual_seed(int(seed))
    if return_transform:
        pred_xyz_new, shift_adam, scale_adam = scale_shift_alignment_chamfer(
            pred_xyz, gt_xyz_raw, gt_mask,
            max_iterations=cham_iters, lr=cham_lr, num_sample=cham_num_sample,
            generator=gen, return_transform=True,
        )
        # Compose: pred_final = scale_adam * (scale_init * pred + shift_init) + shift_adam
        scale_total = scale_adam * scale_init                       # [B, 1, 1]
        shift_total = scale_adam * shift_init + shift_adam          # [B, 1, 3]
    else:
        pred_xyz_new = scale_shift_alignment_chamfer(
            pred_xyz, gt_xyz_raw, gt_mask,
            max_iterations=cham_iters, lr=cham_lr, num_sample=cham_num_sample,
            generator=gen,
        )
        scale_total = None
        shift_total = None

    if filter_after:
        pred_mask = outlier_filtering(
            pred_xyz_new,
            nb_neighbors=outlier_nb_neighbors,
            std_ratio=outlier_std_ratio,
        )
    else:
        pred_mask = torch.ones(
            (pred_xyz_new.shape[0], pred_xyz_new.shape[1]),
            dtype=torch.bool, device=pred_xyz_new.device,
        )

    if return_transform:
        return pred_xyz_new, pred_mask, scale_total, shift_total
    return pred_xyz_new, pred_mask