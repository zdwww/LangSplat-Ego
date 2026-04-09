"""
Co-visibility based frame selection for 3DGS scenes.

Computes a visibility matrix (which cameras see which Gaussians) and uses
greedy set cover to select a small set of frames with maximal Gaussian coverage.

Adapted from FlashSplat-variants/SceneSplat/covisibility.py, but operates on
CamInfo NamedTuples directly to avoid the image-loading overhead of constructing
full Camera objects.
"""

import torch
import numpy as np
from typing import List, Tuple, Dict

from utils.graphics_utils import getWorld2View2, getProjectionMatrix


def _build_full_proj_transforms(cam_infos) -> torch.Tensor:
    """
    Build (N_cams, 4, 4) full projection matrices from CamInfo fields.

    Uses the same math as scene/cameras.py lines 99-101:
        w2v = getWorld2View2(R, T).T
        proj = getProjectionMatrix(znear, zfar, FovX, FovY).T
        full_proj = w2v @ proj   (bmm convention)
    """
    projs = []
    for ci in cam_infos:
        w2v = torch.tensor(
            getWorld2View2(ci.R, ci.T), dtype=torch.float32
        ).transpose(0, 1)
        proj = getProjectionMatrix(
            znear=0.01, zfar=100.0, fovX=ci.FovX, fovY=ci.FovY
        ).transpose(0, 1)
        full = w2v.unsqueeze(0).bmm(proj.unsqueeze(0)).squeeze(0)
        projs.append(full)
    return torch.stack(projs, dim=0)  # (N_cams, 4, 4)


def compute_visibility_matrix(
    cam_infos,
    gaussian_xyz: torch.Tensor,
    subsample: int = 4,
    batch_size: int = 32,
) -> torch.Tensor:
    """
    For each camera, determine which (subsampled) Gaussians fall within its
    view frustum via projection + NDC bounds check.

    Args:
        cam_infos: list of CamInfo NamedTuples (R, T, FovX, FovY fields used).
        gaussian_xyz: (N_gs, 3) tensor of Gaussian positions.
        subsample: keep every Nth Gaussian to reduce memory.
        batch_size: cameras processed per GPU batch.

    Returns:
        (N_cams, N_gs_sub) bool tensor on CPU.
    """
    # Subsample Gaussians
    xyz_sub = gaussian_xyz[::subsample].detach()
    N_gs_sub = xyz_sub.shape[0]
    N_cams = len(cam_infos)

    # Build homogeneous coords on CUDA
    ones = torch.ones(N_gs_sub, 1, device="cuda", dtype=torch.float32)
    xyz_hom = torch.cat([xyz_sub.cuda().float(), ones], dim=1)  # (N_gs_sub, 4)

    # Build projection matrices
    full_projs = _build_full_proj_transforms(cam_infos).cuda()  # (N_cams, 4, 4)

    # Allocate result on CPU
    visibility = torch.zeros(N_cams, N_gs_sub, dtype=torch.bool)

    for start in range(0, N_cams, batch_size):
        end = min(start + batch_size, N_cams)
        B = end - start
        proj_batch = full_projs[start:end]  # (B, 4, 4)

        # Project: (B, N_gs_sub, 4) = (1, N_gs_sub, 4) @ (B, 4, 4)
        clip = xyz_hom.unsqueeze(0).expand(B, -1, -1) @ proj_batch  # (B, N_gs_sub, 4)

        w = clip[:, :, 3:4]  # (B, N_gs_sub, 1)
        ndc = clip[:, :, :2] / (w + 1e-8)  # (B, N_gs_sub, 2)

        visible = (
            (ndc[:, :, 0].abs() < 1.0)
            & (ndc[:, :, 1].abs() < 1.0)
            & (w.squeeze(-1) > 0.01)
        )  # (B, N_gs_sub)

        visibility[start:end] = visible.cpu()

    return visibility


def select_representative_frames(
    visibility: torch.Tensor,
    n_frames: int = 100,
) -> Tuple[List[int], Dict]:
    """
    Greedy set cover: iteratively pick the camera that covers the most
    uncovered Gaussians. Stops early when coverage gain drops to 0.

    Args:
        visibility: (N_cams, N_gs_sub) bool tensor.
        n_frames: maximum number of frames to select.

    Returns:
        (selected_indices, stats) where stats has coverage_pct,
        covered_gaussians, total_gaussians.
    """
    N_cams, N_gs = visibility.shape
    n_frames = min(n_frames, N_cams)

    covered = torch.zeros(N_gs, dtype=torch.bool)
    selected: List[int] = []
    remaining = list(range(N_cams))

    for _ in range(n_frames):
        if not remaining:
            break

        # Vectorized gain computation
        remaining_t = torch.tensor(remaining, dtype=torch.long)
        gains = (visibility[remaining_t] & ~covered).sum(dim=1)
        best_local = gains.argmax().item()
        best_gain = gains[best_local].item()

        if best_gain == 0:
            break

        best_idx = remaining[best_local]
        selected.append(best_idx)
        covered |= visibility[best_idx]
        remaining.pop(best_local)

    covered_count = covered.sum().item()
    coverage_pct = covered_count / N_gs * 100 if N_gs > 0 else 0.0

    stats = {
        "coverage_pct": round(coverage_pct, 2),
        "covered_gaussians": covered_count,
        "total_gaussians_subsampled": N_gs,
        "num_selected": len(selected),
    }

    return selected, stats


def _covis_one_vs_all(
    visibility: torch.Tensor,
    idx: int,
    cam_counts: torch.Tensor,
) -> torch.Tensor:
    """
    Compute IoU between camera `idx` and all cameras.

    Args:
        visibility: (N_cams, N_gs_sub) bool tensor on CPU.
        idx: index of the reference camera.
        cam_counts: (N_cams,) float tensor — precomputed per-camera visible counts.

    Returns:
        (N_cams,) float tensor of IoU values.
    """
    intersection = (visibility & visibility[idx]).sum(dim=1).float()
    union = cam_counts + cam_counts[idx] - intersection
    return intersection / union.clamp(min=1)


def select_frames_coverage_diversity(
    visibility: torch.Tensor,
    diversity_threshold: float = 0.5,
    max_frames: int = 200,
) -> Tuple[List[int], Dict]:
    """
    Two-phase frame selection: greedy set cover for coverage, then
    farthest-point diversity sampling using pairwise covisibility IoU.

    Phase 1: Greedy set cover until no new Gaussians are covered.
    Phase 2: Iteratively select the camera with lowest max-IoU to any
             already-selected camera (most novel viewpoint). Stops when
             the minimum max-IoU exceeds diversity_threshold.

    Args:
        visibility: (N_cams, N_gs_sub) bool tensor.
        diversity_threshold: stop phase 2 when every remaining camera
            shares > this IoU with some selected camera. Higher = more frames.
        max_frames: hard cap on total selected frames.

    Returns:
        (selected_indices, stats) where stats includes phase1/phase2 counts.
    """
    N_cams, N_gs = visibility.shape

    # Precompute per-camera visible Gaussian counts
    cam_counts = visibility.sum(dim=1).float()  # (N_cams,)

    # ---- Phase 1: greedy set cover ----
    covered = torch.zeros(N_gs, dtype=torch.bool)
    selected: List[int] = []
    remaining_set = set(range(N_cams))

    while len(selected) < max_frames and remaining_set:
        remaining_list = list(remaining_set)
        remaining_t = torch.tensor(remaining_list, dtype=torch.long)
        gains = (visibility[remaining_t] & ~covered).sum(dim=1)
        best_local = gains.argmax().item()
        best_gain = gains[best_local].item()

        if best_gain == 0:
            break

        best_idx = remaining_list[best_local]
        selected.append(best_idx)
        covered |= visibility[best_idx]
        remaining_set.discard(best_idx)

    phase1_count = len(selected)
    covered_count = covered.sum().item()
    coverage_pct = covered_count / N_gs * 100 if N_gs > 0 else 0.0

    # ---- Phase 2: farthest-point diversity sampling via IoU ----
    # max_sim[i] = max IoU between camera i and any selected camera
    max_sim = torch.full((N_cams,), -1.0)

    # Initialize max_sim from phase 1 selections
    for idx in selected:
        iou_row = _covis_one_vs_all(visibility, idx, cam_counts)
        max_sim = torch.maximum(max_sim, iou_row)

    # Mark selected cameras so they're never picked again
    for idx in selected:
        max_sim[idx] = float('inf')

    final_min_max_sim = 0.0

    while len(selected) < max_frames:
        candidate = max_sim.argmin().item()
        candidate_sim = max_sim[candidate].item()

        if candidate_sim > diversity_threshold:
            final_min_max_sim = candidate_sim
            break

        final_min_max_sim = candidate_sim
        selected.append(candidate)
        remaining_set.discard(candidate)
        max_sim[candidate] = float('inf')

        # Update max_sim with new camera's IoU
        iou_row = _covis_one_vs_all(visibility, candidate, cam_counts)
        max_sim = torch.maximum(max_sim, iou_row)
        # Re-mark all selected as inf
        for idx in selected:
            max_sim[idx] = float('inf')

    phase2_count = len(selected) - phase1_count

    stats = {
        "coverage_pct": round(coverage_pct, 2),
        "covered_gaussians": covered_count,
        "total_gaussians_subsampled": N_gs,
        "num_selected": len(selected),
        "phase1_frames": phase1_count,
        "phase2_frames": phase2_count,
        "diversity_threshold": diversity_threshold,
        "final_min_max_sim": round(final_min_max_sim, 4),
    }

    return selected, stats
