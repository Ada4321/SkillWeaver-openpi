"""Offline / online action-quality metrics for pi0 trained on IsaacLab data.

All computation is pure numpy — no JAX, no CUDA.  The caller is responsible
for converting JAX arrays to numpy before calling these functions.

Typical call flow (train.py at save_interval):

    pred_norm = jax.device_get(psample_step(rng, state, batch))  # (B,H,D)
    gt_norm   = jax.device_get(batch[1])                          # (B,H,D)
    obs_state = jax.device_get(batch[0].state)                    # (B,S)

    act_stats   = data_config.norm_stats["actions"]
    state_stats = data_config.norm_stats.get("state")

    pred_phys = denormalize(pred_norm, act_stats)
    gt_phys   = denormalize(gt_norm,   act_stats)
    obs_phys  = denormalize(obs_state, state_stats) if state_stats else obs_state

    per_sample = compute_eef_errors(pred_phys, gt_phys, obs_phys, ...)
    metrics    = summarize_eef_metrics(**per_sample)
"""

from __future__ import annotations

import numpy as np

from openpi.eval.robot_fk import check_arm_dim, get_kinematics


# ---------------------------------------------------------------------------
# Normalization helpers
# ---------------------------------------------------------------------------


def denormalize(x_norm: np.ndarray, norm_stats, *, use_quantiles: bool = False) -> np.ndarray:
    """Denormalize back to physical units. Must match the forward normalization
    used by the data pipeline (openpi.transforms.Normalize); mixing quantile and
    z-score silently yields a ~2x scale/offset error on the EEF absolute poses."""
    x_norm = np.asarray(x_norm)
    if use_quantiles:
        q01 = np.asarray(norm_stats.q01)
        q99 = np.asarray(norm_stats.q99)
        return (x_norm + 1.0) / 2.0 * (q99 - q01 + 1e-6) + q01
    return x_norm * (np.asarray(norm_stats.std) + 1e-6) + np.asarray(norm_stats.mean)


# ---------------------------------------------------------------------------
# Rotation helpers
# ---------------------------------------------------------------------------


def _aa_to_rot(aa: np.ndarray) -> np.ndarray:
    """Axis-angle → rotation matrix.

    Args:
        aa: (..., 3)
    Returns:
        (..., 3, 3)
    """
    shape = aa.shape[:-1]
    aa = aa.reshape(-1, 3).astype(np.float64)
    angle = np.linalg.norm(aa, axis=-1)       # (N,)
    safe_angle = np.where(angle > 1e-9, angle, 1.0)
    axis = aa / safe_angle[:, None]            # (N, 3)

    c = np.cos(angle)
    s = np.sin(angle)
    t = 1.0 - c

    nx, ny, nz = axis[:, 0], axis[:, 1], axis[:, 2]
    R = np.stack(
        [
            np.stack([t * nx * nx + c,       t * nx * ny - s * nz, t * nx * nz + s * ny], axis=-1),
            np.stack([t * nx * ny + s * nz,  t * ny * ny + c,      t * ny * nz - s * nx], axis=-1),
            np.stack([t * nx * nz - s * ny,  t * ny * nz + s * nx, t * nz * nz + c     ], axis=-1),
        ],
        axis=-2,
    )  # (N, 3, 3)
    return R.reshape(*shape, 3, 3)


def _rot_to_aa(R: np.ndarray) -> np.ndarray:
    """Rotation matrix → axis-angle.

    Args:
        R: (..., 3, 3)
    Returns:
        (..., 3)
    """
    shape = R.shape[:-2]
    R = R.reshape(-1, 3, 3).astype(np.float64)
    trace = R[:, 0, 0] + R[:, 1, 1] + R[:, 2, 2]
    angle = np.arccos(np.clip((trace - 1.0) / 2.0, -1.0, 1.0))  # (N,)

    # axis from skew-symmetric part: sin(θ)*[n]× = (R - Rᵀ)/2
    skew = (R - R.transpose(0, 2, 1)) / 2.0
    axis = np.stack([skew[:, 2, 1], skew[:, 0, 2], skew[:, 1, 0]], axis=-1)  # (N, 3)
    norm = np.linalg.norm(axis, axis=-1, keepdims=True)
    axis = axis / np.where(norm > 1e-9, norm, 1.0)

    return (axis * angle[:, None]).reshape(*shape, 3)


# ---------------------------------------------------------------------------
# Delta integration helpers
# ---------------------------------------------------------------------------


def _integrate_delta_ee(
    delta_pos: np.ndarray,
    delta_aa: np.ndarray,
    pos0: np.ndarray,
    aa0: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Integrate EE deltas from initial pose.

    Args:
        delta_pos: (B, H, 3)  incremental position deltas
        delta_aa:  (B, H, 3)  incremental axis-angle rotation deltas
        pos0:      (B, 3)     initial EE position
        aa0:       (B, 3)     initial EE axis-angle

    Returns:
        pos_abs: (B, H, 3)  absolute positions
        aa_abs:  (B, H, 3)  absolute axis-angles (as rotation matrices composed)
    """
    pos_abs = np.cumsum(delta_pos, axis=1) + pos0[:, None, :]

    B, H = delta_aa.shape[:2]
    aa_abs = np.zeros_like(delta_aa)
    R_cur = _aa_to_rot(aa0)  # (B, 3, 3)
    for t in range(H):
        R_delta = _aa_to_rot(delta_aa[:, t])   # (B, 3, 3)
        R_cur = np.matmul(R_delta, R_cur)       # R_t = R_Δt @ R_{t-1}
        aa_abs[:, t] = _rot_to_aa(R_cur)

    return pos_abs, aa_abs


# ---------------------------------------------------------------------------
# Main metric functions
# ---------------------------------------------------------------------------


def compute_eef_errors(
    pred_actions_phys: np.ndarray,
    gt_actions_phys: np.ndarray,
    obs_state_phys: np.ndarray,
    *,
    arm_repr: str,
    arm_dim: int,
    state_arm_dim: int,
    arm_is_delta: bool,
    gripper_1d: bool,
    replace_gripper_with_binary: bool = False,
    robot: str | None = None,
) -> dict[str, np.ndarray]:
    """Compute per-timestep EEF and gripper errors.

    Args:
        pred_actions_phys: (B, H, D) denormalized predicted actions.
        gt_actions_phys:   (B, H, D) denormalized ground-truth actions.
        obs_state_phys:    (B, S)    denormalized initial observation state.
        arm_repr:          "joint" or "ee".
        arm_dim:           number of arm columns: the robot's joint count for
                           joint repr (7 panda / 6 widowx), 6 for ee repr.
        state_arm_dim:     arm dims in obs_state (joint count for joint state, 6 for ee state).
        arm_is_delta:      True when actions are incremental (cmd or make_target_residual).
        gripper_1d:        True when gripper is a single scalar.
        replace_gripper_with_binary:
            True when training/eval is in binary-gripper mode (GT gripper dims are ±1).
            False when gripper dims are continuous targets and binary intent must be
            derived as sign(action - state).
        robot:             embodiment key from the shard's dataset_info.json
                           ("panda" / "widowx"); None means panda.

    Returns:
        dict with keys "pos_err", "rot_err", "gripper_err", "gripper_correct",
        "violated", each a 1-D array of shape (B*H,).  "gripper_correct" is bool.
    """
    kin = get_kinematics(robot)
    check_arm_dim(robot, arm_dim, arm_repr=arm_repr)
    pred_actions_phys = np.asarray(pred_actions_phys, dtype=np.float64)
    gt_actions_phys   = np.asarray(gt_actions_phys,   dtype=np.float64)
    obs_state_phys    = np.asarray(obs_state_phys,     dtype=np.float64)

    B, H, _ = pred_actions_phys.shape

    pred_arm = pred_actions_phys[:, :, :arm_dim]   # (B, H, arm_dim)
    gt_arm   = gt_actions_phys[:, :, :arm_dim]     # (B, H, arm_dim)
    pred_g   = pred_actions_phys[:, :, arm_dim:]   # (B, H, gripper_dim)
    gt_g     = gt_actions_phys[:, :, arm_dim:]     # (B, H, gripper_dim)

    obs_arm = obs_state_phys[:, :state_arm_dim]    # (B, state_arm_dim)

    violated_bh = np.zeros((B, H), dtype=bool)

    # ------------------------------------------------------------------
    # Convert arm predictions and GT to absolute EEF (pos, rot_mat)
    # ------------------------------------------------------------------
    if arm_repr == "joint":
        nq = kin.n_joints
        if arm_is_delta:
            # obs_arm provides initial joint angles (state_arm_dim == nq)
            q0 = obs_arm[:, :nq]
            pred_q = np.cumsum(pred_arm, axis=1) + q0[:, None, :]  # (B, H, nq)
            gt_q   = np.cumsum(gt_arm,   axis=1) + q0[:, None, :]  # (B, H, nq)
        else:
            pred_q = pred_arm  # (B, H, nq)
            gt_q   = gt_arm    # (B, H, nq)

        # Check and clamp joint violations (predictions only)
        violated_bh = np.any(
            (pred_q < kin.joint_lower[None, None, :]) | (pred_q > kin.joint_upper[None, None, :]),
            axis=-1,
        )  # (B, H)
        pred_q_clamped = np.clip(pred_q, kin.joint_lower, kin.joint_upper)

        pos_pred, R_pred = kin.fk(pred_q_clamped.reshape(B * H, nq))
        pos_gt,   R_gt   = kin.fk(gt_q.reshape(B * H, nq))

    elif arm_repr == "ee":
        if arm_is_delta:
            # state_arm_dim == kin.n_joints means the state holds joint angles;
            # 6 means it is already [pos(3), axis_angle(3)]. For a 6-DoF arm the two
            # are the same width, so joint state is only assumed when the robot has
            # more joints than an EE pose has components.
            if state_arm_dim == kin.n_joints and kin.n_joints > 6:
                # State has joint angles → FK to get initial EE pose
                pos0, R0 = kin.fk(obs_arm[:, : kin.n_joints])
                aa0 = _rot_to_aa(R0)                # (B, 3)
            else:
                # State has EE pose [pos(3), aa(3)]
                pos0 = obs_arm[:, :3]   # (B, 3)
                aa0  = obs_arm[:, 3:6]  # (B, 3)

            pred_pos, pred_aa = _integrate_delta_ee(
                pred_arm[:, :, :3], pred_arm[:, :, 3:6], pos0, aa0
            )
            gt_pos, gt_aa = _integrate_delta_ee(
                gt_arm[:, :, :3], gt_arm[:, :, 3:6], pos0, aa0
            )
        else:
            pred_pos = pred_arm[:, :, :3]   # (B, H, 3)
            pred_aa  = pred_arm[:, :, 3:6]  # (B, H, 3)
            gt_pos   = gt_arm[:, :, :3]
            gt_aa    = gt_arm[:, :, 3:6]

        pos_pred = pred_pos.reshape(B * H, 3)
        pos_gt   = gt_pos.reshape(B * H, 3)
        R_pred   = _aa_to_rot(pred_aa.reshape(B * H, 3))
        R_gt     = _aa_to_rot(gt_aa.reshape(B * H, 3))

    else:
        raise ValueError(f"Unknown arm_repr: {arm_repr!r}")

    # ------------------------------------------------------------------
    # Position error (L2, metres)
    # ------------------------------------------------------------------
    pos_err = np.linalg.norm(pos_pred - pos_gt, axis=-1)  # (B*H,)

    # ------------------------------------------------------------------
    # Rotation error (geodesic distance, radians)
    # ------------------------------------------------------------------
    R_rel = np.matmul(R_pred.transpose(0, 2, 1), R_gt)    # (B*H, 3, 3)
    trace = R_rel[:, 0, 0] + R_rel[:, 1, 1] + R_rel[:, 2, 2]
    rot_err = np.arccos(np.clip((trace - 1.0) / 2.0, -1.0, 1.0))  # (B*H,)

    # ------------------------------------------------------------------
    # Gripper error (mean |pred - gt|, in storage-domain units)
    # ------------------------------------------------------------------
    pred_g_flat = pred_g.reshape(B * H, -1)
    gt_g_flat   = gt_g.reshape(B * H, -1)
    gripper_err = np.mean(np.abs(pred_g_flat - gt_g_flat), axis=-1)  # (B*H,)

    # ------------------------------------------------------------------
    # Gripper binary classification accuracy (open=-1 vs close=+1).
    #
    #   replace_gripper_with_binary=True : GT gripper dims are already ±1
    #       (convert+backfill produced them with eps+carry-forward), pred is
    #       continuous. Threshold at 0 to recover ±1 on both sides.
    #   replace_gripper_with_binary=False: GT gripper dims are continuous
    #       targets. Recover intent as sign(action - state). Uses state at
    #       chunk t=0 (broadcast across H) — exact at t=0, approximate later
    #       in the chunk; bias is shared by pred and gt so accuracy is
    #       still meaningful.
    # ------------------------------------------------------------------
    gripper_dim = pred_g_flat.shape[-1]
    if replace_gripper_with_binary:
        pred_signal = pred_g_flat
        gt_signal   = gt_g_flat
    else:
        state_g0 = obs_state_phys[:, state_arm_dim : state_arm_dim + gripper_dim]  # (B, gripper_dim)
        state_g  = np.broadcast_to(state_g0[:, None, :], (B, H, gripper_dim)).reshape(B * H, -1)
        pred_signal = pred_g_flat - state_g
        gt_signal   = gt_g_flat   - state_g

    pred_binary = np.where(pred_signal >= 0.0, 1.0, -1.0)
    gt_binary   = np.where(gt_signal   >= 0.0, 1.0, -1.0)
    gripper_correct = np.all(pred_binary == gt_binary, axis=-1)  # (B*H,) bool

    return {
        "pos_err":         pos_err,
        "rot_err":         rot_err,
        "gripper_err":     gripper_err,
        "gripper_correct": gripper_correct,
        "violated":        violated_bh.reshape(B * H),
    }


def extract_ee_positions(
    pred_actions_phys: np.ndarray,   # (B, H, D)
    gt_actions_phys:   np.ndarray,   # (B, H, D)
    obs_state_phys:    np.ndarray,   # (B, S)
    *,
    arm_repr:      str,
    arm_dim:       int,
    state_arm_dim: int,
    arm_is_delta:  bool,
    robot:         str | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Return pos_pred, pos_gt each (B, H, 3) in the robot-base frame.

    Mirrors the arm→EE block of compute_eef_errors but preserves (B, H, 3) shape.
    """
    kin = get_kinematics(robot)
    check_arm_dim(robot, arm_dim, arm_repr=arm_repr)
    pred_actions_phys = np.asarray(pred_actions_phys, dtype=np.float64)
    gt_actions_phys   = np.asarray(gt_actions_phys,   dtype=np.float64)
    obs_state_phys    = np.asarray(obs_state_phys,     dtype=np.float64)

    B, H, _ = pred_actions_phys.shape
    pred_arm = pred_actions_phys[:, :, :arm_dim]
    gt_arm   = gt_actions_phys[:, :, :arm_dim]
    obs_arm  = obs_state_phys[:, :state_arm_dim]

    if arm_repr == "joint":
        nq = kin.n_joints
        if arm_is_delta:
            q0 = obs_arm[:, :nq]
            pred_q = np.cumsum(pred_arm, axis=1) + q0[:, None, :]
            gt_q   = np.cumsum(gt_arm,   axis=1) + q0[:, None, :]
        else:
            pred_q = pred_arm
            gt_q   = gt_arm
        pred_q_clamped = np.clip(pred_q, kin.joint_lower, kin.joint_upper)
        pos_pred, _ = kin.fk(pred_q_clamped.reshape(B * H, nq))
        pos_gt,   _ = kin.fk(gt_q.reshape(B * H, nq))
    elif arm_repr == "ee":
        if arm_is_delta:
            # See compute_eef_errors for why this is == and not >=.
            if state_arm_dim == kin.n_joints and kin.n_joints > 6:
                pos0, R0 = kin.fk(obs_arm[:, : kin.n_joints])
                aa0 = _rot_to_aa(R0)
            else:
                pos0 = obs_arm[:, :3]
                aa0  = obs_arm[:, 3:6]
            pred_pos, _ = _integrate_delta_ee(pred_arm[:, :, :3], pred_arm[:, :, 3:6], pos0, aa0)
            gt_pos,   _ = _integrate_delta_ee(gt_arm[:, :, :3],   gt_arm[:, :, 3:6],   pos0, aa0)
        else:
            pred_pos = pred_arm[:, :, :3]
            gt_pos   = gt_arm[:, :, :3]
        pos_pred = pred_pos.reshape(B * H, 3)
        pos_gt   = gt_pos.reshape(B * H, 3)
    else:
        raise ValueError(f"Unknown arm_repr: {arm_repr!r}")

    return pos_pred.reshape(B, H, 3).astype(np.float32), pos_gt.reshape(B, H, 3).astype(np.float32)


def _plot_ee_traj_3d(
    pos_pred: np.ndarray,  # (N, H, 3)
    pos_gt:   np.ndarray,  # (N, H, 3)
) -> bytes:
    """Render 3D EE trajectories (pred vs GT) as a PNG, one subplot per episode."""
    import io
    import math
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    N = pos_pred.shape[0]
    ncols = min(N, 4)
    nrows = math.ceil(N / ncols)

    fig = plt.figure(figsize=(4 * ncols, 3.5 * nrows))
    elev, azim = 25, -60

    for i in range(N):
        ax = fig.add_subplot(nrows, ncols, i + 1, projection="3d")
        pg, pp = pos_gt[i], pos_pred[i]   # (H, 3)

        ax.plot(pg[:, 0], pg[:, 1], pg[:, 2], color="blue", lw=1.5, label="GT")
        ax.plot(pp[:, 0], pp[:, 1], pp[:, 2], color="red",  lw=1.5, ls="--", label="Pred")

        # Start marker (filled) and end marker (open)
        for pos, clr in ((pg, "blue"), (pp, "red")):
            ax.scatter(*pos[0],  c=clr, s=30, zorder=5)
            ax.scatter(*pos[-1], c=clr, s=30, zorder=5, facecolors="none", edgecolors=clr)

        # Force equal 3-axis extent so inter-trajectory distances look visually
        # proportional.  Floor at 5 cm half-extent (= 10 cm box) so tiny true
        # errors don't get visually amplified by matplotlib's auto-scale.
        all_pts = np.concatenate([pg, pp], axis=0)
        center = all_pts.mean(axis=0)
        half = max((all_pts.max(axis=0) - all_pts.min(axis=0)).max() / 2.0, 0.05)
        ax.set_xlim(center[0] - half, center[0] + half)
        ax.set_ylim(center[1] - half, center[1] + half)
        ax.set_zlim(center[2] - half, center[2] + half)
        try:
            ax.set_box_aspect((1, 1, 1))  # matplotlib 3.3+
        except Exception:
            pass

        mean_err = float(np.mean(np.linalg.norm(pp - pg, axis=-1)))
        ax.set_title(f"ep {i}  Δpos={mean_err:.3f}m", fontsize=8)
        ax.set_xlabel("X"); ax.set_ylabel("Y"); ax.set_zlabel("Z")
        ax.view_init(elev=elev, azim=azim)
        if i == 0:
            ax.legend(fontsize=7)

    fig.tight_layout()
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=100)
    plt.close(fig)
    buf.seek(0)
    return buf.read()


def _plot_ee_traj_front(
    pos_pred:       np.ndarray,         # (N, H, 3)  panda_link0 frame
    pos_gt:         np.ndarray,          # (N, H, 3)
    front_images:   np.ndarray,          # (N, H_img, W_img, 3)  uint8 or [-1,1] float
    camera_params:  list[dict],          # len N, each has camera_ixt(9,) camera_ixt_hw(2,) camera_ext(16,) robot_base_pose_w(16,)
) -> bytes:
    """Project EE trajectories onto front camera images, one subplot per episode."""
    import io
    import math
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.collections import LineCollection

    N = pos_pred.shape[0]
    ncols = min(N, 4)
    nrows = math.ceil(N / ncols)

    fig, axes = plt.subplots(nrows, ncols, figsize=(3 * ncols, 3 * nrows), squeeze=False)

    for i in range(N):
        ax = axes[i // ncols][i % ncols]
        cp = camera_params[i]

        K        = np.asarray(cp["camera_ixt"],    dtype=np.float64).reshape(3, 3)
        ixt_hw   = np.asarray(cp["camera_ixt_hw"], dtype=np.int32)
        ext_w2c  = np.asarray(cp["camera_ext"],    dtype=np.float64).reshape(4, 4)
        T_base_w = np.asarray(cp["robot_base_pose_w"], dtype=np.float64).reshape(4, 4)

        # Scale K from collection resolution to vis image resolution.
        img = front_images[i]
        if img.dtype != np.uint8:
            img = np.clip((img + 1.0) * 127.5, 0, 255).astype(np.uint8)
        vis_h, vis_w = img.shape[:2]

        # The stored JPEGs are letterboxed (image_tools.resize_with_pad in
        # convert_npz_to_rlds._resize_and_encode_jpeg), not squashed: the frame is
        # scaled by ONE factor until its long edge fits, then centred with black
        # bars. Scaling fx and fy independently is only correct when the source is
        # already square -- for simpler's 4:3 front camera it lands the projection
        # up to ~22 px off vertically (the two agree exactly at the principal
        # point and diverge towards the edges, so it reads as a systematic shift).
        # For a square source this reduces to the old expression with pad == 0,
        # so the Franka suites are unaffected.
        scale = min(vis_w / ixt_hw[1], vis_h / ixt_hw[0])   # ixt_hw = [H, W]
        pad_x = (vis_w - ixt_hw[1] * scale) / 2.0
        pad_y = (vis_h - ixt_hw[0] * scale) / 2.0
        K_vis = K.copy()
        K_vis[0, :] *= scale
        K_vis[1, :] *= scale
        K_vis[0, 2] += pad_x
        K_vis[1, 2] += pad_y

        # T_cam_robot: panda_link0 → camera
        T_cam_robot = ext_w2c @ T_base_w

        def _project(pts_robot):
            """pts_robot: (H, 3) → (H, 2) pixel coords; mask invalid."""
            ones = np.ones((pts_robot.shape[0], 1), dtype=np.float64)
            p_cam = (T_cam_robot @ np.hstack([pts_robot, ones]).T).T  # (H, 4)
            valid = p_cam[:, 2] > 0
            u = np.where(valid, K_vis[0, 0] * p_cam[:, 0] / np.where(valid, p_cam[:, 2], 1) + K_vis[0, 2], -1)
            v = np.where(valid, K_vis[1, 1] * p_cam[:, 1] / np.where(valid, p_cam[:, 2], 1) + K_vis[1, 2], -1)
            u = np.clip(u, 0, vis_w - 1)
            v = np.clip(v, 0, vis_h - 1)
            return np.stack([u, v], axis=-1), valid

        ax.imshow(img)

        for pts, cmap_name, ls in (
            (pos_gt[i],   "Blues", "-"),
            (pos_pred[i], "Reds",  "--"),
        ):
            uv, valid = _project(pts.astype(np.float64))
            segs = []
            for t in range(len(uv) - 1):
                if valid[t] and valid[t + 1]:
                    segs.append([uv[t], uv[t + 1]])
            if segs:
                t_vals = np.linspace(0.4, 1.0, len(segs))
                cmap = plt.get_cmap(cmap_name)
                colors = cmap(t_vals)
                lc = LineCollection(segs, colors=colors, linewidths=1.5, linestyles=ls)
                ax.add_collection(lc)
            if valid[0]:
                ax.scatter(uv[0, 0], uv[0, 1], c="blue" if "Blue" in cmap_name else "red", s=20, zorder=5)

        ax.set_title(f"ep {i}", fontsize=8)
        ax.axis("off")

    # Hide unused subplots.
    for j in range(N, nrows * ncols):
        axes[j // ncols][j % ncols].axis("off")

    fig.tight_layout()
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=100)
    plt.close(fig)
    buf.seek(0)
    return buf.read()


def summarize_eef_metrics(
    pos_err: np.ndarray,
    rot_err: np.ndarray,
    gripper_err: np.ndarray,
    violated: np.ndarray,
    gripper_correct: np.ndarray | None = None,
) -> dict[str, float]:
    """Aggregate per-timestep arrays into scalar wandb metrics.

    Args:
        pos_err:         (N,) position errors in metres.
        rot_err:         (N,) rotation errors in radians.
        gripper_err:     (N,) gripper errors in storage-domain units.
        violated:        (N,) bool — joint limit violations (joint repr only).
        gripper_correct: (N,) bool — per-timestep binary intent classification correctness.
                         If provided, adds "gripper_acc" to the output.

    Returns:
        dict mapping wandb key → scalar float.
    """
    out = {
        "eef_pos_err":           float(np.mean(pos_err)),
        "eef_rot_err":           float(np.mean(rot_err)),
        "gripper_err":           float(np.mean(gripper_err)),
        "joint_violation_rate":  float(np.mean(violated)),
        "eef_pos_within_1cm":    float(np.mean(pos_err < 0.01)),
        "eef_pos_within_5cm":    float(np.mean(pos_err < 0.05)),
        "eef_pos_within_10cm":   float(np.mean(pos_err < 0.10)),
        "eef_rot_within_0p1rad": float(np.mean(rot_err < 0.1)),
        "eef_rot_within_0p2rad": float(np.mean(rot_err < 0.2)),
        "eef_rot_within_0p5rad": float(np.mean(rot_err < 0.5)),
    }
    if gripper_correct is not None:
        out["gripper_acc"] = float(np.mean(gripper_correct))
    return out
