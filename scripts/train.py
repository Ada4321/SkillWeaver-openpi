import dataclasses
import functools
import logging
import platform
from typing import Any

import etils.epath as epath
import flax.nnx as nnx
from flax.training import common_utils
import flax.traverse_util as traverse_util
import jax
import jax.experimental
import jax.numpy as jnp
import numpy as np
import optax
import tqdm_loggable.auto as tqdm
import wandb

import openpi.eval.action_metrics as action_metrics
import openpi.models.model as _model
import openpi.shared.array_typing as at
import openpi.shared.nnx_utils as nnx_utils
import openpi.training.checkpoints as _checkpoints
import openpi.training.config as _config
import openpi.training.data_loader as _data_loader
import openpi.training.optimizer as _optimizer
import openpi.training.sharding as sharding
import openpi.training.utils as training_utils
import openpi.training.weight_loaders as _weight_loaders
import openpi.transforms as _transforms


def init_logging():
    """Custom logging format for better readability."""
    level_mapping = {"DEBUG": "D", "INFO": "I", "WARNING": "W", "ERROR": "E", "CRITICAL": "C"}

    class CustomFormatter(logging.Formatter):
        def format(self, record):
            record.levelname = level_mapping.get(record.levelname, record.levelname)
            return super().format(record)

    formatter = CustomFormatter(
        fmt="%(asctime)s.%(msecs)03d [%(levelname)s] %(message)-80s (%(process)d:%(filename)s:%(lineno)s)",
        datefmt="%H:%M:%S",
    )

    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    logger.handlers[0].setFormatter(formatter)


def init_wandb(config: _config.TrainConfig, *, resuming: bool, log_code: bool = False, enabled: bool = True):
    if not enabled:
        wandb.init(mode="disabled")
        return

    ckpt_dir = config.checkpoint_dir
    if not ckpt_dir.exists():
        raise FileNotFoundError(f"Checkpoint directory {ckpt_dir} does not exist.")
    if resuming:
        run_id = (ckpt_dir / "wandb_id.txt").read_text().strip()
        wandb.init(id=run_id, resume="must", project=config.project_name)
    else:
        wandb.init(
            name=config.exp_name,
            config=dataclasses.asdict(config),
            project=config.project_name,
        )
        (ckpt_dir / "wandb_id.txt").write_text(wandb.run.id)

    if log_code:
        wandb.run.log_code(epath.Path(__file__).parent.parent)


def _load_weights_and_validate(loader: _weight_loaders.WeightLoader, params_shape: at.Params) -> at.Params:
    """Loads and validates the weights. Returns a loaded subset of the weights."""
    loaded_params = loader.load(params_shape)
    at.check_pytree_equality(expected=params_shape, got=loaded_params, check_shapes=True, check_dtypes=True)

    # Remove jax.ShapeDtypeStruct from the loaded params. This makes sure that only the loaded params are returned.
    return traverse_util.unflatten_dict(
        {k: v for k, v in traverse_util.flatten_dict(loaded_params).items() if not isinstance(v, jax.ShapeDtypeStruct)}
    )


@at.typecheck
def init_train_state(
    config: _config.TrainConfig, init_rng: at.KeyArrayLike, mesh: jax.sharding.Mesh, *, resume: bool
) -> tuple[training_utils.TrainState, Any]:
    tx = _optimizer.create_optimizer(config.optimizer, config.lr_schedule, weight_decay_mask=None)

    def init(rng: at.KeyArrayLike, partial_params: at.Params | None = None) -> training_utils.TrainState:
        rng, model_rng = jax.random.split(rng)
        # initialize the model (and its parameters).
        model = config.model.create(model_rng)

        # Merge the partial params into the model.
        if partial_params is not None:
            graphdef, state = nnx.split(model)
            # This will produce an error if the partial params are not a subset of the state.
            state.replace_by_pure_dict(partial_params)
            model = nnx.merge(graphdef, state)

        params = nnx.state(model)
        # Convert frozen params to bfloat16.
        params = nnx_utils.state_map(params, config.freeze_filter, lambda p: p.replace(p.value.astype(jnp.bfloat16)))

        return training_utils.TrainState(
            step=0,
            params=params,
            model_def=nnx.graphdef(model),
            tx=tx,
            opt_state=tx.init(params.filter(config.trainable_filter)),
            ema_decay=config.ema_decay,
            ema_params=None if config.ema_decay is None else params,
        )

    train_state_shape = jax.eval_shape(init, init_rng)
    state_sharding = sharding.fsdp_sharding(train_state_shape, mesh, log=True)

    if resume:
        return train_state_shape, state_sharding

    partial_params = _load_weights_and_validate(config.weight_loader, train_state_shape.params.to_pure_dict())
    replicated_sharding = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec())

    # Initialize the train state and mix in the partial params.
    train_state = jax.jit(
        init,
        donate_argnums=(1,),  # donate the partial params buffer.
        in_shardings=replicated_sharding,
        out_shardings=state_sharding,
    )(init_rng, partial_params)

    return train_state, state_sharding


@at.typecheck
def train_step(
    config: _config.TrainConfig,
    rng: at.KeyArrayLike,
    state: training_utils.TrainState,
    batch: tuple[_model.Observation, _model.Actions],
) -> tuple[training_utils.TrainState, dict[str, at.Array]]:
    model = nnx.merge(state.model_def, state.params)
    model.train()

    @at.typecheck
    def loss_fn(
        model: _model.BaseModel, rng: at.KeyArrayLike, observation: _model.Observation, actions: _model.Actions
    ):
        chunked_loss = model.compute_loss(rng, observation, actions, train=True)
        return jnp.mean(chunked_loss)

    train_rng = jax.random.fold_in(rng, state.step)
    observation, actions = batch

    # Filter out frozen params.
    diff_state = nnx.DiffState(0, config.trainable_filter)
    loss, grads = nnx.value_and_grad(loss_fn, argnums=diff_state)(model, train_rng, observation, actions)

    params = state.params.filter(config.trainable_filter)
    updates, new_opt_state = state.tx.update(grads, state.opt_state, params)
    new_params = optax.apply_updates(params, updates)

    # Update the model in place and return the new full state.
    nnx.update(model, new_params)
    new_params = nnx.state(model)

    new_state = dataclasses.replace(state, step=state.step + 1, params=new_params, opt_state=new_opt_state)
    if state.ema_decay is not None:
        new_state = dataclasses.replace(
            new_state,
            ema_params=jax.tree.map(
                lambda old, new: state.ema_decay * old + (1 - state.ema_decay) * new, state.ema_params, new_params
            ),
        )

    # Filter out params that aren't kernels.
    kernel_params = nnx.state(
        model,
        nnx.All(
            nnx.Param,
            nnx.Not(nnx_utils.PathRegex(".*/(bias|scale|pos_embedding|input_embedding)")),
            lambda _, x: x.value.ndim > 1,
        ),
    )
    info = {
        "loss": loss,
        "grad_norm": optax.global_norm(grads),
        "param_norm": optax.global_norm(kernel_params),
    }
    return new_state, info


@at.typecheck
def eval_step(
    config: _config.TrainConfig,
    rng: at.KeyArrayLike,
    state: training_utils.TrainState,
    batch: tuple[_model.Observation, _model.Actions],
) -> dict[str, at.Array]:
    params_for_eval = state.ema_params if state.ema_params is not None else state.params
    model = nnx.merge(state.model_def, params_for_eval)
    model.eval()

    observation, actions = batch
    chunked_loss = model.compute_loss(rng, observation, actions, train=False)
    return {"loss": jnp.mean(chunked_loss)}


def sample_step(
    config: _config.TrainConfig,
    rng: at.KeyArrayLike,
    state: training_utils.TrainState,
    batch: tuple[_model.Observation, _model.Actions],
) -> at.Array:
    """Return denoised actions sampled from the model (no gradient)."""
    params_for_eval = state.ema_params if state.ema_params is not None else state.params
    model = nnx.merge(state.model_def, params_for_eval)
    model.eval()
    return model.sample_actions(rng, batch[0])


def main(config: _config.TrainConfig):
    init_logging()
    logging.info(f"Running on: {platform.node()}")

    if config.batch_size % jax.device_count() != 0:
        raise ValueError(
            f"Batch size {config.batch_size} must be divisible by the number of devices {jax.device_count()}."
        )

    jax.config.update("jax_compilation_cache_dir", str(epath.Path("~/.cache/jax").expanduser()))

    rng = jax.random.key(config.seed)
    train_rng, init_rng = jax.random.split(rng)

    mesh = sharding.make_mesh(config.fsdp_devices)
    data_sharding = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec(sharding.DATA_AXIS))
    replicated_sharding = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec())

    checkpoint_manager, resuming = _checkpoints.initialize_checkpoint_dir(
        config.checkpoint_dir,
        keep_period=config.keep_period,
        overwrite=config.overwrite,
        resume=config.resume,
    )
    init_wandb(config, resuming=resuming, enabled=config.wandb_enabled)

    data_config = config.data.create(config.assets_dirs, config.model)
    data_loader = _data_loader.create_data_loader(
        config,
        sharding=data_sharding,
        shuffle=True,
    )

    def _strip_camera(b):
        # The RLDS loader also yields the sampled camera's calibration (obs, actions, camera) for the
        # validation plots; the training step only takes (obs, actions).
        if len(b) == 3 and isinstance(b[2], dict):
            return b[:2]
        return b

    data_iter = iter(data_loader)
    batch = _strip_camera(next(data_iter))
    logging.info(f"Initialized data loader:\n{training_utils.array_tree_to_info(batch)}")

    # Validation on data_config.rlds_val_data_dirs (RLDS datasets only).
    n_val_batches = 0
    if data_config.rlds_dataset_type == "isaaclab":
        _, n_val_batches = _data_loader.create_rlds_val_iter(
            data_config,
            action_horizon=config.model.action_horizon,
            batch_size=config.batch_size,
            sharding=data_sharding,
        )

    # Log images from first batch to sanity check.
    images_to_log = [
        wandb.Image(np.concatenate([np.array(img[i]) for img in batch[0].images.values()], axis=1))
        for i in range(min(5, len(next(iter(batch[0].images.values())))))
    ]
    wandb.log({"camera_views": images_to_log}, step=0)

    train_state, train_state_sharding = init_train_state(config, init_rng, mesh, resume=resuming)
    jax.block_until_ready(train_state)
    logging.info(f"Initialized train state:\n{training_utils.array_tree_to_info(train_state.params)}")

    if resuming:
        train_state = _checkpoints.restore_state(checkpoint_manager, train_state, data_loader)

    ptrain_step = jax.jit(
        functools.partial(train_step, config),
        in_shardings=(replicated_sharding, train_state_sharding, data_sharding),
        out_shardings=(train_state_sharding, replicated_sharding),
        donate_argnums=(1,),
    )

    peval_step = jax.jit(
        functools.partial(eval_step, config),
        in_shardings=(replicated_sharding, train_state_sharding, data_sharding),
        out_shardings=replicated_sharding,
    )
    psample_step = jax.jit(
        functools.partial(sample_step, config),
        in_shardings=(replicated_sharding, train_state_sharding, data_sharding),
        out_shardings=replicated_sharding,
    )

    # Extract action-representation settings from data config for save-interval metrics.
    _arm_repr      = getattr(config.data, "arm_repr",      "joint")
    _state_repr    = getattr(config.data, "state_repr",    "joint")
    _arm_is_delta  = getattr(config.data, "arm_is_delta",  False)
    _gripper_1d    = getattr(config.data, "gripper_1d",    True)
    # Arm width and embodiment come from the shard (RLDSIsaacLabDataConfig.create
    # reads dataset_info.json); the 7/6 literals are only the fallback for configs
    # that carry no shard metadata. Guessing 7 here is what silently fed a WidowX
    # "6 joints + gripper" vector to the Franka FK — it reshaped cleanly and
    # produced EE poses ~0.7 m off.
    _robot         = getattr(data_config, "robot", None)
    _arm_dim       = getattr(data_config, "arm_dim", None)       or (7 if _arm_repr   == "joint" else 6)
    _state_arm_dim = getattr(data_config, "state_arm_dim", None) or (7 if _state_repr == "joint" else 6)
    # data_config (realized) carries the binary-gripper switch; factory copy
    # is forwarded into it via RLDSIsaacLabDataConfig.create.
    _replace_gripper_with_binary = getattr(data_config, "replace_gripper_with_binary", False)
    # use_quantile_norm lives on the realized DataConfig (set inside create_base_config
    # per model_type), NOT on config.data (the factory); reading it from the factory
    # silently returns False and silently breaks denormalize → squashed EE-traj vis.
    _use_quantiles = getattr(data_config, "use_quantile_norm", False)

    # AbsoluteActions instances on the output side undo training-time DeltaActions
    # (e.g. Droid-style chunk-anchor delta). Apply them in eval below so denoised
    # predictions and ground-truth get back to the storage-domain action space
    # before FK; otherwise compute_eef_errors treats deltas as absolute targets.
    _abs_action_transforms = [
        t for t in data_config.data_transforms.outputs
        if isinstance(t, _transforms.AbsoluteActions)
    ]

    start_step = int(train_state.step)
    pbar = tqdm.tqdm(
        range(start_step, config.num_train_steps),
        initial=start_step,
        total=config.num_train_steps,
        dynamic_ncols=True,
    )

    infos = []
    for step in pbar:
        with sharding.set_mesh(mesh):
            train_state, info = ptrain_step(train_rng, train_state, batch)
        infos.append(info)
        if step % config.log_interval == 0:
            stacked_infos = common_utils.stack_forest(infos)
            reduced_info = jax.device_get(jax.tree.map(jnp.mean, stacked_infos))
            info_str = ", ".join(f"{k}={v:.4f}" for k, v in reduced_info.items())
            pbar.write(f"Step {step}: {info_str}")
            wandb.log(reduced_info, step=step)
            infos = []

        batch = _strip_camera(next(data_iter))

        run_eval = (
            config.eval_interval > 0
            and ((step % config.eval_interval == 0 and step > start_step)
                 or step == config.num_train_steps - 1)
        )
        if run_eval:
            # Full val-set metrics (val/loss + EEF + vis).
            if n_val_batches > 0:
                # Create a fresh iterator for one complete pass over the val set.
                save_val_iter, _ = _data_loader.create_rlds_val_iter(
                    data_config,
                    action_horizon=config.model.action_horizon,
                    batch_size=config.batch_size,
                    sharding=data_sharding,
                )

                act_stats   = data_config.norm_stats.get("actions")
                state_stats = data_config.norm_stats.get("state")

                all_loss            = []
                all_pos_err         = []
                all_rot_err         = []
                all_gripper_err     = []
                all_gripper_correct = []
                all_violated        = []

                # Vis collection (first vis_ee_traj_episodes batches only).
                _vis_n = config.vis_ee_traj_episodes
                vis_pos_pred:   list[np.ndarray] = []
                vis_pos_gt:     list[np.ndarray] = []
                vis_images:     list[np.ndarray] = []
                vis_cam_params: list[dict]        = []

                for _ in range(n_val_batches):
                    sv_batch = next(save_val_iter)

                    # Unpack: may be (obs, actions) or (obs, actions, camera_dict).
                    if len(sv_batch) == 3 and isinstance(sv_batch[2], dict):
                        sv_obs, sv_actions, sv_camera = sv_batch
                    else:
                        sv_obs, sv_actions = sv_batch[0], sv_batch[1]
                        sv_camera = None

                    # Metric 1: flow-matching velocity MSE (same as training)
                    with sharding.set_mesh(mesh):
                        sv_info = peval_step(train_rng, train_state, (sv_obs, sv_actions))
                    all_loss.append(float(jax.device_get(sv_info["loss"])))

                    # Metrics 2+3: denoised EEF errors
                    if act_stats is not None:
                        with sharding.set_mesh(mesh):
                            pred_norm = jax.device_get(psample_step(train_rng, train_state, (sv_obs, sv_actions)))
                        gt_norm   = jax.device_get(sv_actions)
                        obs_state = jax.device_get(sv_obs.state)

                        # Model uses padded action/state dims (e.g. 32); norm_stats are in the real dim (e.g. 8).
                        act_dim = int(np.asarray(act_stats.mean).shape[0])
                        pred_phys = action_metrics.denormalize(np.array(pred_norm)[..., :act_dim], act_stats, use_quantiles=_use_quantiles)
                        gt_phys   = action_metrics.denormalize(np.array(gt_norm)[..., :act_dim],   act_stats, use_quantiles=_use_quantiles)
                        if state_stats is not None:
                            state_dim = int(np.asarray(state_stats.mean).shape[0])
                            obs_phys = action_metrics.denormalize(np.array(obs_state)[..., :state_dim], state_stats, use_quantiles=_use_quantiles)
                        else:
                            obs_phys = np.array(obs_state)

                        # Undo training-time DeltaActions (e.g. Droid-style relative)
                        # so pred/gt are in storage-domain action space before FK.
                        for _abs_t in _abs_action_transforms:
                            pred_phys = _abs_t({"state": obs_phys, "actions": pred_phys})["actions"]
                            gt_phys   = _abs_t({"state": obs_phys, "actions": gt_phys})["actions"]

                        per_sample = action_metrics.compute_eef_errors(
                            pred_phys,
                            gt_phys,
                            obs_phys,
                            arm_repr=_arm_repr,
                            arm_dim=_arm_dim,
                            state_arm_dim=_state_arm_dim,
                            arm_is_delta=_arm_is_delta,
                            gripper_1d=_gripper_1d,
                            replace_gripper_with_binary=_replace_gripper_with_binary,
                            robot=_robot,
                        )
                        all_pos_err.append(per_sample["pos_err"])
                        all_rot_err.append(per_sample["rot_err"])
                        all_gripper_err.append(per_sample["gripper_err"])
                        all_gripper_correct.append(per_sample["gripper_correct"])
                        all_violated.append(per_sample["violated"])

                        # Collect for EE traj vis (limited to first _vis_n batches).
                        if _vis_n > 0 and len(vis_pos_pred) < _vis_n:
                            pp, pg = action_metrics.extract_ee_positions(
                                pred_phys, gt_phys, obs_phys,
                                arm_repr=_arm_repr,
                                arm_dim=_arm_dim,
                                state_arm_dim=_state_arm_dim,
                                arm_is_delta=_arm_is_delta,
                                robot=_robot,
                            )
                            front_key = "base_0_rgb"
                            front_img = np.array(jax.device_get(sv_obs.images[front_key]))  # (B, H, W, 3)
                            B = pp.shape[0]
                            for b in range(B):
                                if len(vis_pos_pred) >= _vis_n:
                                    break
                                vis_pos_pred.append(pp[b])
                                vis_pos_gt.append(pg[b])
                                vis_images.append(front_img[b])
                                if sv_camera is not None:
                                    vis_cam_params.append({k: np.array(v[b]) for k, v in sv_camera.items()})

                save_metrics = {"val/loss": float(np.mean(all_loss))}
                if all_pos_err:
                    eef_metrics = action_metrics.summarize_eef_metrics(
                        pos_err=np.concatenate(all_pos_err),
                        rot_err=np.concatenate(all_rot_err),
                        gripper_err=np.concatenate(all_gripper_err),
                        violated=np.concatenate(all_violated),
                        gripper_correct=np.concatenate(all_gripper_correct),
                    )
                    save_metrics.update({f"val/{k}": v for k, v in eef_metrics.items()})

                wandb.log(save_metrics, step=step)
                msg = (
                    f"Step {step} [eval]: val/loss={save_metrics['val/loss']:.4f}"
                )
                if "val/eef_pos_err" in save_metrics:
                    msg += f"  eef_pos={save_metrics['val/eef_pos_err']*100:.2f}cm"
                    msg += f"  eef_rot={save_metrics['val/eef_rot_err']:.3f}rad"
                pbar.write(msg)

                # EE trajectory visualization.
                if _vis_n > 0 and vis_pos_pred:
                    import pathlib as _pathlib
                    _vis_dir = _pathlib.Path(config.checkpoint_dir) / "ee_traj_vis"
                    _vis_dir.mkdir(parents=True, exist_ok=True)
                    _pos_pred_arr = np.stack(vis_pos_pred)  # (N, H, 3)
                    _pos_gt_arr   = np.stack(vis_pos_gt)    # (N, H, 3)
                    _png_3d = action_metrics._plot_ee_traj_3d(_pos_pred_arr, _pos_gt_arr)
                    _png_3d_path = _vis_dir / f"ee_traj_3d_{step:07d}.png"
                    _png_3d_path.write_bytes(_png_3d)
                    vis_metrics = {"vis/ee_traj_3d": wandb.Image(str(_png_3d_path))}
                    if vis_cam_params and len(vis_cam_params) == len(vis_pos_pred):
                        _imgs_arr = np.stack(vis_images)  # (N, H, W, 3)
                        _png_front = action_metrics._plot_ee_traj_front(
                            _pos_pred_arr, _pos_gt_arr, _imgs_arr, vis_cam_params
                        )
                        _png_front_path = _vis_dir / f"ee_traj_front_{step:07d}.png"
                        _png_front_path.write_bytes(_png_front)
                        vis_metrics["vis/ee_traj_front"] = wandb.Image(str(_png_front_path))
                    wandb.log(vis_metrics, step=step)

        if (step % config.save_interval == 0 and step > start_step) or step == config.num_train_steps - 1:
            _checkpoints.save_state(checkpoint_manager, train_state, data_loader, step)

    logging.info("Waiting for checkpoint manager to finish")
    checkpoint_manager.wait_until_finished()


if __name__ == "__main__":
    main(_config.cli())
