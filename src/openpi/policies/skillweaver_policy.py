"""Policy input/output transforms for SkillWeaver trajectories in RLDS format.

The samples produced by IsaacLabRldsDataset carry one exterior camera view (sampled per step), the wrist
view, the robot state (arm joint positions + gripper) and an action chunk, plus the sampled camera's
calibration for trajectory visualization.
"""

import dataclasses

import numpy as np

from openpi import transforms
from openpi.models import model as _model


def make_skillweaver_example() -> dict:
    """Creates a random input example for the SkillWeaver policy."""
    return {
        "observation/image": np.random.randint(256, size=(224, 224, 3), dtype=np.uint8),
        "observation/wrist_image": np.random.randint(256, size=(224, 224, 3), dtype=np.uint8),
        "observation/state": np.random.rand(8),
        "prompt": "do something",
    }


@dataclasses.dataclass(frozen=True)
class SkillWeaverInputs(transforms.DataTransformFn):
    # Determines which model will be used.
    model_type: _model.ModelType
    # If False, ignore the wrist image and emit a zero-tensor with image_mask=False in the
    # left_wrist_0_rgb slot (right_wrist is always zero + False).
    use_wrist_camera: bool = True

    def __call__(self, data: dict) -> dict:
        base_image = data["observation/image"]
        if self.use_wrist_camera:
            wrist_image = data["observation/wrist_image"]
        else:
            wrist_image = np.zeros_like(base_image)
        state = data["observation/state"]

        wrist_mask = np.True_ if self.use_wrist_camera else np.False_
        match self.model_type:
            case _model.ModelType.PI0 | _model.ModelType.PI05:
                names = ("base_0_rgb", "left_wrist_0_rgb", "right_wrist_0_rgb")
                images = (base_image, wrist_image, np.zeros_like(base_image))
                image_masks = (np.True_, wrist_mask, np.False_)
            case _model.ModelType.PI0_FAST:
                names = ("base_0_rgb", "base_1_rgb", "wrist_0_rgb")
                # We don't mask out padding images for FAST models.
                images = (base_image, np.zeros_like(base_image), wrist_image)
                image_masks = (np.True_, np.True_, wrist_mask)
            case _:
                raise ValueError(f"Unsupported model type: {self.model_type}")

        inputs = {
            "state": state,
            "image": dict(zip(names, images, strict=True)),
            "image_mask": dict(zip(names, image_masks, strict=True)),
        }

        if "actions" in data:
            inputs["actions"] = np.asarray(data["actions"])

        if "prompt" in data:
            if isinstance(data["prompt"], bytes):
                data["prompt"] = data["prompt"].decode("utf-8")
            inputs["prompt"] = data["prompt"]

        # Pass the sampled camera's calibration through so the data loader can yield it for
        # end-effector trajectory visualization during validation.
        for k in ("camera_ixt", "camera_ixt_hw", "camera_ext", "robot_base_pose_w"):
            if k in data:
                inputs[k] = data[k]

        return inputs


@dataclasses.dataclass(frozen=True)
class SkillWeaverOutputs(transforms.DataTransformFn):
    def __call__(self, data: dict) -> dict:
        # 7 arm joints + 1 gripper.
        return {"actions": np.asarray(data["actions"][:, :8])}
