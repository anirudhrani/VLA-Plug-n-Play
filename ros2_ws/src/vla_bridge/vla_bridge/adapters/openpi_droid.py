"""openpi DROID policies (pi05_droid, pi0_fast_droid, ...).

Request keys and preprocessing follow openpi examples/droid/main.py: both images resize_with_pad to 224x224,
joint positions in rad, gripper position 0 = open .. 1 = closed, plus the language prompt.
Response: actions (H, 8) = [7 normalized joint velocities, gripper position]; pi05_droid has H = 15.
The robot-side conversion (clip, q + 0.2*v, gripper binarize) happens in Isaac, not here.
"""
from __future__ import annotations

import numpy as np
from openpi_client import image_tools

ACTION_TYPE = "droid_joint_velocity"
ARM_JOINTS = [f"panda_joint{i}" for i in range(1, 8)]
ACTION_DIM_NAMES = ARM_JOINTS + ["gripper"]
IMAGE_SIZE = 224
HOME = np.array([0, -np.pi / 5, 0, -4 * np.pi / 5, 0, 3 * np.pi / 5, 0])


def build_request(exo_rgb: np.ndarray, wrist_rgb: np.ndarray, joint_names: list[str], joint_pos: list[float],
                  gripper: float, prompt: str) -> dict:
    missing = [n for n in ARM_JOINTS if n not in joint_names]
    if missing:
        raise ValueError(f"joint_states is missing {missing}")
    q = np.array([joint_pos[joint_names.index(n)] for n in ARM_JOINTS], dtype=np.float64)
    return {
        "observation/exterior_image_1_left": image_tools.resize_with_pad(exo_rgb, IMAGE_SIZE, IMAGE_SIZE),
        "observation/wrist_image_left": image_tools.resize_with_pad(wrist_rgb, IMAGE_SIZE, IMAGE_SIZE),
        "observation/joint_position": q,
        "observation/gripper_position": np.array([gripper], dtype=np.float64),
        "prompt": prompt,
    }


def warmup_request() -> dict:
    """Dummy observation used once at startup so the server JIT-compiles before the first real request."""
    black = np.zeros((IMAGE_SIZE, IMAGE_SIZE, 3), np.uint8)
    return build_request(black, black, ARM_JOINTS, HOME.tolist(), 0.0, "warm up")


def parse_response(resp: dict) -> np.ndarray:
    actions = np.asarray(resp["actions"], dtype=np.float64)
    if actions.ndim != 2 or actions.shape[1] != len(ACTION_DIM_NAMES):
        raise ValueError(f"expected actions (H, {len(ACTION_DIM_NAMES)}), got {actions.shape}")
    if not np.all(np.isfinite(actions)):
        raise ValueError("actions contain NaN/inf")
    return actions
