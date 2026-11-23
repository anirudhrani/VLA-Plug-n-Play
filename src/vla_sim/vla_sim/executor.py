"""Executor: the lockstep episode loop (inference layer, plan §6).

    OBSERVE (sim paused) -> WAIT_CHUNK (sim paused) -> EXECUTE n actions (sim plays) -> OBSERVE ...

Actions are applied from the PhysX step callback: one action per `steps_per_action` physics steps
(60 Hz physics / 15 Hz control = 4). Where chunks come from is a ChunkSource: the ROS bridge (M5) or a
scripted list (M4 motion test), so the loop itself is model-agnostic.
"""
from __future__ import annotations

import asyncio
import dataclasses
import enum
import math
import time
from typing import Callable, Protocol

import carb
import numpy as np
import omni.physx
import omni.timeline
import omni.usd
from pxr import Usd

from .sim_io import Observation, SimIO


class ExecState(enum.Enum):
    IDLE = "idle"
    OBSERVING = "observing"
    WAITING = "waiting for chunk"
    EXECUTING = "executing"


class ChunkSource(Protocol):
    async def next_chunk(self, obs: Observation) -> np.ndarray | None:
        """Return the (H, D) raw action chunk for this observation, or None to end the episode."""


# -- action conversion (robot side) -----------------------------------------------------------------
class DroidJointVelocity:
    """DROID `joint_velocity` + `gripper_position` action space (droid/franka/robot.py, robot_ik_solver.py).

    arm:     v = clip(a[:7], -1, 1);  scale down so max|v| <= 1;  q_target = q_now + 0.2 * v   (rad)
    gripper: binarize a[7] at 0.5 -> 0 open / 1 closed -> finger_joint open/closed limit
    """

    MAX_JOINT_DELTA = 0.2  # rad per control step (15 Hz)

    def __init__(self, arm_idx: list[int], grip_idx: int, lower: np.ndarray, upper: np.ndarray,
                 grip_open: float, grip_closed: float):
        self.arm_idx, self.grip_idx = arm_idx, grip_idx
        self.lower, self.upper = lower, upper
        self.grip_open, self.grip_closed = grip_open, grip_closed

    def targets(self, action: np.ndarray, q_now: np.ndarray, current_targets: np.ndarray) -> np.ndarray:
        a = np.clip(np.asarray(action, dtype=np.float64), -1.0, 1.0)
        v = a[:7]
        m = np.abs(v).max()
        if m > 1.0:
            v = v / m
        out = current_targets.copy()
        arm_t = q_now[self.arm_idx] + self.MAX_JOINT_DELTA * v
        out[self.arm_idx] = np.clip(arm_t, self.lower[self.arm_idx], self.upper[self.arm_idx])
        out[self.grip_idx] = self.grip_closed if a[7] > 0.5 else self.grip_open
        return out


class DroidJointPosition:
    """DROID `joint_position` + `gripper_position` action space (openpi *_jointpos_polaris checkpoints).

    arm:     q_target = a[:7]  (absolute joint angles in rad; openpi already converted the model's deltas back
             to absolute with AbsoluteActions), clipped to the joint limits
    gripper: binarize a[7] at 0.5, as for joint velocity
    """

    def __init__(self, arm_idx: list[int], grip_idx: int, lower: np.ndarray, upper: np.ndarray,
                 grip_open: float, grip_closed: float):
        self.arm_idx, self.grip_idx = arm_idx, grip_idx
        self.lower, self.upper = lower, upper
        self.grip_open, self.grip_closed = grip_open, grip_closed

    def targets(self, action: np.ndarray, q_now: np.ndarray, current_targets: np.ndarray) -> np.ndarray:
        a = np.asarray(action, dtype=np.float64)
        out = current_targets.copy()
        out[self.arm_idx] = np.clip(a[:7], self.lower[self.arm_idx], self.upper[self.arm_idx])
        out[self.grip_idx] = self.grip_closed if a[7] > 0.5 else self.grip_open
        return out


CONVERTERS = {"droid_joint_velocity": DroidJointVelocity, "droid_joint_position": DroidJointPosition}


# -- tracking statistics ------------------------------------------------------------------------------
@dataclasses.dataclass
class Tracking:
    """Per action: how far the arm is from the target it was given, measured when the next action starts."""
    n: int = 0
    sum_err: float = 0.0
    max_err: float = 0.0
    max_err_joint: int = -1
    sum_cmd: float = 0.0          # mean |commanded delta| for context

    def add(self, err: np.ndarray, cmd: np.ndarray) -> None:
        e = np.abs(err)
        self.n += 1
        self.sum_err += float(e.max())
        self.sum_cmd += float(np.abs(cmd).max())
        if e.max() > self.max_err:
            self.max_err, self.max_err_joint = float(e.max()), int(e.argmax()) + 1

    def summary(self) -> str:
        if not self.n:
            return "no actions"
        return (f"{self.n} actions · tracking error mean {self.sum_err / self.n:.3f} rad, max {self.max_err:.3f} rad "
                f"(joint {self.max_err_joint}) · mean max|Δq cmd| {self.sum_cmd / self.n:.3f} rad")


# -- executor -----------------------------------------------------------------------------------------
class Executor:
    def __init__(self, sim: SimIO, scene_cfg: dict, on_change: Callable[[], None]):
        self.sim = sim
        self.cfg = scene_cfg
        self._on_change = on_change
        self.state = ExecState.IDLE
        self.status = "idle"
        self.steps_done = 0
        self.tracking = Tracking()
        self._stop = False
        self._tl = omni.timeline.get_timeline_interface()
        self._sub = omni.physx.get_physx_interface().subscribe_physics_step_events(self._on_physics_step)
        self._actions: list[np.ndarray] = []
        self._ai = 0
        self._k = 0
        self._done: asyncio.Future | None = None
        self._last_target: np.ndarray | None = None
        self._last_cmd: np.ndarray | None = None
        self._conv = None
        self._art = None
        self.dof_names: list[str] = []
        self._is_done: Callable[[], str | None] = lambda: None
        self._early: str | None = None   # outcome found by is_done() inside the physics callback
        self.steps_per_action = 4
        self.max_steps = 0
        self.log: list[dict] = []     # per-action record of the last run (for tests / debugging)
        # optional recorder hook: called with (observation, step, sim time) for the first observation and after
        # every action (the world is paused for it), giving a 15 fps view of the episode. None = no extra pauses.
        self.frame_hook: Callable[[Observation, int, float], None] | None = None

    @property
    def running(self) -> bool:
        return self.state != ExecState.IDLE

    def stop(self) -> None:
        self._stop = True

    # -- episode loop ---------------------------------------------------------------------------
    async def run(self, source: ChunkSource, action_type: str, n_exec: int, max_steps: int,
                  control_hz: float = 15.0, is_done: Callable[[], str | None] = lambda: None) -> str:
        """Run until the source ends, Stop, max_steps, or is_done() returns an outcome. Returns the outcome.

        is_done() is checked before every observation and at every action boundary while executing
        (so e.g. success stops the arm within one action, not at the end of the chunk).
        """
        if self.running:
            return "busy"
        self._stop, self.steps_done, self.tracking, self.log = False, 0, Tracking(), []
        self.max_steps = max_steps
        self._is_done, self._early = is_done, None
        self._last_target = self._last_cmd = None
        outcome = "error"
        try:
            self._prepare(action_type, control_hz)
            while True:
                if self._stop:
                    outcome = "stopped"
                    break
                if self.steps_done >= max_steps:
                    outcome = "timeout"
                    break
                done = is_done()
                if done:
                    outcome = done
                    break
                self._set(ExecState.OBSERVING, f"observing · step {self.steps_done}/{max_steps}")
                obs = await self.sim.observe(keep_capture_open=True)
                if self.steps_done == 0:
                    self._call_hook(obs)
                self._set(ExecState.WAITING, f"waiting for chunk · step {self.steps_done}/{max_steps}")
                chunk = await source.next_chunk(obs)
                if chunk is None:
                    outcome = "done"
                    break
                if self._stop:
                    outcome = "stopped"
                    break
                actions = list(np.asarray(chunk)[: min(n_exec, max_steps - self.steps_done)])
                self._set(ExecState.EXECUTING, f"executing {len(actions)} actions · step {self.steps_done}/{max_steps}")
                await self._execute(actions)
                if self._early:
                    outcome = self._early
                    break
        except Exception as e:  # noqa: BLE001
            carb.log_error(f"[vla_sim] episode error: {e}")
            outcome = f"error: {e}"
        finally:
            self._actions, self._done, self._is_done = [], None, (lambda: None)
            try:
                await self.sim.close_capture()
                if not self._tl.is_playing() and not self._tl.is_stopped():
                    self._tl.play()  # world keeps running; the arm holds its last targets
            except Exception as e:  # noqa: BLE001
                carb.log_error(f"[vla_sim] episode cleanup: {e}")
            finally:
                label = outcome.upper() if outcome in ("success", "timeout") else outcome
                self._set(ExecState.IDLE, f"{label} · {self.steps_done} steps · {self.tracking.summary()}")
        return outcome

    def _prepare(self, action_type: str, control_hz: float) -> None:
        if self._tl.is_stopped():
            raise RuntimeError("press Play first (the simulation is stopped)")
        robot = self.cfg["robot"]
        tsps = _physics_rate()
        hz = float(control_hz)
        spa = tsps / hz
        if abs(spa - round(spa)) > 1e-6:
            raise RuntimeError(f"physics rate {tsps} Hz is not a multiple of control rate {hz} Hz")
        self.steps_per_action = int(round(spa))
        art, dof_names = self.sim.joints.articulation()
        self._art = art
        self.dof_names = list(dof_names)   # order of q / target in self.log
        arm_idx = [dof_names.index(n) for n in robot["arm_joints"]]
        grip_idx = dof_names.index(robot["gripper_joint"])
        limits = np.asarray(art.get_dof_limits())[0]
        self._conv = CONVERTERS[action_type](
            arm_idx, grip_idx, limits[:, 0], limits[:, 1],
            math.radians(robot["gripper_open_deg"]), math.radians(robot["gripper_closed_deg"]),
        )
        self._apply_drive_overrides(art, arm_idx, grip_idx)

    def _apply_drive_overrides(self, art, arm_idx: list[int], grip_idx: int) -> None:
        """Optional runtime gain overrides from the scene config (sim only; the USD asset is not modified)."""
        ov = self.cfg["robot"].get("drive_overrides") or {}
        if not ov:
            return
        idx = np.arange(1, dtype=np.int32)
        if ov.get("disable_gravity"):
            # equivalent of the real Franka's gravity compensation: robot links only, objects keep gravity
            g = np.asarray(art.get_disable_gravities()).copy()
            g[:] = 1
            art.set_disable_gravities(g, idx)
        for getter, setter, key, joints in (
            (art.get_dof_stiffnesses, art.set_dof_stiffnesses, "arm_stiffness", arm_idx),
            (art.get_dof_dampings, art.set_dof_dampings, "arm_damping", arm_idx),
            (art.get_dof_stiffnesses, art.set_dof_stiffnesses, "gripper_stiffness", [grip_idx]),
            (art.get_dof_dampings, art.set_dof_dampings, "gripper_damping", [grip_idx]),
            (art.get_dof_max_forces, art.set_dof_max_forces, "gripper_max_force", [grip_idx]),
        ):
            if key in ov:
                vals = np.asarray(getter()).copy()
                vals[0, joints] = ov[key]
                setter(vals, idx)

    async def _execute(self, actions: list[np.ndarray]) -> None:
        """Play until the chunk is done; with a frame hook, pause after every action for a recorder frame."""
        self._actions, self._ai, self._k = actions, 0, 0
        while True:
            self._done = asyncio.get_event_loop().create_future()
            self._tl.play()
            # actions advance in the physics callback; if the timeline is stopped from outside (Isaac's Stop
            # button), no more physics steps come and the chunk would never finish -> end the episode
            while not self._done.done():
                await asyncio.sleep(0.01 if self.frame_hook else 0.05)
                if self._tl.is_stopped() and not self._done.done():
                    self._stop = True
                    self._done.set_result(None)
            if self._stop or self._early or self.frame_hook is None:
                return
            self._call_hook(await self.sim.observe(keep_capture_open=True))
            if self._ai >= len(self._actions):
                return

    def _call_hook(self, obs: Observation) -> None:
        if self.frame_hook is None:
            return
        try:
            self.frame_hook(obs, self.steps_done, obs.stamp_s)
        except Exception as e:  # noqa: BLE001  (recording must never break an episode)
            carb.log_error(f"[vla_sim] recorder: {e}; recording disabled for this episode")
            self.frame_hook = None

    # -- physics step callback (runs before each physics step) ----------------------------------
    def _on_physics_step(self, dt: float) -> None:
        if self.state != ExecState.EXECUTING or self._done is None or self._done.done():
            return
        try:
            art = self._art
            if self._k == 0:
                q = np.asarray(art.get_dof_positions())[0].astype(np.float64)
                if self._last_target is not None:
                    idx = self._conv.arm_idx
                    self.tracking.add(q[idx] - self._last_target[idx], self._last_cmd)
                if self._stop:
                    self._finish()
                    return
                done = self._is_done()
                if done:
                    self._early = done
                    self._finish()
                    return
                cur = np.asarray(art.get_dof_position_targets())[0].astype(np.float64)
                tgt = self._conv.targets(self._actions[self._ai], q, cur)
                art.set_dof_position_targets(tgt[None, :].astype(np.float32), np.arange(1, dtype=np.int32))
                self._last_target = tgt
                self._last_cmd = tgt[self._conv.arm_idx] - q[self._conv.arm_idx]
                self.log.append({"step": self.steps_done, "sim_t": float(self._tl.get_current_time()), "q": q.copy(), "target": tgt.copy(),
                                 "action": np.asarray(self._actions[self._ai]).copy()})
            self._k += 1
            if self._k >= self.steps_per_action:
                self._k = 0
                self._ai += 1
                self.steps_done += 1
                if self._ai >= len(self._actions) or self.frame_hook is not None:
                    self._finish()   # end of chunk, or (recording) end of every action
        except Exception as e:  # noqa: BLE001
            carb.log_error(f"[vla_sim] physics step: {e}")
            self._stop = True
            self._finish()

    def _finish(self) -> None:
        self._tl.pause()
        if self._done is not None and not self._done.done():
            self._done.set_result(None)

    def _set(self, state: ExecState, status: str) -> None:
        self.state, self.status = state, status
        self._on_change()

    def destroy(self) -> None:
        self._stop = True
        self._sub = None


def _physics_rate() -> float:
    from pxr import UsdPhysics
    stage = omni.usd.get_context().get_stage()
    for p in stage.Traverse():
        if p.IsA(UsdPhysics.Scene):
            a = p.GetAttribute("physxScene:timeStepsPerSecond")
            if a and a.HasValue():
                return float(a.Get())
    return 60.0


# -- scripted source for the M4 motion test ----------------------------------------------------------
class ScriptedSource:
    """Returns pre-defined chunks one per observation (ignores the observation)."""

    def __init__(self, chunks: list[np.ndarray]):
        self._chunks = list(chunks)

    async def next_chunk(self, obs: Observation) -> np.ndarray | None:
        return self._chunks.pop(0) if self._chunks else None


def motion_test_chunks(horizon: int = 15) -> list[np.ndarray]:
    """Out-and-back moves on several joints at half and full speed, plus gripper close/open (DROID units)."""
    def chunk(v: dict[int, float], grip: float) -> np.ndarray:
        a = np.zeros((horizon, 8))
        for j, val in v.items():
            a[:, j - 1] = val
        a[:, 7] = grip
        return a
    return [
        chunk({1: +0.5, 4: +0.5, 6: -0.5}, 0.0),   # half speed out (0.1 rad/step)
        chunk({1: -0.5, 4: -0.5, 6: +0.5}, 1.0),   # half speed back, close gripper
        chunk({}, 1.0),                              # hold, gripper closed
        chunk({1: +1.0, 7: +1.0}, 1.0),              # full speed out (0.2 rad/step = 3 rad/s)
        chunk({1: -1.0, 7: -1.0}, 0.0),              # full speed back, open gripper
        chunk({}, 0.0),                              # hold
    ]
