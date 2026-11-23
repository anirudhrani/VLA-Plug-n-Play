"""Reads the sim: camera RGB (Replicator annotators) and robot joint state (PhysX tensors / USD).

This is the only module that knows how to get observations out of Isaac. It returns plain numpy.
"""
from __future__ import annotations

import dataclasses
import math

import carb
import carb.settings
import numpy as np
import omni.kit.app
import omni.physics.tensors as physics_tensors
import omni.replicator.core as rep
import omni.timeline
import omni.usd
from pxr import Sdf, Usd, UsdPhysics


@dataclasses.dataclass
class Observation:
    stamp_s: float                 # sim (timeline) time the observation was taken at
    images: dict[str, np.ndarray]  # camera name -> HxWx3 uint8 RGB
    joint_names: list[str]         # arm joints + gripper joint
    joint_pos: np.ndarray          # rad, same order as joint_names
    gripper: float                 # 0 = open, 1 = closed
    source: str                    # "physx" (sim running/paused) or "usd" (sim stopped)


# Replicator hides these viewport gizmos while capturing, but only restores the ones that had an explicit
# value beforehand. Unset means "shown", so we snapshot them ourselves and restore unset ones to True.
_GIZMO_SETTINGS = ("/app/viewport/grid/enabled", "/app/viewport/outline/enabled",
                   "/app/viewport/show/lights", "/app/viewport/show/camera")


def _snapshot_gizmos() -> dict:
    s = carb.settings.get_settings()
    return {k: s.get(k) for k in _GIZMO_SETTINGS}


def _restore_gizmos(snapshot: dict) -> None:
    s = carb.settings.get_settings()
    for k, v in snapshot.items():
        s.set_bool(k, True if v is None else bool(v))


async def _stop_orchestrator() -> None:
    if rep.orchestrator.get_status() != rep.orchestrator.Status.STOPPED:
        await rep.orchestrator.stop_async()


RENDER_ROOT = Sdf.Path("/Render")
RENDER_SETTLE_FRAMES = 10
RP_PREFIX = "vla_"


class CameraCapture:
    """One render product + RGB annotator per camera.

    Kit authors render products into the stage's *root* layer (that is where the viewport's own product
    lives), so they would end up in the saved scene file. To keep the user's scene untouched, products exist
    only while capture is open (one Observe, or one episode): `open()` snapshots /Render, `close()` removes
    our products and restores every /Render attribute and relationship we changed.
    """

    def __init__(self, cameras: dict, rt_subframes: int = 1):
        self._cfg = cameras
        self.rt_subframes = int(rt_subframes)
        self._rps: dict = {}
        self._annots: dict = {}
        self._backup: Sdf.Layer | None = None
        self._stage = None

    @property
    def is_open(self) -> bool:
        return bool(self._annots)

    def open(self) -> None:
        if self.is_open:
            return
        stage = omni.usd.get_context().get_stage()
        for name, c in self._cfg.items():
            if not stage.GetPrimAtPath(c["prim"]).IsValid():
                raise RuntimeError(f"camera '{name}' not found at {c['prim']}; is sim/franka_workspace.usd open?")
        root = stage.GetRootLayer()
        _remove_our_specs(root)  # leftovers from a session that saved mid-episode
        self._backup = Sdf.Layer.CreateAnonymous("vla_render_backup")
        if root.GetPrimAtPath(RENDER_ROOT):
            Sdf.CopySpec(root, RENDER_ROOT, self._backup, RENDER_ROOT)
        self._stage = stage
        try:
            for name, c in self._cfg.items():
                rp = rep.create.render_product(c["prim"], tuple(c["resolution"]), name=f"{RP_PREFIX}{name}")
                annot = rep.AnnotatorRegistry.get_annotator("rgb")
                annot.attach([rp])
                self._rps[name], self._annots[name] = rp, annot
        except Exception:
            self.close()
            raise

    async def capture(self) -> dict[str, np.ndarray]:
        """Render a fresh frame of the current state (timeline is not advanced) and read all cameras."""
        self.open()
        # rt_subframes > 1: the RTX renderer accumulates over frames, so a single pass right after motion still
        # shows edges where things were a frame or two ago; extra subframes let it converge on the paused state
        await rep.orchestrator.step_async(rt_subframes=self.rt_subframes, pause_timeline=False, delta_time=0.0,
                                          wait_for_render=True)
        out = {}
        for name, annot in self._annots.items():
            data = annot.get_data()
            w, h = self._cfg[name]["resolution"]
            if data is None or getattr(data, "size", 0) == 0:
                raise RuntimeError(f"camera '{name}' returned no data")
            out[name] = np.ascontiguousarray(np.asarray(data).reshape(h, w, -1)[..., :3], dtype=np.uint8)
        return out

    def close(self) -> None:
        for name, annot in self._annots.items():
            try:
                annot.detach([self._rps[name]])
            except Exception:
                pass
        for rp in self._rps.values():
            try:
                rp.destroy()
            except Exception:
                pass
        self._rps, self._annots = {}, {}

    def restore_render(self) -> None:
        """Undo every /Render change in the root layer since open(). Call after Replicator has fully stopped."""
        stage = omni.usd.get_context().get_stage()
        if self._backup is not None and stage is self._stage and stage is not None:
            _restore_render(stage.GetRootLayer(), self._backup)
        self._backup, self._stage = None, None

    def destroy(self) -> None:
        self.close()
        self.restore_render()


def _remove_our_specs(layer: Sdf.Layer) -> None:
    """Drop vla_* render product prims and any relationship targets pointing at them."""
    doomed, rels = [], []

    def visit(path):
        if path.IsPrimPath() and path.name.startswith(RP_PREFIX):
            doomed.append(path)
        elif path.IsPropertyPath() and layer.GetRelationshipAtPath(path):
            rels.append(path)

    if not layer.GetPrimAtPath(RENDER_ROOT):
        return
    layer.Traverse(RENDER_ROOT, visit)
    for path in rels:
        tl = layer.GetRelationshipAtPath(path).targetPathList
        for listname in ("explicitItems", "prependedItems", "appendedItems"):
            items = list(getattr(tl, listname))
            keep = [t for t in items if not t.name.startswith(RP_PREFIX)]
            if len(keep) != len(items):
                setattr(tl, listname, keep)
    for path in sorted(doomed, key=lambda p: -p.pathElementCount):
        parent = layer.GetPrimAtPath(path.GetParentPath())
        if parent and path.name in parent.nameChildren:
            del parent.nameChildren[path.name]


def _restore_render(root: Sdf.Layer, backup: Sdf.Layer) -> None:
    """Undo what capture changed under /Render in the root layer (attribute defaults + our prims/targets)."""
    _remove_our_specs(root)
    if not root.GetPrimAtPath(RENDER_ROOT):
        return
    changed = []

    def visit(path):
        if not path.IsPropertyPath():
            return
        cur = root.GetAttributeAtPath(path)
        if cur is None:
            return
        old = backup.GetAttributeAtPath(path)
        if old is None:
            changed.append((path, None))
        elif cur.default != old.default or cur.HasDefaultValue() != old.HasDefaultValue():
            changed.append((path, old))

    root.Traverse(RENDER_ROOT, visit)
    for path, old in changed:
        cur = root.GetAttributeAtPath(path)
        if old is None:
            owner = root.GetPrimAtPath(path.GetPrimPath())
            owner.RemoveProperty(cur)
        elif old.HasDefaultValue():
            cur.default = old.default
        else:
            cur.ClearDefaultValue()


class JointReader:
    """Joint positions from PhysX while the sim is running/paused, from the authored USD state when stopped."""

    def __init__(self, robot: dict):
        self._cfg = robot
        self.joint_names = list(robot["arm_joints"]) + [robot["gripper_joint"]]
        self._view = None
        self._art = None
        self._idx = None

    def _physx_positions(self) -> np.ndarray | None:
        if omni.timeline.get_timeline_interface().is_stopped():
            self._view = self._art = None
            return None
        if self._view is None or not self._view.is_valid:
            self._view = physics_tensors.create_simulation_view("numpy", omni.usd.get_context().get_stage_id())
            self._art = self._view.create_articulation_view(self._cfg["articulation"])
            dof_names = list(self._art.shared_metatype.dof_names)
            missing = [n for n in self.joint_names if n not in dof_names]
            if missing:
                raise RuntimeError(f"joints {missing} not in articulation DOFs {dof_names}")
            self._idx = [dof_names.index(n) for n in self.joint_names]
        return np.asarray(self._art.get_dof_positions())[0, self._idx].astype(np.float64)

    def articulation(self):
        """(PhysX articulation view, dof names); the simulation must be playing or paused."""
        if self._physx_positions() is None:
            raise RuntimeError("simulation is stopped")
        return self._art, list(self._art.shared_metatype.dof_names)

    def _usd_positions(self) -> np.ndarray:
        stage = omni.usd.get_context().get_stage()
        root = stage.GetPrimAtPath(self._cfg["articulation"])
        by_name = {p.GetName(): p for p in Usd.PrimRange(root, Usd.TraverseInstanceProxies())
                   if p.IsA(UsdPhysics.RevoluteJoint)}
        q = []
        for n in self.joint_names:
            attr = by_name[n].GetAttribute("state:angular:physics:position")
            q.append(math.radians(attr.Get() or 0.0) if attr and attr.HasValue() else 0.0)
        return np.array(q, dtype=np.float64)

    def read(self) -> tuple[np.ndarray, str]:
        q = self._physx_positions()
        return (q, "physx") if q is not None else (self._usd_positions(), "usd")

    def gripper_closedness(self, finger_rad: float) -> float:
        lo, hi = math.radians(self._cfg["gripper_open_deg"]), math.radians(self._cfg["gripper_closed_deg"])
        return float(np.clip((finger_rad - lo) / (hi - lo), 0.0, 1.0))

    def destroy(self) -> None:
        self._view = self._art = None


class SimIO:
    def __init__(self, scene_cfg: dict):
        self.cameras = CameraCapture(scene_cfg["cameras"], scene_cfg.get("capture", {}).get("rt_subframes", 1))
        self.joints = JointReader(scene_cfg["robot"])
        self._gizmos: dict | None = None  # viewport gizmo settings before capture started
        self._capture_on_play = None      # Replicator captureOnPlay before capture started

    async def observe(self, keep_capture_open: bool = False) -> Observation:
        """Freeze the world (pause if playing), render fresh frames, read joints. Caller decides when to resume.

        keep_capture_open=True keeps the camera render products alive (use during an episode, then call
        close_capture()); otherwise they are removed right after this observation.
        """
        tl = omni.timeline.get_timeline_interface()
        if tl.is_playing():
            tl.pause()
        self._begin_capture()
        try:
            images = await self.cameras.capture()
            q, source = self.joints.read()
            stamp = float(tl.get_current_time())
        finally:
            if not keep_capture_open:
                await self.close_capture()
        return Observation(
            stamp_s=stamp, images=images, joint_names=self.joints.joint_names,
            joint_pos=q, gripper=self.joints.gripper_closedness(q[-1]), source=source,
        )

    def _begin_capture(self) -> None:
        if self._gizmos is not None:
            return  # already capturing (episode)
        settings = carb.settings.get_settings()
        self._gizmos = _snapshot_gizmos()
        # With capture-on-play on (Replicator's default), the orchestrator drives the timeline: stopping it
        # stops the sim (resetting physics) and Play hands control to Replicator. We own the timeline.
        self._capture_on_play = settings.get("/omni/replicator/captureOnPlay")
        settings.set_bool("/omni/replicator/captureOnPlay", False)

    async def close_capture(self) -> None:
        """Remove capture render products and stop Replicator's orchestrator.

        While the orchestrator runs it hides viewport gizmos (grid, outlines, light/camera icons) and changes
        a few render settings; stopping it restores them.
        """
        self.cameras.close()
        await _stop_orchestrator()
        # the renderer re-authors some RTX attributes on the viewport product a few frames after
        # Replicator stops; let that settle before reverting /Render
        for _ in range(RENDER_SETTLE_FRAMES):
            await omni.kit.app.get_app().next_update_async()
        self.cameras.restore_render()
        self._restore_viewport()

    def _restore_viewport(self) -> None:
        if self._gizmos is not None:
            _restore_gizmos(self._gizmos)
            self._gizmos = None
            if self._capture_on_play is not None:
                carb.settings.get_settings().set_bool("/omni/replicator/captureOnPlay", bool(self._capture_on_play))
            self._capture_on_play = None

    def destroy(self) -> None:
        self.cameras.close()
        self.joints.destroy()
        if rep.orchestrator.get_status() != rep.orchestrator.Status.STOPPED:
            rep.orchestrator.stop()
        self.cameras.restore_render()
        self._restore_viewport()
        carb.log_info("[vla_sim] sim io destroyed")
