"""Task success check (inference layer, plan §6): "the cube is in the bowl".

The cube's center, expressed in the bowl's own local frame (so a shifted, turned or tipped bowl still works), must be
  - horizontally within `radius_frac` x the bowl's rim radius from the bowl's axis, and
  - vertically between the bowl's bottom and its rim,
continuously for `hold_s` of sim time (so a cube passing through or bouncing out does not count).

Poses come from PhysX while the simulation runs (with Fabric, USD is not updated during simulation)
and from USD when it is stopped. Evaluated before every physics step, so it also works with no episode
running (e.g. dropping the cube into the bowl by hand).
"""
from __future__ import annotations

import omni.physx
import omni.timeline
import omni.usd
from pxr import Gf, Usd, UsdGeom


class SuccessCheck:
    def __init__(self, cfg: dict):
        self.object_path = cfg["object"]
        self.container_path = cfg["container"]
        self.radius_frac = float(cfg.get("radius_frac", 0.8))
        self.hold_s = float(cfg.get("hold_s", 0.3))
        self.name = cfg.get("name", "object in container")
        self._px = omni.physx.get_physx_interface()
        self._tl = omni.timeline.get_timeline_interface()
        self._geom = None            # (center_xy, z_bottom, z_rim, radius, q0): see _container_geom
        self._inside_since: float | None = None
        self.inside = False          # condition holds right now
        self.success = False         # condition has held for hold_s
        self.detail = "–"
        self._sub = self._px.subscribe_physics_step_events(self._on_step)

    # -- geometry ---------------------------------------------------------------------------------
    def _container_geom(self, stage):
        """Container shape in its own (scaled) local frame, so it does not depend on how it is turned.

        The box comes from the untransformed bound (object space) times the prim's scale. "Up" is the local
        axis that points closest to world up in the loaded pose (meshes are often authored Y-up).
        A world-aligned box would grow when the container is yawed (e.g. by randomization: +25% at 18 deg).
        """
        if self._geom is None:
            prim = stage.GetPrimAtPath(self.container_path)
            if not prim.IsValid():
                raise RuntimeError(f"container prim {self.container_path} not found")
            bc = UsdGeom.BBoxCache(Usd.TimeCode.Default(), [UsdGeom.Tokens.default_, UsdGeom.Tokens.render])
            r = bc.ComputeUntransformedBound(prim).ComputeAlignedRange()
            xf = Gf.Transform(UsdGeom.Xformable(prim).ComputeLocalToWorldTransform(Usd.TimeCode.Default()))
            sc, q = xf.GetScale(), xf.GetRotation().GetQuat()
            lo = [min(r.GetMin()[i] * sc[i], r.GetMax()[i] * sc[i]) for i in range(3)]
            hi = [max(r.GetMin()[i] * sc[i], r.GetMax()[i] * sc[i]) for i in range(3)]
            ups = [q.Transform(Gf.Vec3d(*(1.0 if j == i else 0.0 for j in range(3))))[2] for i in range(3)]
            k = max(range(3), key=lambda i: abs(ups[i]))
            sign = 1.0 if ups[k] > 0 else -1.0
            a, b = [i for i in range(3) if i != k]
            z_bot, z_rim = (lo[k], hi[k]) if sign > 0 else (-hi[k], -lo[k])
            self._geom = (k, sign, (a, b), ((lo[a] + hi[a]) / 2, (lo[b] + hi[b]) / 2),
                          z_bot, z_rim, min(hi[a] - lo[a], hi[b] - lo[b]) / 2)
        return self._geom

    @staticmethod
    def _usd_pose(stage, path: str) -> tuple[Gf.Vec3d, Gf.Quatd]:
        m = Gf.Transform(UsdGeom.Xformable(stage.GetPrimAtPath(path)).ComputeLocalToWorldTransform(Usd.TimeCode.Default()))
        return Gf.Vec3d(m.GetTranslation()), Gf.Quatd(m.GetRotation().GetQuat())

    def _world_pose(self, stage, path: str) -> tuple[Gf.Vec3d, Gf.Quatd]:
        """World position + orientation of a rigid body (PhysX when simulating, else USD)."""
        if self._tl.is_stopped():   # PhysX has no bodies then (and logs an error if asked)
            return self._usd_pose(stage, path)
        t = self._px.get_rigidbody_transformation(path)
        if t.get("ret_val"):
            p, q = t["position"], t["rotation"]   # rotation is (x, y, z, w)
            return Gf.Vec3d(*p), Gf.Quatd(float(q[3]), Gf.Vec3d(float(q[0]), float(q[1]), float(q[2])))
        return self._usd_pose(stage, path)

    def _evaluate(self) -> bool:
        stage = omni.usd.get_context().get_stage()
        k, sign, (a, b), (ca, cb), z_bot, z_rim, radius = self._container_geom(stage)
        obj_p, _ = self._world_pose(stage, self.object_path)
        con_p, con_q = self._world_pose(stage, self.container_path)
        rel = con_q.GetInverse().Transform(obj_p - con_p)     # object in the container's local frame
        d_xy = ((rel[a] - ca) ** 2 + (rel[b] - cb) ** 2) ** 0.5
        h = sign * rel[k]
        ok_xy = d_xy <= self.radius_frac * radius
        ok_z = z_bot <= h <= z_rim
        self.detail = (f"{'in' if ok_xy else 'out of'} footprint ({100 * d_xy:.1f}/{100 * self.radius_frac * radius:.1f} cm), "
                       f"{'below' if ok_z else 'not below'} rim (h={100 * (h - z_bot):.1f}/{100 * (z_rim - z_bot):.1f} cm)")
        return ok_xy and ok_z

    # -- updates --------------------------------------------------------------------------------
    def _on_step(self, _dt: float) -> None:
        try:
            self.update()
        except Exception as e:  # noqa: BLE001
            self.detail = f"check failed: {e}"

    def update(self) -> bool:
        """Re-evaluate now; returns `success`."""
        self.inside = self._evaluate()
        now = self._tl.get_current_time()
        if not self.inside:
            self._inside_since = None
        elif self._inside_since is None or now < self._inside_since:   # (sim time restarts on Stop/Play)
            self._inside_since = now
        self.success = self.inside and now - self._inside_since >= self.hold_s
        return self.success

    def reset(self) -> None:
        self._inside_since, self.inside, self.success, self._geom = None, False, False, None

    @property
    def status(self) -> str:
        state = "SUCCESS" if self.success else ("inside, settling…" if self.inside else "not yet")
        return f"{self.name}: {state}"

    def destroy(self) -> None:
        self._sub = None
