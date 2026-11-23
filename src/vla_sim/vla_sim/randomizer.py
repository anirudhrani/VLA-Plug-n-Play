"""Object pose randomization for fair experiments (works on any scene).

Detection: a prim is a randomizable object if it is a dynamic rigid body (not kinematic) with collision
geometry, not part of an articulation (robot), and small (largest side < `max_size_m`).

Randomize: each selected object is moved by up to +-xy_range in world x/y and turned by up to +-yaw_range
about the world vertical through its own origin. The new pose keeps the object over the static surface it
rests on (detected from colliders below it) and apart from the other objects (rejection sampling).

Poses are written to the stage's *session layer* (never saved to the scene file) as three extra xform ops
put in front of the object's own ops:
    translate:vla_rand = p0 + d,  orient:vla_rand = R(axis_up, yaw),  translate:vla_rand_back = -p0
so  M = T(p0 + d) * R * T(-p0) * M_original  (rotation about the object's own origin, any op stack).
Timeline Stop + Play (Reset) restarts from the randomized pose; Restore removes the session overrides.
"""
from __future__ import annotations

import dataclasses
import math
import random

import omni.usd
from pxr import Gf, Sdf, Usd, UsdGeom, UsdPhysics

OPS = ("xformOp:translate:vla_rand", "xformOp:orient:vla_rand", "xformOp:translate:vla_rand_back")


@dataclasses.dataclass
class RandObject:
    path: str
    name: str
    default_pos: Gf.Vec3d          # world position of the origin, as loaded
    default_yaw_deg: float         # world heading of the object's x axis, as loaded
    radius: float                  # xy footprint radius (m), from the world bounding box
    bottom_z: float
    support: tuple | None = None   # (xmin, ymin, xmax, ymax) of the surface below it, or None
    support_name: str = ""
    enabled: bool = True
    offset: tuple = (0.0, 0.0, 0.0)   # current randomization: dx, dy (m), dyaw (deg)


def _bbox_cache():
    return UsdGeom.BBoxCache(Usd.TimeCode.Default(), [UsdGeom.Tokens.default_, UsdGeom.Tokens.render])


def _yaw_deg(m: Gf.Matrix4d) -> float:
    x = m.TransformDir(Gf.Vec3d(1, 0, 0))
    if abs(x[0]) < 1e-9 and abs(x[1]) < 1e-9:   # x axis vertical: use y
        x = m.TransformDir(Gf.Vec3d(0, 1, 0))
    return math.degrees(math.atan2(x[1], x[0]))


class Randomizer:
    def __init__(self, cfg: dict | None = None):
        cfg = cfg or {}
        self.xy_range = float(cfg.get("xy_range_m", 0.05))
        self.yaw_range = float(cfg.get("yaw_range_deg", 20.0))
        self.min_gap = float(cfg.get("min_gap_m", 0.02))
        self.edge_margin = float(cfg.get("edge_margin_m", 0.02))
        self.max_size = float(cfg.get("max_size_m", 0.4))
        self.objects: list[RandObject] = []
        self.seed: int | None = None
        self.message = ""

    # -- detection ---------------------------------------------------------------------------------
    def scan(self, stage=None) -> list[RandObject]:
        """Find randomizable objects; their *default* pose is the stage pose without our overrides."""
        stage = stage or omni.usd.get_context().get_stage()
        self.restore(stage)  # defaults must not include an old randomization
        bc = _bbox_cache()
        xc = UsdGeom.XformCache()
        robots = [p.GetPath() for p in stage.Traverse() if p.HasAPI(UsdPhysics.ArticulationRootAPI)]
        statics = []      # colliders that are not rigid bodies: tables, shelves, ground
        found = []
        it = iter(Usd.PrimRange(stage.GetPseudoRoot()))
        for p in it:
            if any(p.GetPath().HasPrefix(r) for r in robots):
                continue
            if p.HasAPI(UsdPhysics.RigidBodyAPI):
                rb = UsdPhysics.RigidBodyAPI(p)
                dynamic = (rb.GetRigidBodyEnabledAttr().Get() is not False) and not rb.GetKinematicEnabledAttr().Get()
                has_coll = any(q.HasAPI(UsdPhysics.CollisionAPI) for q in Usd.PrimRange(p))
                if dynamic and has_coll:
                    r = bc.ComputeWorldBound(p).ComputeAlignedRange()
                    size = r.GetSize()
                    if not r.IsEmpty() and max(size) < self.max_size:
                        m = xc.GetLocalToWorldTransform(p)
                        found.append(RandObject(
                            path=str(p.GetPath()), name=p.GetName(), default_pos=Gf.Vec3d(m.ExtractTranslation()),
                            default_yaw_deg=_yaw_deg(m), radius=0.5 * max(size[0], size[1]), bottom_z=r.GetMin()[2]))
                it.PruneChildren()
                continue
            if p.HasAPI(UsdPhysics.CollisionAPI):
                r = bc.ComputeWorldBound(p).ComputeAlignedRange()
                if not r.IsEmpty():
                    statics.append((p.GetPath(), r))
        for o in found:   # support: the static collider right below the object's footprint
            best = None
            for path, r in statics:
                lo, hi = r.GetMin(), r.GetMax()
                if lo[0] <= o.default_pos[0] <= hi[0] and lo[1] <= o.default_pos[1] <= hi[1] and hi[2] <= o.bottom_z + 0.02:
                    if best is None or hi[2] > best[1].GetMax()[2]:
                        best = (path, r)
            if best is not None:
                lo, hi = best[1].GetMin(), best[1].GetMax()
                if (hi[0] - lo[0]) < 20 and (hi[1] - lo[1]) < 20:   # an "infinite" ground plane gives no bound
                    o.support = (lo[0], lo[1], hi[0], hi[1])
                o.support_name = str(best[0])
        self.objects = found
        self.seed = None
        self.message = f"{len(found)} object(s) found" if found else "no randomizable objects in this scene"
        return found

    # -- randomize / restore -----------------------------------------------------------------------
    def randomize(self, seed: int | None = None) -> int:
        stage = omni.usd.get_context().get_stage()
        if seed is None:
            seed = random.randrange(1_000_000)
        rng = random.Random(seed)
        objs = [o for o in self.objects if o.enabled]
        placed: list[tuple[float, float, float]] = [  # objects that are not being moved stay where they are
            (o.default_pos[0], o.default_pos[1], o.radius) for o in self.objects if not o.enabled]
        result = {}
        for o in objs:
            for attempt in range(200):
                dx, dy = rng.uniform(-self.xy_range, self.xy_range), rng.uniform(-self.xy_range, self.xy_range)
                dyaw = rng.uniform(-self.yaw_range, self.yaw_range)
                x, y = o.default_pos[0] + dx, o.default_pos[1] + dy
                if o.support is not None:
                    xmin, ymin, xmax, ymax = o.support
                    m = o.radius + self.edge_margin
                    if not (xmin + m <= x <= xmax - m and ymin + m <= y <= ymax - m):
                        continue
                if all(math.hypot(x - px, y - py) >= o.radius + pr + self.min_gap for px, py, pr in placed):
                    break
            else:
                dx = dy = dyaw = 0.0   # no valid spot: leave this object at its default
                x, y = o.default_pos[0], o.default_pos[1]
            placed.append((x, y, o.radius))
            result[o.path] = (dx, dy, dyaw)
        self._remove_overrides(stage, objs)   # each object's own transform is read without old offsets
        with Usd.EditContext(stage, stage.GetSessionLayer()):
            for o in objs:
                self._apply(stage, o, *result[o.path])
        self.seed = seed
        self.message = f"randomized {len(objs)} object(s) · seed {seed}"
        return seed

    def _apply(self, stage, o: RandObject, dx: float, dy: float, dyaw: float) -> None:
        prim = stage.GetPrimAtPath(o.path)
        xf = UsdGeom.Xformable(prim)
        # work in the parent's frame: world offset/axis -> parent coordinates
        parent_w = UsdGeom.XformCache().GetParentToWorldTransform(prim)
        to_parent = parent_w.GetInverse()
        # object origin in parent coordinates, from its own ops (our overrides were removed before this)
        own = list(xf.GetOrderedXformOps())
        p0 = xf.GetLocalTransformation(Usd.TimeCode.Default()).ExtractTranslation()
        d = to_parent.TransformDir(Gf.Vec3d(dx, dy, 0.0))
        up = to_parent.TransformDir(Gf.Vec3d(0, 0, 1)).GetNormalized()
        q = Gf.Rotation(up, dyaw).GetQuat()
        attrs = {
            OPS[0]: (Sdf.ValueTypeNames.Double3, Gf.Vec3d(p0 + d)),
            OPS[1]: (Sdf.ValueTypeNames.Quatd, Gf.Quatd(q)),
            OPS[2]: (Sdf.ValueTypeNames.Double3, Gf.Vec3d(-p0)),
        }
        for name, (tn, val) in attrs.items():
            a = prim.GetAttribute(name) or prim.CreateAttribute(name, tn, custom=False)
            a.Set(val)
        order = list(OPS) + [op.GetOpName() for op in own]
        prim.GetAttribute("xformOpOrder").Set(order)
        o.offset = (dx, dy, dyaw)

    def restore(self, stage=None) -> None:
        """Remove our session-layer overrides (objects go back to the scene file's poses)."""
        stage = stage or omni.usd.get_context().get_stage()
        if stage is None:
            return
        self._remove_overrides(stage, self.objects)
        self.seed = None
        self.message = "defaults restored" if self.objects else self.message

    @staticmethod
    def _remove_overrides(stage, objs) -> None:
        layer = stage.GetSessionLayer()
        with Sdf.ChangeBlock():
            for o in objs:
                spec = layer.GetPrimAtPath(o.path)
                if spec is not None:
                    for name in OPS + ("xformOpOrder",):
                        if name in spec.properties:
                            spec.RemoveProperty(spec.properties[name])
                o.offset = (0.0, 0.0, 0.0)

    def snapshot(self) -> dict:
        """Current world poses + seed (for recordings)."""
        stage = omni.usd.get_context().get_stage()
        xc = UsdGeom.XformCache()
        out = {}
        for o in self.objects:
            m = xc.GetLocalToWorldTransform(stage.GetPrimAtPath(o.path))
            out[o.name] = {"path": o.path, "pos": [round(v, 4) for v in m.ExtractTranslation()],
                           "yaw_deg": round(_yaw_deg(m), 2), "offset": [round(v, 4) for v in o.offset],
                           "randomized": o.enabled and any(o.offset)}
        return {"seed": self.seed, "objects": out}
