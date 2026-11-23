"""VLA Sim extension entry point: panel, model processes, ROS I/O, episode loop."""
from __future__ import annotations

import asyncio
import time
from pathlib import Path

import carb
import omni.ext
import omni.kit.app
import omni.timeline
import omni.usd
import yaml

from .bridge_source import BridgeSource
from .catalog import load_catalog
from .process_mgr import ModelState, ProcessManager
from .ui import VlaSimWindow

_instance: "VlaSimExtension | None" = None
DEFAULT_SCENE = "scene_droid_scene1.yaml"   # used when the open stage matches no scene config


def get_instance() -> "VlaSimExtension | None":
    """The running extension (for scripting/tests via the Isaac Python console)."""
    return _instance


class VlaSimExtension(omni.ext.IExt):
    def on_startup(self, ext_id: str) -> None:
        ext_path = Path(omni.kit.app.get_app().get_extension_manager().get_extension_path(ext_id))
        self.repo_root = ext_path.parents[1]  # <repo>/src/vla_sim -> <repo>
        try:
            models = load_catalog(self.repo_root)
        except Exception as e:
            carb.log_error(f"[vla_sim] could not read configs/models.yaml: {e}")
            models = []
        self._window = None
        self.sim = self.ros = self.executor = None
        self._episode_task: asyncio.Task | None = None
        self.source = None                 # BridgeSource of the running/last episode
        self.last_outcome: str = ""
        self.scene: dict = {}
        self.obs_seq = 0
        self.last_obs: str = "–"
        self._obs_task: asyncio.Task | None = None
        self._reset_task: asyncio.Task | None = None
        self._tick_task: asyncio.Task | None = None
        self.task = None                   # SuccessCheck (M6)
        self.record_enabled = False        # panel "Record" checkbox
        self.randomizer = None             # object pose randomization for the current scene
        self.rand_version = 0              # bumped when the object list changes (panel rebuilds its rows)
        self._rand_task: asyncio.Task | None = None
        self.last_recording = ""           # folder of the last recording, or a recorder message
        self.scene_name = "–"
        self._stage_sub = None
        try:
            from .ros_io import RosIO
            # ROS topics and camera names are the same in every scene config
            self.ros = RosIO(self._scene_configs()[DEFAULT_SCENE])
            self._select_scene()
            self._stage_sub = omni.usd.get_context().get_stage_event_stream().create_subscription_to_pop(
                self._on_stage_event, name="vla_sim scene select")
        except Exception as e:
            carb.log_error(f"[vla_sim] sim/ROS I/O unavailable: {e}")
            self.last_obs = f"ROS/sim I/O unavailable: {e}"
        self.last_chunk: str = "–"
        self.chunk = None                 # latest ActionChunk (consumed by the executor from M4)
        self._request_wall: dict[int, float] = {}
        if self.ros is not None:
            self.ros.on_chunk = self._on_chunk
            self.ros.on_status = lambda _s: self._refresh()
        self.procs = ProcessManager(
            self.repo_root, on_change=self._refresh,
            bridge_status=lambda: self.ros.bridge_status if self.ros else "",
            reset_bridge_status=lambda: self.ros.reset_bridge_status() if self.ros else None,
        )
        self._window = VlaSimWindow(models, self.procs, on_instruction=self._on_instruction,
                                    on_observe=self.observe_once, status=self,
                                    on_stop=self.stop_episode, on_test_motion=self.test_motion,
                                    on_reset=self.reset_scene, on_randomize=self.randomize_objects,
                                    on_restore=self.restore_objects, on_rescan=self.rescan_objects)
        self._tick_task = asyncio.ensure_future(self._tick())
        global _instance
        _instance = self
        carb.log_info(f"[vla_sim] started; repo root {self.repo_root}; models {[m.name for m in models]}")

    # -- scene selection: one config per USD stage (configs/scene_*.yaml, key `usd`) --------------
    def _scene_configs(self) -> dict[str, dict]:
        out = {}
        for f in sorted((self.repo_root / "configs").glob("scene_*.yaml")):
            cfg = yaml.safe_load(f.read_text()) or {}
            cfg["_file"] = f.name
            out[f.name] = cfg
        return out

    def _select_scene(self) -> None:
        """(Re)build the scene-dependent parts (capture, executor, success check) for the open stage."""
        url = omni.usd.get_context().get_stage_url() or ""
        path = url.removeprefix("file:")
        configs = self._scene_configs()
        chosen = next((c for c in configs.values() if c.get("usd") and
                       Path(path).resolve() == (self.repo_root / c["usd"]).resolve()), configs[DEFAULT_SCENE])
        if self.scene.get("_file") == chosen["_file"] and self.sim is not None:
            return
        for io in (self.task, self.executor, self.sim):
            if io is not None:
                try:
                    io.destroy()
                except Exception as e:  # noqa: BLE001
                    carb.log_warn(f"[vla_sim] scene switch: {e}")
        from .sim_io import SimIO
        from .executor import Executor
        self.scene = chosen
        self.sim = SimIO(chosen)
        self.executor = Executor(self.sim, chosen, on_change=lambda: self._refresh())
        self.task = None
        if chosen.get("task"):
            from .task import SuccessCheck
            self.task = SuccessCheck(chosen["task"])
        if self.ros is not None:
            self.ros._cams = chosen["cameras"]   # frame ids
        if self.randomizer is not None:
            try:
                self.randomizer.restore()
            except Exception:  # noqa: BLE001  (old stage may already be gone)
                pass
        from .randomizer import Randomizer
        self.randomizer = Randomizer(chosen.get("randomization"))
        self.rescan_objects()
        self.scene_name = chosen["_file"].removeprefix("scene_").removesuffix(".yaml")
        carb.log_info(f"[vla_sim] scene config {chosen['_file']} for stage {path or '(unsaved)'}")
        self._refresh()

    def _on_stage_event(self, event) -> None:
        if event.type == int(omni.usd.StageEventType.OPENED):
            if self.episode_running:
                self.executor.stop()
            try:
                self.scene = {}            # new stage: always rebuild (and rescan objects), even for the same config
                self._select_scene()
            except Exception as e:  # noqa: BLE001
                carb.log_error(f"[vla_sim] scene select: {e}")

    def _refresh(self) -> None:
        # a model episode cannot outlive its model
        if self.source is not None and self.episode_running and self.procs.state != ModelState.READY:
            self.executor.stop()
        if self._window:
            self._window.refresh()

    async def _tick(self) -> None:
        """Refresh the panel's live task line a few times per second (it changes on physics steps)."""
        while True:
            await asyncio.sleep(0.25)
            if self._window:
                self._window.refresh_task()

    @property
    def task_status(self) -> tuple[str, str]:
        if self.task is None:
            return "no task configured", ""
        if omni.timeline.get_timeline_interface().is_stopped():
            try:
                self.task.update()   # no physics steps while stopped: evaluate the USD poses directly
            except Exception as e:  # noqa: BLE001
                return f"check failed: {e}", ""
        return self.task.status, self.task.detail

    def _next_seq(self) -> int:
        self.obs_seq += 1
        return self.obs_seq

    # -- M5: instruction -> episode ---------------------------------------------------------------
    def _on_instruction(self, text: str) -> None:
        """Enter/Send: publish the instruction; start an episode if none is running (else it applies from
        the next request of the running episode)."""
        carb.log_info(f"[vla_sim] instruction: {text!r}")
        if self.ros is None or self.executor is None:
            return
        self.ros.publish_instruction(text)
        if self.procs.state == ModelState.READY and not self.episode_running:
            self._episode_task = asyncio.ensure_future(self._run_episode(text))

    async def _run_episode(self, text: str) -> None:
        model = self.procs.model.raw
        ep = self.scene.get("episode", {})
        rec = None
        app = omni.kit.app.get_app()
        try:
            # the bridge must have the new instruction before the first request goes out
            t0 = time.monotonic()
            while self.ros.instruction_ack != text:
                if time.monotonic() - t0 > 3.0:
                    raise RuntimeError("bridge did not acknowledge the instruction")
                await asyncio.sleep(0.02)
            tl = omni.timeline.get_timeline_interface()
            if tl.is_stopped():
                tl.play()
                for _ in range(10):  # let physics initialize
                    await app.next_update_async()
            self.source = BridgeSource(self.ros, self._next_seq, float(ep.get("chunk_timeout_s", 10.0)))
            hz = float(model.get("control_hz", 15))
            rec = None
            rand_info = self.randomizer.snapshot() if self.randomizer is not None else None
            if self.record_enabled:
                from .recorder import Recorder
                rec = Recorder(self.repo_root / "outputs" / "recordings", self.scene_name, self.procs.model.name,
                               text, fps=int(round(hz)))
                self.executor.frame_hook = lambda obs, step, t: rec.add_frame(obs.images, step, t)
                self.source.on_request = lambda seq, obs: rec.add_model_input(seq, obs.images)
                self.last_recording = f"recording → {rec.dir.relative_to(self.repo_root)}"
                self._refresh()
            self.last_outcome = await self.executor.run(
                self.source, action_type=model["action_type"], n_exec=int(model.get("open_loop_horizon", 8)),
                max_steps=int(round(float(ep.get("timeout_s", 30.0)) * hz)), control_hz=hz,
                is_done=lambda: "success" if self.task is not None and self.task.success else None,
            )
        except Exception as e:  # noqa: BLE001
            self.last_outcome = f"error: {e}"
            carb.log_error(f"[vla_sim] episode: {e}")
        finally:
            if self.executor is not None:
                self.executor.frame_hook = None
        if rec is not None:
            self.last_recording = f"saving {rec.dir.relative_to(self.repo_root)} …"
            self._refresh()
            try:
                path = await asyncio.get_event_loop().run_in_executor(
                    None, rec.finish, self.last_outcome, self.executor.status, list(self.executor.log),
                    self.executor.dof_names, list(self.source.latencies_ms) if self.source else [],
                    {"open_loop_horizon": int(model.get("open_loop_horizon", 8)), "control_hz": hz,
                     "physics_steps_per_action": self.executor.steps_per_action,
                     "task_check": self.task.status if self.task else None,
                     "randomization": rand_info})
                warn = f" ({rec.error})" if rec.error else ""
                self.last_recording = f"saved {path.relative_to(self.repo_root)} · {rec.n_frames} frames{warn}"
            except Exception as e:  # noqa: BLE001
                self.last_recording = f"recording failed: {e}"
                carb.log_error(f"[vla_sim] recorder: {e}")
        carb.log_info(f"[vla_sim] episode '{text}' ended: {self.last_outcome} · {self.episode_status}")
        self._refresh()

    def _on_chunk(self, chunk) -> None:
        self.chunk = chunk
        if self.source is not None:
            self.source.on_chunk(chunk)
        t_req = self._request_wall.pop(chunk.seq, None)
        latency = f" · {1e3 * (chunk.received_wall - t_req):.0f} ms after request" if t_req else ""
        h, d = chunk.actions.shape
        first = " ".join(f"{v:+.2f}" for v in chunk.actions[0])
        self.last_chunk = f"chunk seq {chunk.seq} · {h}x{d} · obs t={chunk.stamp_s:.3f}s{latency}\nstep 0: [{first}]"
        carb.log_info(f"[vla_sim] {self.last_chunk}")
        self._refresh()

    @property
    def bridge_status(self) -> str:
        return self.ros.bridge_status if self.ros else ""

    # -- M4: executor --------------------------------------------------------------------------
    @property
    def episode_status(self) -> str:
        if not self.executor:
            return "unavailable"
        lat = self.source.latency_summary() if self.source is not None else ""
        return f"{self.executor.status} · {lat}" if lat else self.executor.status

    @property
    def episode_running(self) -> bool:
        return bool(self.executor and self.executor.running)

    def stop_episode(self) -> None:
        if self.executor:
            self.executor.stop()

    def test_motion(self) -> None:
        """Run scripted DROID-format chunks through the full lockstep loop (no model needed)."""
        if self.executor is None or self.episode_running:
            return
        from .executor import ScriptedSource, motion_test_chunks
        chunks = motion_test_chunks()
        self.source = None
        self._episode_task = asyncio.ensure_future(self.executor.run(
            ScriptedSource(chunks), action_type="droid_joint_velocity", n_exec=8, max_steps=8 * len(chunks) + 1))

    # -- randomization -----------------------------------------------------------------------------
    @property
    def randomizing(self) -> bool:
        return bool(self._rand_task and not self._rand_task.done())

    def rescan_objects(self) -> None:
        if self.randomizer is None or self.episode_running or self.randomizing:
            return
        try:
            self.randomizer.scan()
        except Exception as e:  # noqa: BLE001
            self.randomizer.message = f"scan failed: {e}"
            carb.log_error(f"[vla_sim] randomization scan: {e}")
        self.rand_version += 1
        self._refresh()

    def randomize_objects(self) -> None:
        self._start_rand(lambda: self.randomizer.randomize())

    def restore_objects(self) -> None:
        self._start_rand(self.randomizer.restore)

    def _start_rand(self, fn) -> None:
        if self.randomizer is None or not self.randomizer.objects or self.episode_running or self.randomizing or self.resetting:
            return
        self._rand_task = asyncio.ensure_future(self._apply_rand(fn))

    async def _apply_rand(self, fn) -> None:
        """Poses are authored with the timeline stopped (PhysX re-reads them on Play), then the sim plays
        so objects settle; Reset (Stop + Play) later restarts from these same poses."""
        app = omni.kit.app.get_app()
        tl = omni.timeline.get_timeline_interface()
        try:
            tl.stop()
            for _ in range(3):
                await app.next_update_async()
            fn()
            if self.task is not None:
                self.task.reset()
            tl.play()
            for _ in range(10):
                await app.next_update_async()
        except Exception as e:  # noqa: BLE001
            self.randomizer.message = f"failed: {e}"
            carb.log_error(f"[vla_sim] randomization: {e}")
        self._refresh()

    # -- M6: reset ---------------------------------------------------------------------------------
    @property
    def resetting(self) -> bool:
        return bool(self._reset_task and not self._reset_task.done())

    def reset_scene(self) -> None:
        if self.resetting or (self._obs_task and not self._obs_task.done()):
            return
        self._reset_task = asyncio.ensure_future(self._reset_scene())

    async def _reset_scene(self) -> None:
        """Stop any episode, then Stop + Play the timeline: PhysX restores the saved stage state (arm at the
        DROID home pose, gripper open, cube and bowl at their saved poses, all velocities zero)."""
        app = omni.kit.app.get_app()
        tl = omni.timeline.get_timeline_interface()
        try:
            if self.episode_running:
                self.executor.stop()
            if self._episode_task and not self._episode_task.done():
                await self._episode_task
            self._refresh()
            tl.stop()
            for _ in range(3):
                await app.next_update_async()
            if self.task is not None:
                self.task.reset()
            tl.play()
            for _ in range(10):  # let physics initialize before anything reads the articulation
                await app.next_update_async()
            self.last_outcome = ""
            if self.executor:
                self.executor.status = "reset · idle"
            carb.log_info("[vla_sim] scene reset")
        except Exception as e:  # noqa: BLE001
            carb.log_error(f"[vla_sim] reset failed: {e}")
            if self.executor:
                self.executor.status = f"reset failed: {e}"
        self._refresh()

    # -- M2: one on-demand observation (debug button) -------------------------------------------
    def observe_once(self) -> None:
        if self.sim is None or self.ros is None or self.episode_running or (self._obs_task and not self._obs_task.done()):
            return
        self._obs_task = asyncio.ensure_future(self._observe_once())

    async def _observe_once(self) -> None:
        tl = omni.timeline.get_timeline_interface()
        was_playing = tl.is_playing()
        t0 = time.perf_counter()
        try:
            obs = await self.sim.observe()
            self.obs_seq += 1
            self._request_wall[self.obs_seq] = time.monotonic()
            self.ros.publish_observation(obs, self.obs_seq)
            shapes = ", ".join(f"{k} {v.shape[1]}x{v.shape[0]}" for k, v in obs.images.items())
            self.last_obs = (f"seq {self.obs_seq} · t={obs.stamp_s:.3f}s · {shapes} · joints from {obs.source} · "
                             f"gripper {obs.gripper:.2f} · {1e3 * (time.perf_counter() - t0):.0f} ms")
        except Exception as e:
            carb.log_error(f"[vla_sim] observe failed: {e}")
            self.last_obs = f"observe failed: {e}"
        finally:
            if was_playing:
                tl.play()
            self._refresh()

    def on_shutdown(self) -> None:
        global _instance
        _instance = None
        self._stage_sub = None
        if self._obs_task and not self._obs_task.done():
            self._obs_task.cancel()
        if getattr(self, "randomizer", None) is not None:
            try:
                self.randomizer.restore()   # leave no hidden session-layer overrides behind
            except Exception:  # noqa: BLE001
                pass
        for t in (self._episode_task, self._reset_task, self._tick_task, getattr(self, "_rand_task", None)):
            if t and not t.done():
                t.cancel()
        for io in (getattr(self, "task", None), getattr(self, "executor", None), getattr(self, "sim", None),
                   getattr(self, "ros", None)):
            if io is not None:
                try:
                    io.destroy()
                except Exception as e:
                    carb.log_warn(f"[vla_sim] shutdown: {e}")
        self.sim = self.ros = self.executor = self.task = None
        if getattr(self, "procs", None):
            self.procs.shutdown()  # never leave a model server holding GPU memory
        if self._window:
            self._window.destroy()
            self._window = None
