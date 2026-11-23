"""VLA Sim panel: model dropdown + load/unload (Load switches models), instruction box, run controls."""
from __future__ import annotations

from typing import Callable

import carb.input
import omni.appwindow
import omni.ui as ui

from .catalog import ModelSpec
from .process_mgr import ModelState, ProcessManager

_STATE_COLOR = {
    ModelState.NO_MODEL: 0xFF9A9A9A,
    ModelState.LOADING: 0xFF33B5FF,   # ABGR: orange
    ModelState.UNLOADING: 0xFF33B5FF,
    ModelState.READY: 0xFF4CC24C,     # green
    ModelState.FAILED: 0xFF4040E0,    # red
}


def _enter_pressed() -> bool:
    """True if Enter is held right now (distinguishes Enter from focus-loss in end_edit)."""
    try:
        iface = carb.input.acquire_input_interface()
        kb = omni.appwindow.get_default_app_window().get_keyboard()
        keys = (carb.input.KeyboardInput.ENTER, carb.input.KeyboardInput.NUMPAD_ENTER)
        return any(iface.get_keyboard_value(kb, k) for k in keys)
    except Exception:
        return False


class VlaSimWindow:
    def __init__(self, models: list[ModelSpec], procs: ProcessManager,
                 on_instruction: Callable[[str], None], on_observe: Callable[[], None], status,
                 on_stop: Callable[[], None] = lambda: None, on_test_motion: Callable[[], None] = lambda: None,
                 on_reset: Callable[[], None] = lambda: None, on_randomize: Callable[[], None] = lambda: None,
                 on_restore: Callable[[], None] = lambda: None, on_rescan: Callable[[], None] = lambda: None):
        self._models = models
        self._procs = procs
        self._on_instruction = on_instruction
        self._on_observe = on_observe
        self._on_stop = on_stop
        self._on_test_motion = on_test_motion
        self._on_reset = on_reset
        self._on_randomize, self._on_restore, self._on_rescan = on_randomize, on_restore, on_rescan
        self._rand_version = -1
        self._status_src = status  # object exposing .last_obs
        self._prompt_sub = None
        self.window = ui.Window("VLA Sim", width=440, height=580)
        self._build()
        self.refresh()

    # -- layout -----------------------------------------------------------------------------
    def _build(self) -> None:
        with self.window.frame:
            with ui.VStack(spacing=8, height=0):
                with ui.CollapsableFrame("Model", height=0):
                    with ui.VStack(spacing=6, height=0):
                        with ui.HStack(height=24, spacing=6):
                            ui.Label("Model", width=70)
                            names = [m.name for m in self._models] or ["(no models in configs/models.yaml)"]
                            self._combo = ui.ComboBox(0, *names)
                        self._combo.model.get_item_value_model().add_value_changed_fn(lambda _m: self.refresh())
                        with ui.HStack(height=16):
                            ui.Spacer(width=76)
                            self._model_info = ui.Label("", style={"color": 0xFF9A9A9A, "font_size": 12})
                        with ui.HStack(height=26, spacing=6):
                            self._load_btn = ui.Button(
                                "Load", clicked_fn=self._load,
                                tooltip="Load the selected model (a different loaded model is unloaded first)")
                            self._unload_btn = ui.Button("Unload", clicked_fn=self._procs.unload)
                        self._cache_state: dict[str, bool | None] = {}
                        with ui.HStack(height=20):
                            ui.Label("Status", width=70)
                            self._status = ui.Label("")
                        with ui.HStack(height=20):
                            ui.Label("GPU", width=70)
                            self._gpu = ui.Label("–")
                        self._detail = ui.Label("", word_wrap=True, height=0, style={"color": 0xFF9A9A9A, "font_size": 13})

                with ui.CollapsableFrame("Instruction", height=0):
                    with ui.VStack(spacing=6, height=0):
                        with ui.HStack(height=26, spacing=6):
                            self._prompt = ui.StringField()
                            self._send_btn = ui.Button("Send", width=60, clicked_fn=self._send)
                        self._prompt_sub = self._prompt.model.subscribe_end_edit_fn(self._on_end_edit)
                        with ui.HStack(height=20):
                            ui.Label("Active", width=70)
                            self._active = ui.Label("–", word_wrap=True)

                with ui.CollapsableFrame("Run", height=0):
                    with ui.VStack(spacing=6, height=0):
                        with ui.HStack(height=26, spacing=6):
                            self._stop_btn = ui.Button("Stop", clicked_fn=lambda: self._on_stop(),
                                                       tooltip="Stop the running episode; the arm holds its last target")
                            self._reset_btn = ui.Button(
                                "Reset", clicked_fn=lambda: self._on_reset(),
                                tooltip="Stop the episode and restore the saved scene: arm at the DROID home pose, "
                                        "gripper open, cube and bowl at their saved poses (timeline Stop + Play)")
                        with ui.HStack(height=20, spacing=6):
                            self._record_cb = ui.CheckBox(width=20)
                            ui.Label("Record episodes (video + model inputs + step log → outputs/recordings/)",
                                     style={"font_size": 13})
                        self._record_cb.model.add_value_changed_fn(
                            lambda m: setattr(self._status_src, "record_enabled", m.as_bool))
                        self._rec_label = ui.Label("", word_wrap=True, height=0, style={"color": 0xFF9A9A9A, "font_size": 13})
                        with ui.HStack(height=20):
                            ui.Label("Scene", width=70)
                            self._scene = ui.Label("–")
                        with ui.HStack(height=20):
                            ui.Label("Episode", width=70)
                            self._episode = ui.Label("idle", word_wrap=True, height=0)
                        with ui.HStack(height=20):
                            ui.Label("Task", width=70)
                            self._task = ui.Label("–", word_wrap=True, height=0)
                        self._task_detail = ui.Label("", word_wrap=True, height=0, style={"color": 0xFF9A9A9A, "font_size": 13})

                with ui.CollapsableFrame("Randomization", height=0):
                    with ui.VStack(spacing=6, height=0):
                        with ui.HStack(height=26, spacing=6):
                            self._rand_btn = ui.Button(
                                "Randomize", clicked_fn=lambda: self._on_randomize(),
                                tooltip="Move each ticked object a little (x/y and heading) within the workspace. "
                                        "Poses go to the session layer (never saved to the scene file); "
                                        "Reset restarts from them.")
                            self._restore_btn = ui.Button("Restore defaults", clicked_fn=lambda: self._on_restore(),
                                                          tooltip="Put every object back at its default pose")
                            self._rescan_btn = ui.Button("Rescan", width=70, clicked_fn=lambda: self._on_rescan(),
                                                         tooltip="Detect the randomizable objects again")
                        self._rand_msg = ui.Label("", word_wrap=True, height=0, style={"color": 0xFF9A9A9A, "font_size": 13})
                        self._rand_rows = ui.Frame(height=0, build_fn=self._build_rand_rows)

                with ui.CollapsableFrame("Debug", height=0, collapsed=False):
                    with ui.VStack(spacing=6, height=0):
                        self._test_btn = ui.Button(
                            "Test motion", height=26, clicked_fn=lambda: self._on_test_motion(),
                            tooltip="Scripted DROID-format chunks through the full lockstep loop (no model). "
                                    "Moves joints 1/4/6/7 out and back at half and full speed and closes/opens the gripper. "
                                    "The sim must be playing.",
                        )
                        self._observe_btn = ui.Button(
                            "Observe once", height=26, clicked_fn=self._on_observe,
                            tooltip="Pause, render both cameras, publish images + joint states + request on ROS 2, resume. "
                                    "With a model loaded and an instruction sent, the bridge answers with one action chunk.",
                        )
                        self._obs_label = ui.Label("–", word_wrap=True, height=0, style={"color": 0xFF9A9A9A, "font_size": 13})
                        self._chunk_label = ui.Label("–", word_wrap=True, height=0, style={"color": 0xFF9A9A9A, "font_size": 13})
                        with ui.HStack(height=20):
                            ui.Label("Bridge", width=70)
                            self._bridge_label = ui.Label("–", word_wrap=True, style={"font_size": 13})

    # -- actions ----------------------------------------------------------------------------
    def _selected_model(self) -> ModelSpec | None:
        idx = self._combo.model.get_item_value_model().as_int
        return self._models[idx] if 0 <= idx < len(self._models) else None

    def _load(self) -> None:
        model = self._selected_model()
        if model:
            self._procs.switch(model)      # unloads a different loaded model first, then loads this one

    def _on_end_edit(self, _model) -> None:
        if _enter_pressed():
            self._send()

    def _send(self) -> None:
        text = self._prompt.model.as_string.strip()
        if not text or self._procs.state != ModelState.READY:
            return
        self._active.text = text
        self._on_instruction(text)

    # -- state → widgets --------------------------------------------------------------------
    def refresh(self) -> None:
        st = self._procs.state
        name = self._procs.model.name if self._procs.model else ""
        self._status.text = f"{st.value}  {name}".strip()
        self._status.set_style({"color": _STATE_COLOR[st]})
        self._detail.text = self._procs.detail
        g = self._procs.gpu_mib
        self._gpu.text = f"{g / 1024:.1f} GB (server)" if g else ("external server" if self._procs.external else "–")

        ready = st == ModelState.READY
        sel = self._selected_model()
        if sel is not None:
            if sel.name not in self._cache_state or (sel.name == name and ready and not self._cache_state[sel.name]):
                self._cache_state[sel.name] = sel.checkpoint_cached()
            c = self._cache_state[sel.name]
            self._model_info.text = sel.description + ("" if c is None else
                                                       " · downloaded" if c else " · downloads on first load")
        same = sel is not None and sel.name == name and st in (ModelState.LOADING, ModelState.READY)
        self._combo.enabled = st != ModelState.UNLOADING
        self._load_btn.enabled = sel is not None and st != ModelState.UNLOADING and not same
        self._unload_btn.enabled = st in (ModelState.LOADING, ModelState.READY, ModelState.FAILED)
        self._prompt.enabled = ready
        self._send_btn.enabled = ready
        running = bool(getattr(self._status_src, "episode_running", False))
        self._stop_btn.enabled = running
        self._test_btn.enabled = not running
        self._observe_btn.enabled = not running
        self._episode.text = getattr(self._status_src, "episode_status", "idle")
        self._rec_label.text = getattr(self._status_src, "last_recording", "")
        self._record_cb.enabled = not running
        hint = getattr(self._status_src, "scene", {}).get("instruction_hint")
        self._scene.text = getattr(self._status_src, "scene_name", "–") + (f"  (try: \"{hint}\")" if hint else "")
        resetting = bool(getattr(self._status_src, "resetting", False))
        r = getattr(self._status_src, "randomizer", None)
        version = getattr(self._status_src, "rand_version", 0)
        if version != self._rand_version:
            self._rand_version = version
            self._rand_rows.rebuild()
        busy = running or resetting or bool(getattr(self._status_src, "randomizing", False))
        has_objs = bool(r and r.objects)
        self._rand_btn.enabled = has_objs and not busy
        self._restore_btn.enabled = has_objs and not busy
        self._rescan_btn.enabled = not busy
        self._rand_msg.text = r.message if r else "–"
        for o, lbl in getattr(self, "_rand_offset_labels", []):
            dx, dy, dyaw = o.offset
            lbl.text = f"{100 * dx:+.1f}, {100 * dy:+.1f} cm · {dyaw:+.0f}°" if any(o.offset) else "default"
        self._reset_btn.enabled = not resetting
        self._test_btn.enabled = not running and not resetting
        self._observe_btn.enabled = not running and not resetting
        self._send_btn.enabled = ready and not resetting
        self.refresh_task()
        if not ready:
            self._active.text = "–"
        self._obs_label.text = getattr(self._status_src, "last_obs", "–")
        self._chunk_label.text = getattr(self._status_src, "last_chunk", "–")
        self._bridge_label.text = getattr(self._status_src, "bridge_status", "") or "–"

    def _build_rand_rows(self) -> None:
        r = getattr(self._status_src, "randomizer", None)
        self._rand_offset_labels = []
        with ui.VStack(spacing=2, height=0):
            if r is None or not r.objects:
                return
            with ui.HStack(height=18):
                ui.Label("", width=22)
                ui.Label("Object", width=110, style={"color": 0xFF9A9A9A, "font_size": 13})
                ui.Label("Default pose (x, y, z m · yaw)", style={"color": 0xFF9A9A9A, "font_size": 13})
                ui.Label("Offset", width=150, style={"color": 0xFF9A9A9A, "font_size": 13})
            for o in r.objects:
                with ui.HStack(height=20):
                    cb = ui.CheckBox(width=22)
                    cb.model.set_value(o.enabled)
                    cb.model.add_value_changed_fn(lambda m, obj=o: setattr(obj, "enabled", m.as_bool))
                    ui.Label(o.name, width=110, tooltip=o.path + (f"\non {o.support_name}" if o.support_name else ""))
                    p = o.default_pos
                    ui.Label(f"{p[0]:+.3f}, {p[1]:+.3f}, {p[2]:+.3f} · {o.default_yaw_deg:+.0f}°", style={"font_size": 13})
                    self._rand_offset_labels.append((o, ui.Label("–", width=150, style={"font_size": 13})))

    def refresh_task(self) -> None:
        if not self.window:
            return
        status, detail = getattr(self._status_src, "task_status", ("–", ""))
        self._task.text = status
        self._task.set_style({"color": 0xFF4CC24C if status.endswith("SUCCESS") else 0xFFCCCCCC})
        self._task_detail.text = detail

    def destroy(self) -> None:
        self._prompt_sub = None
        if self.window:
            self.window.destroy()
            self.window = None
