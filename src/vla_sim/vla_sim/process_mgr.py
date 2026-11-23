"""Starts/stops the model's helper processes: the policy server and the vla_bridge ROS 2 node.

Processes run in their own session (process group) with a clean environment, so Isaac's
PYTHONPATH / LD_LIBRARY_PATH and the system ROS setup never leak into openpi's Python 3.11 venv.
"""
from __future__ import annotations

import asyncio
import enum
import os
import signal
import subprocess
import time
import urllib.request
from pathlib import Path
from typing import Callable

import carb

from .catalog import ModelSpec

LOAD_TIMEOUT_S = 2400.0  # first load may include a ~12 GB checkpoint download
BRIDGE_TIMEOUT_S = 180.0  # connect + warm-up inference (JIT compile) ~15 s normally
STOP_TIMEOUT_S = 10.0


class ModelState(enum.Enum):
    NO_MODEL = "No model loaded"
    LOADING = "Loading…"
    READY = "Ready"
    UNLOADING = "Unloading…"
    FAILED = "Failed"


def _clean_env(extra_keep: tuple[str, ...] = ()) -> dict[str, str]:
    keep = ("HOME", "USER", "LOGNAME", "LANG", "LC_ALL", "TERM", "SHELL", "XDG_RUNTIME_DIR") + extra_keep
    env = {k: os.environ[k] for k in keep if k in os.environ}
    home = env.get("HOME", str(Path.home()))
    env["PATH"] = f"{home}/.local/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin"
    return env


def _healthy(port: int) -> bool:
    try:
        with urllib.request.urlopen(f"http://localhost:{port}/healthz", timeout=0.5) as r:
            return r.status == 200
    except Exception:
        return False


def _gpu_mem_mib(session_id: int) -> int | None:
    """Sum GPU memory of all processes in the given session (uv spawns the real python as a child)."""
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-compute-apps=pid,used_memory", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=3,
        ).stdout
    except Exception:
        return None
    total, found = 0, False
    for line in out.strip().splitlines():
        try:
            pid, mem = (int(x) for x in line.split(","))
            if os.getsid(pid) == session_id:
                total, found = total + mem, True
        except (ValueError, ProcessLookupError, PermissionError):
            continue
    return total if found else None


def _kill_stale_bridges() -> list[int]:
    """Kill leftover vla_bridge nodes (e.g. from an Isaac session that crashed or was closed without Unload).

    They would still answer /vla/request with errors from their dead websocket and abort our episodes.
    """
    pids = []
    for d in Path("/proc").iterdir():
        if not d.name.isdigit() or int(d.name) == os.getpid():
            continue
        try:
            cmd = (d / "cmdline").read_bytes().split(b"\0")
            if b"vla_bridge.vla_node" in cmd and d.stat().st_uid == os.getuid():
                pids.append(int(d.name))
        except OSError:
            continue
    for sig in (signal.SIGINT, signal.SIGKILL):
        alive = []
        for pid in pids:
            try:
                os.kill(pid, sig)
                alive.append(pid)
            except ProcessLookupError:
                pass
        if not alive:
            break
        deadline = time.monotonic() + 3.0
        while time.monotonic() < deadline and any(Path(f"/proc/{p}").exists() for p in alive):
            time.sleep(0.1)
    if pids:
        carb.log_warn(f"[vla_sim] killed stale vla_bridge processes: {pids}")
    return pids


class ManagedProcess:
    def __init__(self, name: str, cmd: list[str], cwd: Path, log_path: Path, env_keep: tuple[str, ...] = ()):
        self.name, self.cmd, self.cwd, self.log_path = name, cmd, cwd, log_path
        self.env_keep = env_keep
        self.proc: subprocess.Popen | None = None

    def start(self) -> None:
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        log = open(self.log_path, "w")
        carb.log_info(f"[vla_sim] starting {self.name}: {' '.join(self.cmd)} (log: {self.log_path})")
        self.proc = subprocess.Popen(
            self.cmd, cwd=self.cwd, env=_clean_env(self.env_keep), stdin=subprocess.DEVNULL,
            stdout=log, stderr=subprocess.STDOUT, start_new_session=True,
        )
        log.close()

    def alive(self) -> bool:
        return self.proc is not None and self.proc.poll() is None

    def _signal(self, sig: int) -> None:
        try:
            os.killpg(self.proc.pid, sig)
        except ProcessLookupError:
            pass

    async def stop(self) -> None:
        if not self.alive():
            return
        self._signal(signal.SIGINT)
        deadline = time.monotonic() + STOP_TIMEOUT_S
        while self.alive() and time.monotonic() < deadline:
            await asyncio.sleep(0.2)
        if self.alive():
            self._signal(signal.SIGKILL)
            self.proc.wait(timeout=5)

    def stop_blocking(self) -> None:
        if not self.alive():
            return
        self._signal(signal.SIGINT)
        try:
            self.proc.wait(timeout=STOP_TIMEOUT_S)
        except subprocess.TimeoutExpired:
            self._signal(signal.SIGKILL)
            self.proc.wait(timeout=5)

    def last_log_line(self) -> str:
        try:
            with open(self.log_path, "rb") as f:
                f.seek(0, os.SEEK_END)
                f.seek(max(0, f.tell() - 4096))
                tail = f.read().decode(errors="replace").replace("\r", "\n")
            lines = [ln.strip() for ln in tail.splitlines() if ln.strip()]
            return lines[-1] if lines else ""
        except OSError:
            return ""


class ProcessManager:
    """Owns the lifecycle of the loaded model's processes and reports state changes."""

    def __init__(self, repo_root: Path, on_change: Callable[[], None],
                 bridge_status: Callable[[], str] = lambda: "", reset_bridge_status: Callable[[], None] = lambda: None):
        self.repo_root = repo_root
        self._on_change = on_change
        self._bridge_status = bridge_status
        self._reset_bridge_status = reset_bridge_status
        self._bridge: ManagedProcess | None = None
        self.state = ModelState.NO_MODEL
        self.model: ModelSpec | None = None
        self.detail = ""          # one-line status detail (error, last log line, …)
        self.gpu_mib: int | None = None
        self.external = False     # server was already running and is not owned by us
        self._server: ManagedProcess | None = None
        self._task: asyncio.Task | None = None

    # -- public API -------------------------------------------------------------------------
    def load(self, model: ModelSpec) -> None:
        if self.state in (ModelState.LOADING, ModelState.READY, ModelState.UNLOADING):
            return
        self._task = asyncio.ensure_future(self._load(model))

    def switch(self, model: ModelSpec) -> None:
        """One-click load: unload whatever is loaded (or loading / failed), then load `model`."""
        if self.state == ModelState.UNLOADING:
            return
        if self.model and self.model.name == model.name and self.state in (ModelState.LOADING, ModelState.READY):
            return
        if self.state == ModelState.NO_MODEL:
            self.load(model)
            return
        if self._task and not self._task.done():
            self._task.cancel()
        self._task = asyncio.ensure_future(self._switch(model))

    async def _switch(self, model: ModelSpec) -> None:
        await self._unload()
        await self._load(model)

    def unload(self) -> None:
        if self.state in (ModelState.NO_MODEL, ModelState.UNLOADING):
            return
        if self._task and not self._task.done():
            self._task.cancel()
        self._task = asyncio.ensure_future(self._unload())

    def shutdown(self) -> None:
        if self._task and not self._task.done():
            self._task.cancel()
        if self._bridge:
            self._bridge.stop_blocking()
            self._bridge = None
        if self._server:
            self._server.stop_blocking()
            self._server = None

    @property
    def log_path(self) -> Path | None:
        return self._server.log_path if self._server else None

    # -- internals --------------------------------------------------------------------------
    def _set(self, state: ModelState, detail: str = "") -> None:
        self.state, self.detail = state, detail
        self._on_change()

    async def _load(self, model: ModelSpec) -> None:
        self.model, self.external, self.gpu_mib = model, False, None
        loop = asyncio.get_event_loop()
        logs = self.repo_root / "outputs" / "logs"
        t0 = time.monotonic()

        # 1. policy server (reuse one that is already running on the port)
        if await loop.run_in_executor(None, _healthy, model.port):
            self.external = True
        else:
            self._server = ManagedProcess(f"{model.name} server", model.server_command(self.repo_root),
                                          self.repo_root, logs / f"{model.name}_server.log")
            try:
                self._server.start()
            except OSError as e:
                self._set(ModelState.FAILED, f"could not start server: {e}")
                return
            self._set(ModelState.LOADING, "starting policy server…")
            while not await loop.run_in_executor(None, _healthy, model.port):
                await asyncio.sleep(1.0)
                if not self._server.alive():
                    self._set(ModelState.FAILED, f"server exited (code {self._server.proc.returncode}); see {self._server.log_path}")
                    return
                if time.monotonic() - t0 > LOAD_TIMEOUT_S:
                    await self._server.stop()
                    self._set(ModelState.FAILED, f"server not ready after {LOAD_TIMEOUT_S:.0f} s; see {self._server.log_path}")
                    return
                self.detail = f"server {time.monotonic() - t0:.0f} s · {self._server.last_log_line()[:90]}"
                self._on_change()

        # 2. ROS bridge node: connects to the server and runs one warm-up inference
        await loop.run_in_executor(None, _kill_stale_bridges)
        self._reset_bridge_status()
        self._bridge = ManagedProcess(
            f"{model.name} bridge", [str(self.repo_root / "scripts" / "run_bridge.sh"), "--model", model.name],
            self.repo_root, logs / f"{model.name}_bridge.log", env_keep=("ROS_DOMAIN_ID", "RMW_IMPLEMENTATION", "DISPLAY"),
        )
        try:
            self._bridge.start()
        except OSError as e:
            self._set(ModelState.FAILED, f"could not start bridge: {e}")
            return
        self._set(ModelState.LOADING, "starting ROS bridge…")
        t1 = time.monotonic()
        while True:
            await asyncio.sleep(0.5)
            status = self._bridge_status()
            if status.startswith("ready"):
                break
            if status.startswith("error"):
                self._set(ModelState.FAILED, f"bridge: {status}; see {self._bridge.log_path}")
                return
            if not self._bridge.alive():
                self._set(ModelState.FAILED, f"bridge exited (code {self._bridge.proc.returncode}); see {self._bridge.log_path}")
                return
            if time.monotonic() - t1 > BRIDGE_TIMEOUT_S:
                self._set(ModelState.FAILED, f"bridge not ready after {BRIDGE_TIMEOUT_S:.0f} s; see {self._bridge.log_path}")
                return
            self.detail = f"bridge {time.monotonic() - t1:.0f} s · {status or 'starting'}"
            self._on_change()

        where = f"external server on :{model.port}" if self.external else f"server on :{model.port}"
        self._set(ModelState.READY, f"{where} + bridge (loaded in {time.monotonic() - t0:.0f} s)")
        # 3. keep watching: GPU memory + unexpected exits
        while self.state == ModelState.READY:
            for proc in (self._server, self._bridge):
                if proc is not None and not proc.alive():
                    self._set(ModelState.FAILED, f"{proc.name} exited unexpectedly; see {proc.log_path}")
                    return
            if self._server:
                self.gpu_mib = await loop.run_in_executor(None, _gpu_mem_mib, self._server.proc.pid)
            self._on_change()
            await asyncio.sleep(2.0)

    async def _unload(self) -> None:
        self._set(ModelState.UNLOADING)
        if self._bridge:
            await self._bridge.stop()
            self._bridge = None
        self._reset_bridge_status()
        if self._server:
            await self._server.stop()
            self._server = None
        self.model, self.gpu_mib, self.external = None, None, False
        self._set(ModelState.NO_MODEL)
