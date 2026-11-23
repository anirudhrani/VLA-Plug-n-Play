"""Episode recorder: smooth video of both cameras + what the model saw + a per-step log.

One folder per episode, outputs/recordings/<YYYYmmdd-HHMMSS>_<scene>_<model>/:
  video.mp4         exterior | wrist side by side, one frame per action (15 fps = real time), with a caption
  model_inputs/     seq_NNNN.png: the two 224x224 images exactly as sent to the model for each request
  steps.npz         per action: step, sim time, joint positions, targets, raw action (+ joint names)
  episode.json      scene, model, instruction, outcome, steps, latencies, chunk timing, paths

Frames go to a system `ffmpeg` process (H.264, yuv420p) from a background thread, so the episode loop
never waits on encoding.
"""
from __future__ import annotations

import json
import queue
import subprocess
import threading
import time
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

PANEL_H = 360          # each camera is scaled to this height in the video (16:9 -> 640x360)


def resize_with_pad(img: np.ndarray, height: int = 224, width: int = 224) -> np.ndarray:
    """Same as openpi_client.image_tools.resize_with_pad (PIL bilinear, zero padding, centered)."""
    im = Image.fromarray(img)
    cw, ch = im.size
    if (cw, ch) == (width, height):
        return img
    ratio = max(cw / width, ch / height)
    rh, rw = int(ch / ratio), int(cw / ratio)
    out = Image.new(im.mode, (width, height), 0)
    out.paste(im.resize((rw, rh), resample=Image.BILINEAR), (max(0, int((width - rw) / 2)), max(0, int((height - rh) / 2))))
    return np.asarray(out)


class Recorder:
    def __init__(self, root: Path, scene: str, model: str, instruction: str, fps: int = 15):
        stamp = time.strftime("%Y%m%d-%H%M%S")
        self.dir = Path(root) / f"{stamp}_{scene}_{model}"
        (self.dir / "model_inputs").mkdir(parents=True, exist_ok=True)
        self.meta = {"scene": scene, "model": model, "instruction": instruction, "fps": fps,
                     "started": time.strftime("%Y-%m-%d %H:%M:%S")}
        self.fps = fps
        self.n_frames = 0
        self.n_inputs = 0
        self._q: queue.Queue = queue.Queue(maxsize=64)
        self._proc: subprocess.Popen | None = None
        self._size: tuple[int, int] | None = None
        self._thread = threading.Thread(target=self._writer, daemon=True)
        self._thread.start()
        self.error: str | None = None

    # -- video ------------------------------------------------------------------------------------
    def add_frame(self, images: dict[str, np.ndarray], step: int, sim_t: float, note: str = "") -> None:
        """One video frame: cameras side by side (exo first), with a caption bar."""
        panels = []
        for name in ("exo", "wrist"):
            if name in images:
                im = Image.fromarray(images[name])
                w = round(im.width * PANEL_H / im.height)
                panels.append(im.resize((w, PANEL_H), Image.BILINEAR))
        width = sum(p.width for p in panels)
        frame = Image.new("RGB", (width + width % 2, PANEL_H + 28), (20, 20, 20))
        x = 0
        for p in panels:
            frame.paste(p, (x, 0))
            x += p.width
        caption = f"step {step:3d}   t = {sim_t:5.2f} s   \"{self.meta['instruction']}\"   {note}"
        ImageDraw.Draw(frame).text((8, PANEL_H + 7), caption, fill=(230, 230, 230))
        try:
            self._q.put_nowait(np.asarray(frame))
            self.n_frames += 1
        except queue.Full:
            self.error = "video writer fell behind; frames dropped"

    def _writer(self) -> None:
        while True:
            frame = self._q.get()
            if frame is None:
                break
            try:
                if self._proc is None:
                    h, w, _ = frame.shape
                    self._size = (w, h)
                    self._proc = subprocess.Popen(
                        ["ffmpeg", "-loglevel", "error", "-y", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{w}x{h}",
                         "-r", str(self.fps), "-i", "-", "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "20",
                         "-movflags", "+faststart", str(self.dir / "video.mp4")],
                        stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=open(self.dir / "ffmpeg.log", "w"),
                    )
                if frame.shape[1::-1] != self._size:   # a scene switch mid-episode cannot happen, but be safe
                    continue
                self._proc.stdin.write(frame.tobytes())
            except Exception as e:  # noqa: BLE001
                self.error = f"video: {e}"

    # -- model inputs -------------------------------------------------------------------------------
    def add_model_input(self, seq: int, images: dict[str, np.ndarray]) -> None:
        pair = [resize_with_pad(images[n]) for n in ("exo", "wrist") if n in images]
        Image.fromarray(np.concatenate(pair, axis=1)).save(self.dir / "model_inputs" / f"seq_{seq:04d}.png")
        self.n_inputs += 1

    # -- finish ------------------------------------------------------------------------------------
    def finish(self, outcome: str, status: str, log: list[dict], joint_names: list[str],
               latencies_ms: list[float], extra: dict | None = None) -> Path:
        """Flush the video and write steps.npz + episode.json. Blocking (call from a worker thread)."""
        self._q.put(None)
        self._thread.join(timeout=60)
        if self._proc is not None:
            self._proc.stdin.close()
            self._proc.wait(timeout=120)
            if self._proc.returncode != 0:
                self.error = f"ffmpeg exited with {self._proc.returncode} (see ffmpeg.log)"
        if log:
            np.savez_compressed(
                self.dir / "steps.npz",
                step=np.array([r["step"] for r in log]), sim_t=np.array([r.get("sim_t", np.nan) for r in log]),
                q=np.stack([r["q"] for r in log]), target=np.stack([r["target"] for r in log]),
                action=np.stack([r["action"] for r in log]), joint_names=np.array(joint_names),
            )
        lat = np.asarray(latencies_ms) if latencies_ms else np.array([])
        self.meta.update({
            "outcome": outcome, "status": status, "ended": time.strftime("%Y-%m-%d %H:%M:%S"),
            "steps": len(log), "sim_seconds": round(len(log) / self.fps, 2),
            "video_frames": self.n_frames, "model_requests": self.n_inputs,
            "latency_ms": {"mean": round(float(lat.mean()), 1), "max": round(float(lat.max()), 1)} if lat.size else None,
            "gripper_close_commands": int(sum(float(r["action"][-1]) > 0.5 for r in log)),
            "files": {"video": "video.mp4", "steps": "steps.npz" if log else None, "model_inputs": "model_inputs/"},
            "recorder_error": self.error,
        })
        if extra:
            self.meta.update(extra)
        (self.dir / "episode.json").write_text(json.dumps(self.meta, indent=2))
        return self.dir
