"""ChunkSource backed by the vla_bridge node: publish the observation on ROS 2, await the matching chunk."""
from __future__ import annotations

import asyncio
import time

import numpy as np

from .ros_io import ActionChunk, RosIO
from .sim_io import Observation


class BridgeSource:
    def __init__(self, ros: RosIO, next_seq, chunk_timeout_s: float = 10.0):
        self.ros = ros
        self._next_seq = next_seq          # callable -> int (shared with Observe once, so seqs never repeat)
        self.chunk_timeout_s = chunk_timeout_s
        self._waiting: dict[int, asyncio.Future] = {}
        self.latencies_ms: list[float] = []
        self.last_obs: Observation | None = None
        self.last_chunk: ActionChunk | None = None
        self.on_request = None             # optional callback(seq, obs) when an observation is sent (recorder)

    def on_chunk(self, chunk: ActionChunk) -> None:
        fut = self._waiting.pop(chunk.seq, None)
        if fut is not None and not fut.done():
            fut.set_result(chunk)

    async def next_chunk(self, obs: Observation) -> np.ndarray:
        seq = self._next_seq()
        if self.on_request is not None:
            try:
                self.on_request(seq, obs)
            except Exception:  # noqa: BLE001  (recording must never break an episode)
                self.on_request = None
        fut = asyncio.get_event_loop().create_future()
        self._waiting[seq] = fut
        self.last_obs = obs
        t0 = time.monotonic()
        self.ros.publish_observation(obs, seq)
        prefix = f"error: request seq {seq}"
        try:
            while not fut.done():
                if self.ros.bridge_status.startswith(prefix):
                    raise RuntimeError(self.ros.bridge_status[len("error: "):])
                if time.monotonic() - t0 > self.chunk_timeout_s:
                    raise RuntimeError(f"no action chunk for seq {seq} within {self.chunk_timeout_s:.0f} s")
                await asyncio.sleep(0.005)
        finally:
            self._waiting.pop(seq, None)
        chunk = fut.result()
        if abs(chunk.stamp_s - obs.stamp_s) > 1e-6:
            raise RuntimeError(f"chunk seq {seq} is for t={chunk.stamp_s:.3f}s, observation was t={obs.stamp_s:.3f}s")
        self.latencies_ms.append(1e3 * (chunk.received_wall - t0))
        self.last_chunk = chunk
        return chunk.actions

    def latency_summary(self) -> str:
        if not self.latencies_ms:
            return ""
        lat = np.asarray(self.latencies_ms)
        return f"latency last {lat[-1]:.0f} ms, mean {lat.mean():.0f} ms"
