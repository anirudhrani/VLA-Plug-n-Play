"""The extension's ROS 2 node (Isaac's bundled Humble rclpy, Python 3.12).

Publishes observations on demand. All messages of one observation carry the same stamp, so the
bridge node can pair them exactly. Only standard message types are used: custom messages would
have to be built against Isaac's internal Python 3.12 ROS, which we avoid.
"""
from __future__ import annotations

import array
import asyncio
import dataclasses
import time
from typing import Callable

import numpy as np
import omni.kit.app
import rclpy
from builtin_interfaces.msg import Time
from rclpy.executors import SingleThreadedExecutor
from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import Image, JointState
from std_msgs.msg import Header, String
from trajectory_msgs.msg import JointTrajectory

from .sim_io import Observation


RELIABLE = QoSProfile(depth=10, reliability=ReliabilityPolicy.RELIABLE)
LATCHED = QoSProfile(depth=1, reliability=ReliabilityPolicy.RELIABLE, durability=DurabilityPolicy.TRANSIENT_LOCAL)


@dataclasses.dataclass
class ActionChunk:
    seq: int
    stamp_s: float           # observation stamp the chunk was computed from
    dim_names: list[str]
    actions: np.ndarray      # (H, D) raw model actions
    received_wall: float     # time.monotonic() at arrival


def _stamp(t: float) -> Time:
    sec = int(t)
    return Time(sec=sec, nanosec=int(round((t - sec) * 1e9)))


class RosIO:
    def __init__(self, scene_cfg: dict, node_name: str = "vla_sim"):
        self._cams = scene_cfg["cameras"]
        topics = scene_cfg["topics"]
        self._ctx = rclpy.Context()
        rclpy.init(context=self._ctx)
        self.node = rclpy.create_node(node_name, context=self._ctx)
        self._img_pubs = {name: self.node.create_publisher(Image, c["topic"], 10) for name, c in self._cams.items()}
        self._joint_pub = self.node.create_publisher(JointState, topics["joint_states"], 10)
        self._grip_pub = self.node.create_publisher(JointState, topics["gripper"], 10)
        self._req_pub = self.node.create_publisher(Header, topics["request"], 10)
        self._instr_pub = self.node.create_publisher(String, topics["instruction"], LATCHED)
        self.node.create_subscription(JointTrajectory, topics["action_chunk"], self._on_chunk, RELIABLE)
        self.node.create_subscription(String, topics["bridge_status"], self._on_status, LATCHED)
        self.node.create_subscription(String, topics["instruction_ack"], self._on_ack, LATCHED)
        self.bridge_status: str = ""
        self.instruction_ack: str | None = None
        self.on_chunk: Callable[[ActionChunk], None] | None = None
        self.on_status: Callable[[str], None] | None = None
        self._executor = SingleThreadedExecutor(context=self._ctx)
        self._executor.add_node(self.node)
        self._spin_task = asyncio.ensure_future(self._spin())

    async def _spin(self) -> None:
        """Process incoming ROS messages once per app update (callbacks run on Kit's main thread)."""
        app = omni.kit.app.get_app()
        while self._ctx.ok():
            self._executor.spin_once(timeout_sec=0)
            await app.next_update_async()

    def _on_chunk(self, msg: JointTrajectory) -> None:
        try:
            chunk = ActionChunk(
                seq=int(msg.header.frame_id), stamp_s=msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9,
                dim_names=list(msg.joint_names), actions=np.array([p.positions for p in msg.points], dtype=np.float64),
                received_wall=time.monotonic(),
            )
        except (ValueError, TypeError):
            return
        if self.on_chunk:
            self.on_chunk(chunk)

    def _on_status(self, msg: String) -> None:
        self.bridge_status = msg.data
        if self.on_status:
            self.on_status(msg.data)

    def _on_ack(self, msg: String) -> None:
        self.instruction_ack = msg.data

    def publish_instruction(self, text: str) -> None:
        self._instr_pub.publish(String(data=text))

    def reset_bridge_status(self) -> None:
        self.bridge_status = ""

    def publish_observation(self, obs: Observation, seq: int) -> None:
        stamp = _stamp(obs.stamp_s)
        for name, img in obs.images.items():
            h, w, _ = img.shape
            msg = Image()
            msg.header = Header(stamp=stamp, frame_id=self._cams[name]["frame_id"])
            msg.height, msg.width, msg.encoding, msg.is_bigendian, msg.step = h, w, "rgb8", 0, w * 3
            msg.data = array.array("B", img.tobytes())
            self._img_pubs[name].publish(msg)

        js = JointState()
        js.header = Header(stamp=stamp, frame_id="")
        js.name = list(obs.joint_names)
        js.position = [float(v) for v in obs.joint_pos]
        self._joint_pub.publish(js)

        g = JointState()
        g.header = Header(stamp=stamp, frame_id="")
        g.name, g.position = ["gripper"], [float(obs.gripper)]
        self._grip_pub.publish(g)

        # request last: by the time the bridge sees it, the matching observation messages are already out
        self._req_pub.publish(Header(stamp=stamp, frame_id=str(seq)))

    def destroy(self) -> None:
        if self._spin_task and not self._spin_task.done():
            self._spin_task.cancel()
        try:
            self._executor.shutdown(timeout_sec=0)
        except Exception:
            pass
        try:
            self.node.destroy_node()
        finally:
            if self._ctx.ok():
                rclpy.shutdown(context=self._ctx)
