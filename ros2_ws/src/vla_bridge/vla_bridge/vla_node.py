"""vla_bridge node: observation (ROS 2) -> policy server (websocket) -> action chunk (ROS 2).

For each /vla/request it waits until the images, /joint_states and /vla/gripper with exactly the request's
stamp have arrived, sends them (plus the latest /vla/instruction) to the policy server, and publishes the
returned chunk on /vla/action_chunk.

/vla/action_chunk is a trajectory_msgs/JointTrajectory used as a container:
  header.stamp    = the observation stamp (pairs the chunk with its request)
  header.frame_id = the request seq
  joint_names     = action dimensions (DROID: panda_joint1..7, gripper)
  points[k]       = step k, time_from_start = k / control_hz, positions = the model's RAW action values
                    (DROID: normalized joint velocities + gripper position; converted to targets in Isaac)

/vla/bridge_status (std_msgs/String, latched): "connecting", "warming_up", "ready ...", "error: ...".
/vla/instruction_ack (std_msgs/String, latched): echo of the last instruction received.
"""
from __future__ import annotations

import argparse
import os
import queue
import threading
import time
from pathlib import Path

import numpy as np
import rclpy
import rclpy.executors
import rclpy.task
import yaml
from builtin_interfaces.msg import Duration
from openpi_client import websocket_client_policy
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import Image, JointState
from std_msgs.msg import Header, String
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint

from . import adapters

RELIABLE = QoSProfile(depth=10, reliability=ReliabilityPolicy.RELIABLE)
LATCHED = QoSProfile(depth=1, reliability=ReliabilityPolicy.RELIABLE, durability=DurabilityPolicy.TRANSIENT_LOCAL)
PART_TIMEOUT_S = 5.0     # how long a request may wait for its observation messages
KEEP_STAMPS = 20         # observation stamps kept in the buffer


def _key(stamp) -> tuple[int, int]:
    return (stamp.sec, stamp.nanosec)


def _image_to_np(msg: Image) -> np.ndarray:
    if msg.encoding != "rgb8":
        raise ValueError(f"expected rgb8, got {msg.encoding}")
    data = np.frombuffer(bytes(msg.data), dtype=np.uint8)
    rows = data.reshape(msg.height, msg.step)
    return np.ascontiguousarray(rows[:, : msg.width * 3].reshape(msg.height, msg.width, 3))


class BridgeNode(Node):
    def __init__(self, args: argparse.Namespace, scene: dict):
        super().__init__("vla_bridge")
        self.args = args
        self.adapter = adapters.get(args.adapter)
        topics = scene["topics"]
        self.cam_topics = {name: c["topic"] for name, c in scene["cameras"].items()}  # exo, wrist
        self.parts = list(self.cam_topics) + ["joints", "gripper"]

        self._lock = threading.Lock()
        self._buf: dict[tuple, dict] = {}        # stamp -> {part: msg}
        self._pending: dict[tuple, tuple] = {}   # stamp -> (seq, wall time received)
        self._instruction: str | None = None
        self._jobs: queue.Queue = queue.Queue()
        self._client = None
        self._ready = False

        for name, topic in self.cam_topics.items():
            self.create_subscription(Image, topic, lambda m, n=name: self._on_part(n, m), RELIABLE)
        self.create_subscription(JointState, topics["joint_states"], lambda m: self._on_part("joints", m), RELIABLE)
        self.create_subscription(JointState, topics["gripper"], lambda m: self._on_part("gripper", m), RELIABLE)
        self.create_subscription(Header, topics["request"], self._on_request, RELIABLE)
        self.create_subscription(String, topics.get("instruction", "/vla/instruction"), self._on_instruction, LATCHED)
        self._chunk_pub = self.create_publisher(JointTrajectory, topics.get("action_chunk", "/vla/action_chunk"), RELIABLE)
        self._status_pub = self.create_publisher(String, topics.get("bridge_status", "/vla/bridge_status"), LATCHED)
        self._ack_pub = self.create_publisher(String, topics.get("instruction_ack", "/vla/instruction_ack"), LATCHED)
        self.create_timer(0.5, self._check_timeouts)
        # exit with the process that started us (Isaac): an orphaned bridge would keep answering requests
        self._parent_pid = os.getppid()
        self.exit_future = rclpy.task.Future()
        self.create_timer(1.0, self._check_parent)

        threading.Thread(target=self._connect_and_warm_up, daemon=True).start()
        threading.Thread(target=self._worker, daemon=True).start()

    # -- status ---------------------------------------------------------------------------------
    def _status(self, text: str) -> None:
        self._status_pub.publish(String(data=text))
        self.get_logger().info(f"status: {text}")

    # -- policy server --------------------------------------------------------------------------
    def _connect_and_warm_up(self) -> None:
        self._status("connecting")
        try:
            self._client = websocket_client_policy.WebsocketClientPolicy(self.args.host, self.args.port)
            self.get_logger().info(f"connected to {self.args.host}:{self.args.port}, metadata {self._client.get_server_metadata()}")
            self._status("warming_up")
            t0 = time.perf_counter()
            self.adapter.parse_response(self._client.infer(self.adapter.warmup_request()))
            self._ready = True
            self._status(f"ready · {self.args.model} · warm-up {time.perf_counter() - t0:.1f} s")
        except Exception as e:  # noqa: BLE001
            self._status(f"error: policy server: {e}")

    # -- observation assembly -------------------------------------------------------------------
    def _on_instruction(self, msg: String) -> None:
        self._instruction = msg.data.strip() or None
        self.get_logger().info(f"instruction: {self._instruction!r}")
        # echo it so the sim side knows every later request will use it
        self._ack_pub.publish(String(data=msg.data))

    def _on_part(self, part: str, msg) -> None:
        k = _key(msg.header.stamp)
        with self._lock:
            # prune by arrival order, not by stamp: sim time restarts from 0 on every Stop/Play
            entry = self._buf.pop(k, {})
            entry[part] = msg
            self._buf[k] = entry          # dicts keep insertion order -> most recent last
            while len(self._buf) > KEEP_STAMPS:
                del self._buf[next(iter(self._buf))]
        self._dispatch_if_complete(k)

    def _on_request(self, msg: Header) -> None:
        k = _key(msg.stamp)
        with self._lock:
            self._pending[k] = (msg.frame_id, time.monotonic())
        self._dispatch_if_complete(k)

    def _dispatch_if_complete(self, k: tuple) -> None:
        with self._lock:
            if k not in self._pending or any(p not in self._buf.get(k, {}) for p in self.parts):
                return
            seq, _ = self._pending.pop(k)
            obs = dict(self._buf[k])
        self._jobs.put((seq, k, obs))

    def _check_timeouts(self) -> None:
        now, late = time.monotonic(), []
        with self._lock:
            for k, (seq, t) in list(self._pending.items()):
                if now - t > PART_TIMEOUT_S:
                    del self._pending[k]
                    late.append((seq, [p for p in self.parts if p not in self._buf.get(k, {})]))
        for seq, missing in late:
            self.get_logger().error(f"request seq {seq}: observation incomplete after {PART_TIMEOUT_S:.0f} s, missing {missing}")
            self._status(f"error: request seq {seq}: observation incomplete, missing {', '.join(missing)}")

    def _check_parent(self) -> None:
        if os.getppid() != self._parent_pid:
            self.get_logger().warn(f"parent process {self._parent_pid} is gone, exiting")
            self.exit_future.set_result(True)

    # -- inference ------------------------------------------------------------------------------
    def _worker(self) -> None:
        while rclpy.ok():
            seq, k, obs = self._jobs.get()
            try:
                self._infer_and_publish(seq, k, obs)
            except Exception as e:  # noqa: BLE001
                self.get_logger().error(f"request seq {seq}: {e}")
                self._status(f"error: request seq {seq}: {e}")

    def _infer_and_publish(self, seq: str, k: tuple, obs: dict) -> None:
        if not self._ready:
            raise RuntimeError("model not ready yet")
        if not self._instruction:
            raise RuntimeError("no instruction yet (type one in the VLA Sim panel)")
        js, grip = obs["joints"], obs["gripper"]
        request = self.adapter.build_request(
            _image_to_np(obs["exo"]), _image_to_np(obs["wrist"]), list(js.name), list(js.position),
            float(grip.position[0]), self._instruction,
        )
        t0 = time.perf_counter()
        actions = self.adapter.parse_response(self._client.infer(request))
        infer_ms = 1e3 * (time.perf_counter() - t0)

        msg = JointTrajectory()
        msg.header.stamp = js.header.stamp
        msg.header.frame_id = seq
        msg.joint_names = list(self.adapter.ACTION_DIM_NAMES)
        dt = 1.0 / self.args.control_hz
        for i, row in enumerate(actions):
            t = i * dt
            msg.points.append(JointTrajectoryPoint(
                positions=[float(v) for v in row],
                time_from_start=Duration(sec=int(t), nanosec=int(round((t - int(t)) * 1e9))),
            ))
        self._chunk_pub.publish(msg)
        self._status(f"ready · {self.args.model} · seq {seq} · {actions.shape[0]}x{actions.shape[1]} · {infer_ms:.0f} ms")


def main() -> None:
    repo = Path(__file__).resolve().parents[4]  # <repo>/ros2_ws/src/vla_bridge/vla_bridge/vla_node.py
    ap = argparse.ArgumentParser(description="VLA bridge node")
    ap.add_argument("--model", default="pi05_droid", help="catalog name (configs/models.yaml)")
    ap.add_argument("--host", default="localhost")
    ap.add_argument("--port", type=int, default=None, help="policy server port (default: from catalog)")
    ap.add_argument("--adapter", default=None, help="adapter name (default: from catalog)")
    ap.add_argument("--control-hz", type=float, default=None)
    ap.add_argument("--scene", default=str(repo / "configs" / "scene_droid_scene1.yaml"))
    args, ros_args = ap.parse_known_args()
    model = yaml.safe_load((repo / "configs" / "models.yaml").read_text())["models"][args.model]
    args.port = args.port or int(model["port"])
    args.adapter = args.adapter or model["adapter"]
    args.control_hz = args.control_hz or float(model.get("control_hz", 15))
    scene = yaml.safe_load(Path(args.scene).read_text())

    rclpy.init(args=ros_args)
    node = BridgeNode(args, scene)
    try:
        rclpy.spin_until_future_complete(node, node.exit_future)
    except (KeyboardInterrupt, rclpy.executors.ExternalShutdownException):
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
