#!/usr/bin/env python3
"""Show the latest observation published by the VLA Sim panel (both cameras + joints + seq/stamp).

Why not rqt_image_view: it subscribes best-effort, and our images are large (~700 KB) one-shot messages,
which DDS drops for best-effort readers on this machine. This viewer subscribes RELIABLE, like the bridge.

Run with system ROS 2 (Humble) sourced:
    python3 scripts/view_observation.py
Then press "Observe once" in the panel. The window updates on every observation.
"""
import argparse
import tkinter as tk

import rclpy
from PIL import Image as PILImage
from PIL import ImageTk
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import Image, JointState
from std_msgs.msg import Header

TOPICS = {"exo": "/vla/exo/image_raw", "wrist": "/vla/wrist/image_raw"}


def stamp_str(s) -> str:
    return f"{s.sec}.{s.nanosec:09d}"


class Viewer(Node):
    def __init__(self, root: tk.Tk, once: bool = False):
        super().__init__("vla_observation_viewer")
        self.once = once
        qos = QoSProfile(depth=5, reliability=ReliabilityPolicy.RELIABLE)
        self.root = root
        self.labels, self.photos, self.stamps = {}, {}, {}
        frame = tk.Frame(root, bg="black")
        frame.pack()
        for col, name in enumerate(TOPICS):
            tk.Label(frame, text=f"{name}  ({TOPICS[name]})", fg="white", bg="black").grid(row=0, column=col)
            self.labels[name] = tk.Label(frame, bg="gray20", width=80, height=25)
            self.labels[name].grid(row=1, column=col, padx=4, pady=4)
        self.info = tk.Label(root, text="waiting for an observation — press 'Observe once' in the VLA Sim panel",
                             font=("monospace", 10), justify="left", anchor="w")
        self.info.pack(fill="x", padx=6, pady=6)
        self.joints, self.gripper, self.request = None, None, None
        for name, topic in TOPICS.items():
            self.create_subscription(Image, topic, lambda m, n=name: self.on_image(n, m), qos)
        self.create_subscription(JointState, "/joint_states", lambda m: (setattr(self, "joints", m), self.check_complete()), qos)
        self.create_subscription(JointState, "/vla/gripper", lambda m: (setattr(self, "gripper", m), self.check_complete()), qos)
        self.create_subscription(Header, "/vla/request", self.on_request, qos)

    def on_image(self, name: str, msg: Image) -> None:
        if msg.encoding != "rgb8":
            self.get_logger().warn(f"{name}: unexpected encoding {msg.encoding}")
            return
        img = PILImage.frombytes("RGB", (msg.width, msg.height), bytes(msg.data))
        self.photos[name] = ImageTk.PhotoImage(img)  # keep a reference or Tk drops it
        self.labels[name].configure(image=self.photos[name], width=msg.width, height=msg.height)
        self.stamps[name] = stamp_str(msg.header.stamp)
        self.check_complete()

    def on_request(self, msg: Header) -> None:
        self.request = msg
        self.shown = False
        self.check_complete()
        self.root.after(3000, lambda r=msg: self.update_info() if self.request is r and not self.shown else None)

    def check_complete(self) -> None:
        """Show the observation once every part with the request's stamp has arrived (large images can lag)."""
        if self.request is None or getattr(self, "shown", True):
            return
        want = stamp_str(self.request.stamp)
        parts = [self.stamps.get(n) for n in TOPICS]
        parts += [stamp_str(m.header.stamp) if m else None for m in (self.joints, self.gripper)]
        if all(p == want for p in parts):
            self.update_info()

    def update_info(self) -> None:
        self.shown = True
        r = self.request
        lines = [f"seq {r.frame_id}   request stamp {stamp_str(r.stamp)}"]
        lines.append("image stamps: " + ", ".join(f"{k} {v}" for k, v in self.stamps.items()))
        if self.joints:
            lines.append(f"joints @ {stamp_str(self.joints.header.stamp)}: " +
                         "  ".join(f"{n.replace('panda_', '')}={p:+.3f}" for n, p in zip(self.joints.name, self.joints.position)))
        if self.gripper:
            lines.append(f"gripper @ {stamp_str(self.gripper.header.stamp)}: {self.gripper.position[0]:.2f} (0 open, 1 closed)")
        missing = [n for n in TOPICS if self.stamps.get(n) != stamp_str(r.stamp)]
        if missing:
            lines.append(f"no image for this request yet from: {', '.join(missing)}")
        all_stamps = set(self.stamps.values()) | {stamp_str(r.stamp)}
        if self.joints:
            all_stamps.add(stamp_str(self.joints.header.stamp))
        lines.append("stamps match ✓" if len(all_stamps) == 1 else f"STAMP MISMATCH: {sorted(all_stamps)}")
        self.info.configure(text="\n".join(lines))
        if self.once:
            print("\n".join(lines) + "\nimages: " + ", ".join(f"{k} {p.width()}x{p.height()}" for k, p in self.photos.items()), flush=True)
            self.root.after(500, self.root.destroy)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--once", action="store_true", help="print the first observation and exit (for tests)")
    args = ap.parse_args()
    rclpy.init()
    root = tk.Tk()
    root.title("VLA observation")
    node = Viewer(root, once=args.once)

    def spin() -> None:
        rclpy.spin_once(node, timeout_sec=0)
        root.after(20, spin)

    root.after(20, spin)
    try:
        root.mainloop()
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
