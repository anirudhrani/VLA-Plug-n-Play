"""M0 check: send one dummy DROID observation to the openpi server and print the action chunk.

Run with openpi's environment:
    cd third_party/openpi && uv run --no-sync python ../../scripts/test_policy_server.py
"""
import argparse
import time

import numpy as np
from openpi_client import websocket_client_policy

p = argparse.ArgumentParser()
p.add_argument("--host", default="localhost")
p.add_argument("--port", type=int, default=8000)
p.add_argument("--prompt", default="put the red cube in the bowl")
args = p.parse_args()

client = websocket_client_policy.WebsocketClientPolicy(args.host, args.port)
print("server metadata:", client.get_server_metadata())

home = np.array([0, -np.pi / 5, 0, -4 * np.pi / 5, 0, 3 * np.pi / 5, 0], dtype=np.float32)
obs = {
    "observation/exterior_image_1_left": np.zeros((224, 224, 3), np.uint8),
    "observation/wrist_image_left": np.zeros((224, 224, 3), np.uint8),
    "observation/joint_position": home,
    "observation/gripper_position": np.zeros(1, np.float32),
    "prompt": args.prompt,
}
for i in range(3):  # first call includes JIT compile
    t = time.perf_counter()
    actions = client.infer(obs)["actions"]
    print(f"call {i}: actions {actions.shape} {actions.dtype}, {1e3 * (time.perf_counter() - t):.0f} ms")
assert actions.ndim == 2 and actions.shape[1] == 8, actions.shape  # pi05_droid: (15, 8)
np.set_printoptions(precision=3, suppress=True)
print("first action:", actions[0])
print("OK")
