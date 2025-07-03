from mlagents_envs.environment import UnityEnvironment
from mlagents_envs.envs.unity_parallel_env import UnityParallelEnv
import matplotlib.pyplot as plt
import sys
import cv2
from mlagents_envs.envs.custom_side_channel import CustomDataChannel, StringSideChannel
from uuid import UUID
import math
import numpy as np
from teamX import TeamX
teamX = TeamX()

import socket
import json
import io
import subprocess
import time
from PIL import Image

def wait_for_connection(port, name):
    s = socket.socket()
    s.bind(('localhost', port))
    s.listen(1)
    print(f"Waiting for {name} on port {port}...")
    conn, _ = s.accept()
    print(f"{name} connected.")

    # Send handshake
    conn.send(json.dumps({"msg": "ready?"}).encode())
    try:
        reply = json.loads(conn.recv(1024).decode())
        if reply.get("status") != "ready":
            raise Exception(f"{name} failed handshake.")
    except socket.timeout:
        raise Exception(f"{name} handshake timed out.")
    print(f"{name} is ready.")
    return conn

def send_obs_rew_receive_act(sock, obs: np.ndarray, rew: float):
    sock.settimeout(0.01)  # Set timeout for socket operations

    # print(f"obs dtype: {obs.dtype}, shape: {obs.shape}, min: {np.min(obs)}, max: {np.max(obs)}")
    if np.max(obs) == 0:
        print("[Warning] OBS is all black.")

    try:
        # Ensure obs is (H, W, C) for RGB
        if obs.ndim == 3 and obs.shape[0] in [1, 3, 4]:  # From (C, H, W)
            obs = obs.transpose(1, 2, 0)  # Convert to (H, W, C)
        elif obs.ndim != 3 or obs.shape[2] not in [1, 3, 4]:
            raise ValueError(f"Unsupported obs shape for image: {obs.shape}")

        # print("Sending image with shape:", obs.shape)

        # Convert to PIL image
        if np.issubdtype(obs.dtype, np.floating):
            obs = (obs * 255).clip(0, 255)

        pil_img = Image.fromarray(obs.astype(np.uint8))

        # Encode as PNG
        buffer = io.BytesIO()
        pil_img.save(buffer, format='PNG')
        img_bytes = buffer.getvalue()

        # Send image length and data
        sock.send(len(img_bytes).to_bytes(4, 'big'))
        sock.sendall(img_bytes)

        # Send reward as JSON
        reward_msg = json.dumps({"reward": rew}).encode()
        sock.send(len(reward_msg).to_bytes(4, 'big'))
        sock.sendall(reward_msg)

        # Receive action length
        act_len_bytes = sock.recv(4)
        if not act_len_bytes:
            raise ValueError("No action length received.")

        act_len = int.from_bytes(act_len_bytes, 'big')

        # Receive action data
        act_data = b""
        while len(act_data) < act_len:
            part = sock.recv(act_len - len(act_data))
            if not part:
                raise ValueError("Socket closed before full action received.")
            act_data += part

        # Parse action
        action = json.loads(act_data.decode())
        return action

    except socket.timeout:
        print("Timeout while waiting for agent.")
        return None
    except Exception as e:
        print(f"[send_obs_rew_receive_act] Error: {e}")
        return None


subprocess.Popen("python agent_socket_right.py", shell=True)

# left_conn = wait_for_connection(6001, "Left Agent")
right_conn = wait_for_connection(6002, "Right Agent")


string_channel = StringSideChannel()
channel = CustomDataChannel()

reward_cum = [0,0]
channel.send_data(serve=212, p1=reward_cum[0], p2=reward_cum[1])

print("Hello dPickleBall Trainer")

unity_env = UnityEnvironment("/home/gsk/Desktop/build_linux/dp.x86_64", side_channels=[string_channel, channel])
# unity_env = UnityEnvironment(None, side_channels=[string_channel, channel])
print("environment created")
env = UnityParallelEnv(unity_env)
print("petting zoo setup")
observation = env.reset()
print("ready to go!")

reward_left = reward_right = 0
step = 0

# print available agents
print("Agent Names", env.agents)
print("reward:", reward_cum)

try: 
    while env.agents:

        #observation available from agent0 only
        observation = observation['PAgent1?team=0?agent_id=0']['observation'][0]
        act_right = send_obs_rew_receive_act(right_conn, observation, reward_right)
        if act_right is None:
            act_right = np.array([0, 0, 0], dtype=np.int32)
        actions = {'PAgent1?team=0?agent_id=0':teamX.policy(observation, reward_left),'PAgent2?team=0?agent_id=1': act_right}

        observation, reward, done, info = env.step(actions)

        reward_cum[0] += reward['PAgent1?team=0?agent_id=0']
        reward_cum[1] += reward['PAgent2?team=0?agent_id=1']

        if reward['PAgent1?team=0?agent_id=0'] + reward['PAgent2?team=0?agent_id=1']>0:
            print("reward:", reward_cum)

        step += 1


except KeyboardInterrupt:
    print("Training interrupted")
finally:
    env.close()  # Important! Ensures Unity is notified and exits cleanly


    



