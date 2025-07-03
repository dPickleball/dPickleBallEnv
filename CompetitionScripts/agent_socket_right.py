import socket
import json
import io
from PIL import Image
import numpy as np
from collections import deque
import cv2

def connect_to_server(port):
    s = socket.socket()
    s.connect(('localhost', port))

    # Wait for handshake
    data = s.recv(1024)
    msg = json.loads(data.decode())
    if msg.get("msg") == "ready?":
        s.send(json.dumps({"status": "ready"}).encode())
    else:
        raise Exception("Unexpected handshake message.")

    return s

def receive_image_and_reward(sock):
    try:
        # Receive image length and image
        img_len_bytes = sock.recv(4)
        img_len = int.from_bytes(img_len_bytes, 'big')
        img_data = b""
        while len(img_data) < img_len:
            part = sock.recv(img_len - len(img_data))
            if not part:
                break
            img_data += part

        img = Image.open(io.BytesIO(img_data))
        obs = np.array(img)

        # Receive reward message length and data
        reward_len_bytes = sock.recv(4)
        reward_len = int.from_bytes(reward_len_bytes, 'big')
        reward_data = b""
        while len(reward_data) < reward_len:
            part = sock.recv(reward_len - len(reward_data))
            if not part:
                break
            reward_data += part

        reward_msg = json.loads(reward_data.decode())
        reward = reward_msg["reward"]

        return obs, reward
    except socket.timeout:
        print("Timeout while receiving data.")
        return None, None

def send_action(sock, action_array: np.ndarray):
    assert action_array.shape == (3,), "Action must be a 3-element vector"
    action_list = action_array.tolist()
    action_msg = json.dumps(action_list).encode()
    sock.send(len(action_msg).to_bytes(4, 'big'))
    sock.sendall(action_msg)


port = 6002
sock = connect_to_server(port)
frame_stack = 4
frames = deque(maxlen=frame_stack)


while True:
    observation, reward = receive_image_and_reward(sock)

    # img = np.transpose(observation, (1, 2, 0))  # now shape is (84, 168, 3)
    # Convert to uint8 and RGB to BGR for OpenCV
    img_uint8 = cv2.cvtColor((observation).astype(np.uint8), cv2.COLOR_RGB2BGR)

    
    # cv2.imshow('Camera', img_uint8)
    # if cv2.waitKey(1) & 0xFF == ord('q'):
    #     print("Quitting display")
    #     break

    obs = cv2.cvtColor(observation, cv2.COLOR_RGB2GRAY)  # (H, W)
    obs = np.expand_dims(obs, axis=0)  # (1, H, W)

    frames.append(obs)

    stacked_obs = np.concatenate(list(frames), axis=0)  # (stack, H, W)


    if obs is None:
        continue  # Skip if timeout or error

    send_action(sock, np.array([1, 0, 0], dtype=np.int32)) # moving right



