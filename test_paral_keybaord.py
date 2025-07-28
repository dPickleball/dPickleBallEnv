from mlagents_envs.environment import UnityEnvironment
from mlagents_envs.envs.unity_parallel_env import UnityParallelEnv
import matplotlib.pyplot as plt
import sys
import cv2
from mlagents_envs.envs.custom_side_channel import CustomDataChannel, StringSideChannel
from uuid import UUID
import math
import numpy as np
from pynput import keyboard

key_states = set()

def on_press(key):
    try:
        key_states.add(key.char)
    except AttributeError:
        pass

def on_release(key):
    try:
        key_states.discard(key.char)
    except AttributeError:
        pass

# Start listener
listener = keyboard.Listener(on_press=on_press, on_release=on_release)
listener.start()

string_channel = StringSideChannel()
channel = CustomDataChannel()

reward_cum = [0,0]
channel.send_data(serve=212, p1=reward_cum[0], p2=reward_cum[1])

print("Hello dPickleBall Trainer")

unity_env = UnityEnvironment(r"C:\Users\User\Desktop\build_windows_V4\dp.exe", side_channels=[string_channel, channel])
print("environment created")
env = UnityParallelEnv(unity_env)
print("petting zoo setup")
env.reset()
print("ready to go!")

# print available agents
print("Agent Names", env.agents)


try: 
    while env.agents:

        action_left = [0, 0, 0]
        action_right = [0, 0, 0]

        if 'w' in key_states:
            action_left[0] = 1
        if 's' in key_states:
            action_left[0] = 2
        if 'd' in key_states:
            action_left[1] = 1
        if 'a' in key_states:
            action_left[1] = 2
        if 'e' in key_states:
            action_left[2] = 1
        if 'q' in key_states:
            action_left[2] = 2

        if 'i' in key_states:
            action_right[0] = 1
        if 'k' in key_states:
            action_right[0] = 2
        if 'l' in key_states:
            action_right[1] = 1
        if 'j' in key_states:
            action_right[1] = 2
        if 'o' in key_states:
            action_right[2] = 1
        if 'u' in key_states:
            action_right[2] = 2

        actions = {'PAgent1?team=0?agent_id=0':action_left,'PAgent2?team=0?agent_id=1':action_right}

        observation, reward, done, info = env.step(actions)

        # print(observation, reward, done, info)

        reward_cum[0] += reward['PAgent1?team=0?agent_id=0']
        reward_cum[1] += reward['PAgent2?team=0?agent_id=1']

        # print("reward:", reward_cum, done)

        if done['PAgent1?team=0?agent_id=0'] or done['PAgent2?team=0?agent_id=1']:
            sys.exit()

        obs = observation['PAgent1?team=0?agent_id=0']['observation'][0]

        #print(obs.shape)
        
        # img = np.transpose(obs, (1, 2, 0))  # now shape is (84, 168, 3)
        # # Convert to uint8 and RGB to BGR for OpenCV
        # img_uint8 = cv2.cvtColor((img * 255).astype(np.uint8), cv2.COLOR_RGB2BGR)
        # cv2.imshow('Camera', img_uint8)

        # if cv2.waitKey(1) & 0xFF == ord('q'):
        #     print("Quitting display")
        #     break
        
        # print(f"Step {step}: action = {action}")

        step += 1



except KeyboardInterrupt:
    print("Training interrupted")
finally:
    env.close()  # Important! Ensures Unity is notified and exits cleanly
