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
from teamY import TeamY
from threading import Thread, Lock

teamX = TeamX()
teamY = TeamY()

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
action_right = np.array([0, 0, 0])
action_left = np.array([0, 0, 0])


# Async to take care response time, if slow, use previous action
latest_action_x = np.array([0, 0, 0], dtype=np.int32)
action_x_lock = Lock()
policy_thread_y = None  # Keep track of thread object

latest_action_y = np.array([0, 0, 0], dtype=np.int32)
action_y_lock = Lock()
policy_thread_x = None  # Keep track of thread object

def update_policy_async_y(policy_fn, obs, rew):
    global latest_action_y, policy_thread_y

    def run_policy():
        global latest_action_y  # <- Required to write to global array
        try:
            result = policy_fn(obs, rew)
            with action_y_lock:
                latest_action_y[:] = result  # <-- use [:] to update in-place
        except Exception as e:
            print(f"[Warning] teamY policy failed later: {e}")

    if policy_thread_y is None or not policy_thread_y.is_alive():
        policy_thread_y = Thread(target=run_policy)
        policy_thread_y.start()

def update_policy_async_x(policy_fn, obs, rew):
    global latest_action_x, policy_thread_x

    def run_policy():
        global latest_action_x  
        try:
            result = policy_fn(obs, rew)
            with action_x_lock:
                latest_action_x[:] = result  
        except Exception as e:
            print(f"[Warning] teamY policy failed later: {e}")

    if policy_thread_x is None or not policy_thread_x.is_alive():
        policy_thread_x = Thread(target=run_policy)
        policy_thread_x.start()

try: 
    while env.agents:

        #observation available from agent0 only
        observation = observation['PAgent1?team=0?agent_id=0']['observation'][0]

        # Left : Use fallback action
        with action_x_lock:
            fallback_x = latest_action_x.copy()
        # Start async update (will only start if no active thread)
        update_policy_async_x(teamX.policy, observation, reward_left)
        # Use fallback for this step
        action_left = fallback_x

        # Right : Use fallback action
        with action_y_lock:
            fallback_y = latest_action_y.copy()
        # Start async update (will only start if no active thread)
        update_policy_async_y(teamY.policy, observation, reward_right)
        # Use fallback for this step
        action_right = fallback_y

        
        actions = {'PAgent1?team=0?agent_id=0':action_left,'PAgent2?team=0?agent_id=1':action_right}

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


    



