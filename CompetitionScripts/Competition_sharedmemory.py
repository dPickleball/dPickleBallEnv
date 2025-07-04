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
# from teamY import TeamY

from multiprocessing import shared_memory
import time
import subprocess
import atexit


print("[Sender] Receiver launched")
# === Shared Memory Setup ===
obs_shape = (84, 168, 3)
rew_shape = (2,)
action_shape = (3,)

shm_obs = shared_memory.SharedMemory(create=True, size=np.prod(obs_shape) * 1, name="shm_obs")
shm_rew = shared_memory.SharedMemory(create=True, size=2 * 4, name="shm_rew")
shm_step = shared_memory.SharedMemory(create=True, size=4, name="shm_step")
shm_action = shared_memory.SharedMemory(create=True, size=3 * 4, name="shm_action")

obs_array = np.ndarray(obs_shape, dtype=np.uint8, buffer=shm_obs.buf)
rew_array = np.ndarray(rew_shape, dtype=np.int32, buffer=shm_rew.buf)
step_array = np.ndarray((), dtype=np.int32, buffer=shm_step.buf)
action_array = np.ndarray(action_shape, dtype=np.int32, buffer=shm_action.buf)

# run right player script
subprocess.Popen("conda run -n dpickleball python agent_sharedmemory_right.py", shell=True)
time.sleep(1)

# run left player script
teamX = TeamX()


# Unity Env
string_channel = StringSideChannel()
channel = CustomDataChannel()

reward_cum = [0,0]
channel.send_data(serve=212, p1=reward_cum[0], p2=reward_cum[1])

print("Hello dPickleBall Trainer")

unity_env = UnityEnvironment("/home/gsk/Desktop/build_linux/dp.x86_64", side_channels=[string_channel, channel])

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
        img = np.transpose(observation, (1, 2, 0))  # now shape is (84, 168, 3)
        # # Convert to uint8 
        img_uint8 = (img * 255).astype(np.uint8)

        obs_array[:] = img_uint8
        rew_array[:] = np.array([reward_left,reward_right])
        step_array[...] = step

        # print(f"[Sender] Step {step}, reward={rew_array.tolist()}")

        # Read action
        act_right = action_array.copy()
        # print(f"[Sender] Action received: {act_right}")

        actions = {'PAgent1?team=0?agent_id=0':teamX.policy(observation, reward_left),'PAgent2?team=0?agent_id=1':act_right}

        observation, reward, done, info = env.step(actions)

        reward_cum[0] += int(reward['PAgent1?team=0?agent_id=0'])
        reward_cum[1] += int(reward['PAgent2?team=0?agent_id=1'])

        if reward['PAgent1?team=0?agent_id=0'] + reward['PAgent2?team=0?agent_id=1']>0:
            print("reward:", reward_cum)

        step += 1


except KeyboardInterrupt:
    print("Training interrupted")
finally:
    env.close()  # Important! Ensures Unity is notified and exits cleanly
    print("[Agent] Cleaning up shared memory...")
    for shm in [shm_obs, shm_rew, shm_step, shm_action]:
        try:
            shm.close()
            shm.unlink()
            print(f"[Cleanup] Unlinked {shm.name}")
        except FileNotFoundError:
            pass
        except Exception as e:
            print(f"[Cleanup] Error unlinking {shm.name}: {e}")


    



