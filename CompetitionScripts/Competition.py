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

try: 
    while env.agents:

        #observation available from agent0 only
        observation = observation['PAgent1?team=0?agent_id=0']['observation'][0]

        actions = {'PAgent1?team=0?agent_id=0':teamX.policy(observation, reward_left),'PAgent2?team=0?agent_id=1':teamY.policy(observation, reward_left)}

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


    



