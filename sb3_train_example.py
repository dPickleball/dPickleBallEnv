from mlagents_envs.environment import UnityEnvironment
import matplotlib.pyplot as plt
import sys
import cv2
from mlagents_envs.envs.custom_side_channel import CustomDataChannel, StringSideChannel
import math
import numpy as np

import torch
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.torch_layers import NatureCNN
from stable_baselines3.common.policies import ActorCriticPolicy

from mylib import SharedObsUnityGymWrapper, CustomCNN


string_channel = StringSideChannel()
channel = CustomDataChannel()

#init points
reward_cum = [0,0]
# Decode game config code
# The last digit is the serve mode:
#     1 = left serve only
#     2 = right serve only
#     3 = random serve
# The remaining digits (before the last one) indicate the match point.
# Example: 213 → match point = 21, serve = 3 (random serve)
# Example: 12 → match point = 1, serve = 2 (right serve)
channel.send_data(serve=12, p1=reward_cum[0], p2=reward_cum[1])


print("Hello dPickleBall Trainer")

unity_env = UnityEnvironment("/home/gsk/Desktop/build_linux/dp.x86_64", side_channels=[string_channel, channel])

print("environment created")
env = SharedObsUnityGymWrapper(unity_env)
vec_env = DummyVecEnv([lambda: env])

obs = env.reset()

policy_kwargs = dict(
    features_extractor_class=CustomCNN,
    features_extractor_kwargs=dict(features_dim=512)
)
device = torch.device("cpu")
model = PPO(
    policy="CnnPolicy",
    env=env,
    policy_kwargs=policy_kwargs,
    verbose=3,
    device=device  # <<< Force CPU
)

model.learn(total_timesteps=100_000)


model.save(f"{env.unwrapped.metadata.get('name')}_{time.strftime('%Y%m%d-%H%M%S')}")

print("Model has been saved.")

print(f"Finished training on {str(env.unwrapped.metadata['name'])}.")

env.close()
