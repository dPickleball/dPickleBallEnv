from mlagents_envs.environment import UnityEnvironment
from mlagents_envs.envs.unity_parallel_env import UnityParallelEnv
import matplotlib.pyplot as plt
import sys
import cv2
from mlagents_envs.envs.custom_side_channel import CustomDataChannel, StringSideChannel
from uuid import UUID
import math
import numpy as np
import os

import ray
from ray import tune
from ray.rllib.algorithms.ppo import PPO
from ray.rllib.algorithms.dqn import DQNConfig
from ray.rllib.algorithms.ppo import PPOConfig
from ray.rllib.env.wrappers.pettingzoo_env import ParallelPettingZooEnv
from ray.rllib.models import ModelCatalog
from ray.rllib.models.torch.torch_modelv2 import TorchModelV2
from ray.tune.registry import register_env
from torch import nn
from pettingzoo import ParallelEnv
from ray import get_runtime_context
from ray.rllib.env.multi_agent_env import MultiAgentEnv
from gymnasium.spaces import Box, MultiDiscrete
from collections import deque



string_channel = StringSideChannel()
channel = CustomDataChannel()

reward_cum = [0,0]
channel.send_data(serve=212, p1=reward_cum[0], p2=reward_cum[1])

print("Hello dPickleBall Trainer")

reward_left = reward_right = 0
step = 0
lookbackframes=64

import torch
class CNNModelV2(TorchModelV2, nn.Module):
    def __init__(self, obs_space, act_space, num_outputs, *args, **kwargs):
        TorchModelV2.__init__(self, obs_space, act_space, num_outputs, *args, **kwargs)
        nn.Module.__init__(self)
        self.cnn = nn.Sequential(
            nn.Conv2d(lookbackframes, 32, [8, 8], stride=(4, 4)),
            nn.ReLU(),
            nn.Conv2d(32, 64, [4, 4], stride=(2, 2)),
            nn.ReLU(),
            nn.Conv2d(64, 64, [3, 3], stride=(1, 1)),
            nn.ReLU(),
        )

        # Dynamically determine output size
        with torch.no_grad():
            dummy_input = torch.zeros(1, lookbackframes, 84, 168)  # CHW
            n_flatten = self.cnn(dummy_input).view(1, -1).shape[1]

        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(n_flatten, 512),
            nn.ReLU(),
        )

        self.policy_fn = nn.Linear(512, num_outputs)
        self.value_fn = nn.Linear(512, 1)

    def forward(self, input_dict, state, seq_lens):
        x = input_dict["obs"]#.permute(0, 3, 1, 2)  # NHWC -> NCHW
        x = self.cnn(x)
        x = self.fc(x)
        self._value_out = self.value_fn(x)
        return self.policy_fn(x), state

    def value_function(self):
        return self._value_out.flatten()

class SharedObsWrapper(MultiAgentEnv):
    def __init__(self, config):
        super().__init__()
        agent_map = {
                "left": env.agents[0],
                "right": env.agents[1]
            }

        unity_env = UnityEnvironment(
                    file_name="/home/gsk/Desktop/build_linux/dp.x86_64",
                    no_graphics=False,  # <- required for GUI
                    side_channels=[string_channel, channel]
                )
        env = UnityParallelEnv(unity_env)

        self.env = env  # UnityParallelEnv instance
        self.agent_map = agent_map  # e.g., {"left": "PAgent1...", "right": "PAgent2..."}
        
        self._agents = list(agent_map.keys())
        self.agents = list(agent_map.keys())
        self._agent_ids = set(self._agents)

        self.obs_stack = {
            agent: deque(maxlen=2*lookbackframes)
            for agent in self._agents
        }


    def reset(self, *, seed=None, options=None):
        raw_obs = self.env.reset()

        first_unity_agent = list(raw_obs.keys())[0]
        rgb = raw_obs[first_unity_agent]["observation"][0]  # Shape: [3, H, W]
        gray = np.dot(rgb.transpose(1, 2, 0), [0.299, 0.587, 0.114])  # [H, W]
        gray = gray[None, :, :]  # Shape: [1, H, W]

        for agent in self._agents:
            self.obs_stack[agent].clear()
            for _ in range(2*lookbackframes):
                self.obs_stack[agent].append(gray.copy())

        obs = {
            agent: np.concatenate(list(self.obs_stack[agent])[0::2], axis=0)  # Shape: [4, H, W]
            for agent in self._agents
        }
        # print(obs["left"].shape, obs["right"].shape)
        return obs, {}

    def step(self, action_dict):
        unity_actions = {
            self.agent_map[agent]: act for agent, act in action_dict.items()
        }
        raw_obs, raw_rew, raw_done, raw_info = self.env.step(unity_actions)
        raw_trunc = {agent_id: False for agent_id in raw_obs}  # fallback if needed

        
        rewards = {agent: raw_rew.get(self.agent_map[agent], 0.0) for agent in self._agents}
        terminateds = {agent: raw_done.get(self.agent_map[agent], False) for agent in self._agents}
        truncateds = {agent: raw_trunc.get(self.agent_map[agent], False) for agent in self._agents}
        infos = {agent: raw_info.get(self.agent_map[agent], {}) for agent in self._agents}

        # Apply penalty: if opponent scores, subtract 1 from self
        left_raw = raw_rew.get(self.agent_map["left"], 0.0)
        right_raw = raw_rew.get(self.agent_map["right"], 0.0)
        rewards = {
            "left": left_raw - (1.0 if right_raw >= 1.0 else 0.0),
            "right": right_raw - (1.0 if left_raw >= 1.0 else 0.0),
        }

        
        if any(r >= 1.0 for r in rewards.values()):
            for agent in self._agents:
                terminateds[agent] = True
            terminateds["__all__"] = True
            truncateds["__all__"] = False
        else:
            terminateds["__all__"] = False
            truncateds["__all__"] = False

        first = list(raw_obs.keys())[0]
        rgb = raw_obs[first]["observation"][0]
        gray = np.dot(rgb.transpose(1, 2, 0), [0.299, 0.587, 0.114])
        gray = gray[None, :, :]

        for agent in self._agents:
            self.obs_stack[agent].append(gray.copy())

        obs = {
            agent: np.concatenate(list(self.obs_stack[agent])[1::2], axis=0)
            for agent in self._agents
        }
        # print(obs["left"].shape, obs["right"].shape)

        return obs, rewards, terminateds, truncateds, infos


    def close(self):
        try:
            self.env.close()
        except Exception as e:
            print(f"[Warning] Failed to close Unity env: {e}")

    def seed(self, seed=None):
        # Optional: you can propagate the seed to the underlying Unity env if needed
        if hasattr(self.env, "seed"):
            return self.env.seed(seed)
        # Just return the seed for compatibility
        return [seed]


if __name__ == "__main__":
    ray.init(local_mode=True)

    env_name = "dPickleBall"
    register_env(env_name, lambda cfg: SharedObsWrapper(cfg))
    ModelCatalog.register_custom_model("CNNModelV2", CNNModelV2)

    obs_space = Box(shape=(lookbackframes, 84, 168), low=-np.inf, high=np.inf, dtype=np.float32)
    act_space = MultiDiscrete([3, 3, 3])

    config = (
        PPOConfig()
        .environment(env=env_name, clip_actions=True)
        .rollouts(num_rollout_workers=0,rollout_fragment_length=512)
        .training(
            train_batch_size=512,       # more env steps per iteration
            sgd_minibatch_size=64,      # less noisy gradients
            num_sgd_iter=8,             # multiple passes over data
            lr=5e-5,                     # low learning rate (stable for pixel input)
            gamma=0.99,                 # discount factor
            lambda_=0.95,               # GAE lambda (0.9 is a bit low for Pong)
            use_gae=True,
            clip_param=0.2,             # PPO clipping range, 0.2 is standard
            grad_clip=0.5,              # optional gradient clipping
            entropy_coeff=0.01,         # encourage exploration, lower for stability
            vf_loss_coeff=1.0,          # higher weight on value loss
            model={"vf_share_layers": False}  # optionally share layers or not
        )
        .debugging(log_level="ERROR")
        .framework(framework="torch")
        .resources(num_gpus=int(os.environ.get("RLLIB_NUM_GPUS", "0")))
        .multi_agent(
            policies={
                "left_policy": (None, obs_space, act_space, {}),
                "right_policy": (None, obs_space, act_space, {}),
            },
            policy_mapping_fn=lambda agent_id, *args, **kwargs: f"{agent_id}_policy",
        )

    )
    # Set the model config separately
    config.model = {
        "custom_model": "CNNModelV2",
        "custom_model_config": {},  # Add any custom params if needed
        "max_seq_len": 100,  
    }


    tune.run(
        "PPO",
        name="PPO",
        stop={"timesteps_total": 5000000 if not os.environ.get("CI") else 50000},
        checkpoint_freq=100,
        local_dir="~/ray_results/" + env_name,
        config=config.to_dict(),
    )
    

