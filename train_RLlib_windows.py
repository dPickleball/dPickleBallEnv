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

print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU count: {torch.cuda.device_count()}")
    print(f"Current GPU: {torch.cuda.get_device_name(0)}")
    print(f"GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")

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
    def __init__(self, config, worker_id=0):
        super().__init__()
        agent_map = {
                "left": "PAgent1?team=0?agent_id=0",
                "right": "PAgent2?team=0?agent_id=1"
            }

        unity_env = UnityEnvironment(
                    file_name=os.path.join(os.path.dirname(os.path.abspath(__file__)), "build_windows", "dp.exe"),
                    no_graphics=False,
                    side_channels=[string_channel, channel],
                    worker_id=worker_id
                )
        env = UnityParallelEnv(unity_env)

        self.env = env
        self.agent_map = agent_map
        
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
        rgb = raw_obs[first_unity_agent]["observation"][0]
        gray = np.dot(rgb.transpose(1, 2, 0), [0.299, 0.587, 0.114])
        gray = gray[None, :, :]

        for agent in self._agents:
            self.obs_stack[agent].clear()
            for _ in range(2*lookbackframes):
                self.obs_stack[agent].append(gray.copy())

        obs = {
            agent: np.concatenate(list(self.obs_stack[agent])[0::2], axis=0).astype(np.float32)
            for agent in self._agents
        }
        return obs, {}

    def step(self, action_dict):
        unity_actions = {
            self.agent_map[agent]: act for agent, act in action_dict.items()
        }
        raw_obs, raw_rew, raw_done, raw_info = self.env.step(unity_actions)
        raw_trunc = {agent_id: False for agent_id in raw_obs}

        
        rewards = {agent: raw_rew.get(self.agent_map[agent], 0.0) for agent in self._agents}
        terminateds = {agent: raw_done.get(self.agent_map[agent], False) for agent in self._agents}
        truncateds = {agent: raw_trunc.get(self.agent_map[agent], False) for agent in self._agents}
        infos = {agent: raw_info.get(self.agent_map[agent], {}) for agent in self._agents}


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
            agent: np.concatenate(list(self.obs_stack[agent])[1::2], axis=0).astype(np.float32)
            for agent in self._agents
        }

        return obs, rewards, terminateds, truncateds, infos


    def close(self):
        try:
            self.env.close()
        except Exception as e:
            print(f"[Warning] Failed to close Unity env: {e}")

    def seed(self, seed=None):
        if hasattr(self.env, "seed"):
            return self.env.seed(seed)
        return [seed]


if __name__ == "__main__":
    import subprocess
    try:
        subprocess.run(["taskkill", "/f", "/im", "dp.exe"], capture_output=True)
    except:
        pass
    
    ray.init(
        local_mode=False,
        num_gpus=1,
        ignore_reinit_error=True
    )

    env_name = "dPickleBall"
    
    def create_env(config):
        worker_id = config.get("worker_id", 0)
        return SharedObsWrapper(config, worker_id)
    
    register_env(env_name, create_env)
    ModelCatalog.register_custom_model("CNNModelV2", CNNModelV2)

    obs_space = Box(shape=(lookbackframes, 84, 168), low=-np.inf, high=np.inf, dtype=np.float32)
    act_space = MultiDiscrete([3, 3, 3])

    config = (
        PPOConfig()
        .environment(env=env_name, clip_actions=True)
        .rollouts(num_rollout_workers=1, rollout_fragment_length=512)
        .training(
            train_batch_size=2048,
            sgd_minibatch_size=128,
            num_sgd_iter=8,
            lr=5e-5,
            gamma=0.99,
            lambda_=0.95,
            use_gae=True,
            clip_param=0.2,
            grad_clip=0.5,
            entropy_coeff=0.01,
            vf_loss_coeff=1.0,
            model={"vf_share_layers": False},
            _enable_learner_api=False
        )
        .rl_module(_enable_rl_module_api=False)
        .debugging(log_level="ERROR")
        .framework(framework="torch")
        .resources(num_gpus=1)
        .multi_agent(
            policies={
                "left_policy": (None, obs_space, act_space, {}),
                "right_policy": (None, obs_space, act_space, {}),
            },
            policy_mapping_fn=lambda agent_id, *args, **kwargs: f"{agent_id}_policy",
        )

    )
    config.model = {
        "custom_model": "CNNModelV2",
        "custom_model_config": {},
        "max_seq_len": 100,  
    }


    try:
        tune.run(
            "PPO",
            name="PPO",
            stop={"timesteps_total": 5000000 if not os.environ.get("CI") else 50000},
            checkpoint_freq=100,
            local_dir=os.path.normpath(os.path.abspath("./ray_results/" + env_name)),
            config=config.to_dict(),
        )
    finally:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        ray.shutdown()
    
