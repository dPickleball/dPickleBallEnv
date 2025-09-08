import numpy as np
import cv2
from collections import deque
from gym import Env, spaces
from gym.utils import seeding
from mlagents_envs.environment import UnityEnvironment
import torch
import torch.nn as nn
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from mlagents_envs.envs.unity_parallel_env import UnityParallelEnv

class SharedObsUnityGymWrapper(Env):
    def __init__(self, unity_env, frame_stack=64, img_size=(168, 84), grayscale=False):
        self.env = UnityParallelEnv(unity_env)

        # left agent 0, right agent 1
        self.agent = self.env.possible_agents[1]       # agent to be controlled
        self.agent_other = self.env.possible_agents[0] # agent at opposite
        self.agent_obs = self.env.possible_agents[0]   # obs is only available in agent 0, always 0
        self.frame_stack = frame_stack
        self.img_size = img_size
        self.grayscale = grayscale
        self.frames = deque(maxlen=frame_stack)
        self._np_random = None

        # Observation space
        base_obs = self.env.observation_spaces[self.agent_obs][0]
        c, h, w = base_obs.shape
        self._transpose = (c == 3)

        # Final obs shape after manual preprocessing
        if grayscale:
            obs_shape = (frame_stack, img_size[1], img_size[0])  # (stack, H, W)
        else:
            obs_shape = (frame_stack * c, img_size[1], img_size[0])  # (stack*C, H, W)

        self.observation_space = spaces.Box(
            low=0.0, high=1.0, shape=obs_shape, dtype=np.float32
        )
        # self.action_space = self.env.action_spaces[self.agent]
        
    def _preprocess(self, obs):
        obs = obs[:, 10:-10, 10:-10]
        # obs = obs[:, 10:-10, 30:-30]
        # Transpose from (C, H, W) → (H, W, C)
        if self._transpose:
            obs = obs.transpose(1, 2, 0)

        obs = (obs * 255).astype(np.uint8)
        # Masking
        threshold = 242
        agent_mask = np.where(obs[..., 0] > threshold, 1, 0).astype(np.uint8)
        
        border_mask = np.zeros_like(agent_mask)
        border_mask[4, 48:100] = 1
        border_mask[59, 48:100] = 1

        ball_mask_low = np.array([220,223,31], dtype=np.uint8)
        ball_mask_high = np.array([250,250,50], dtype=np.uint8)
        ball_mask = cv2.inRange(obs, ball_mask_low, ball_mask_high)
        ball_mask >>= 7
        
        combined_obs = np.stack((agent_mask, border_mask, ball_mask), axis=0)
        
        return combined_obs

    # def _preprocess(self, obs):
    #     obs = obs[:, 10:-10, 10:-10]
    #     # obs = obs[:, 10:-10, 30:-30]
    #     # Transpose from (C, H, W) → (H, W, C)
    #     if self._transpose:
    #         obs = obs.transpose(1, 2, 0)

    #     obs = (obs * 255).astype(np.uint8)
    #     # Masking
    #     threshold = 233
    #     agent_border_mask = np.where(obs[..., 0] > threshold, 170, 0).astype(np.uint8)
    #     agent_border_mask[4, 48:100] = 85
    #     agent_border_mask[59, 48:100] = 85

    #     ball_mask_low = np.array([220,223,31], dtype=np.uint8)
    #     ball_mask_high = np.array([250,250,50], dtype=np.uint8)
    #     ball_mask = cv2.inRange(obs, ball_mask_low, ball_mask_high)
        
    #     combined_obs = np.maximum(agent_border_mask, ball_mask)
    #     combined_obs = np.expand_dims(combined_obs, axis=0)
    #     combined_obs = combined_obs.astype(np.float32) / 255.0
        
    #     return combined_obs
        
        # # Resize and grayscale
        # obs = cv2.resize(obs, self.img_size, interpolation=cv2.INTER_AREA)

        # if self.grayscale:
        #     obs = cv2.cvtColor(obs, cv2.COLOR_RGB2GRAY)  # (H, W)
        #     obs = np.expand_dims(obs, axis=0)  # (1, H, W)
        # else:
        #     obs = obs.transpose(2, 0, 1)  # (C, H, W)
        
        # if obs.max() > 1.01: 
        #     obs = obs.astype(np.float32)/255.0
        # obs = obs.astype(np.float32) / 255.0  # Normalize to [0, 1]
        # print("after preprocess", obs.shape, obs.min(), obs.max())
        # return obs
    
    # def _remove_border(self, obs):
    #     obs[14, 58:109+1] = 0
    #     obs[69, 58:109+1] = 0
    #     return obs

    def reset(self, *, seed=None, options=None):
        if seed is not None:
            self._np_random, seed = seeding.np_random(seed)
            if hasattr(self.env, "seed"):
                self.env.seed(seed)

        obs_dict = self.env.reset()
        obs = self._preprocess(obs_dict[self.agent_obs]['observation'][0])

        for _ in range(self.frame_stack):
            self.frames.append(obs)

        return np.concatenate(list(self.frames), axis=0), {}  # (stack, H, W)

    def step(self, action):
        actions = {self.agent: action}
        obs_dict, rewards, terminations, infos = self.env.step(actions)

        obs = self._preprocess(obs_dict[self.agent_obs]['observation'][0])
        self.frames.append(obs)

        stacked_obs = np.concatenate(list(self.frames), axis=0)  # (stack, H, W)


        if (rewards[self.agent] + rewards[self.agent_other]) > 0:
            print ("Rewards: ", rewards[self.agent], rewards[self.agent_other])

        return stacked_obs, rewards[self.agent] - rewards[self.agent_other], terminations[self.agent], False, infos[self.agent]

    def render(self):
        return self.env.render()

    def close(self):
        self.env.close()


class CustomCNN(BaseFeaturesExtractor):
    def __init__(self, observation_space, features_dim=512):
        super().__init__(observation_space, features_dim)
        n_input_channels = observation_space.shape[0]  # typically 4 for stacked frames

        self.cnn = nn.Sequential(
            nn.Conv2d(n_input_channels, 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.Flatten()
        )

        # Compute the output size of CNN
        with torch.no_grad():
            sample_input = torch.zeros(1, *observation_space.shape)
            sample_output = self.cnn(sample_input)
            cnn_output_dim = sample_output.shape[1]

        # Final linear layer to get to desired features_dim
        self.linear = nn.Sequential(
            nn.Linear(cnn_output_dim, features_dim),
            nn.ReLU()
        )

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        return self.linear(self.cnn(observations))


class ImpalaBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.relu = nn.ReLU(inplace=True)
        self.residual = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)
        )

    def forward(self, x):
        out = self.relu(self.conv(x))
        out = self.pool(out)
        out = self.relu(self.residual(out) + out)
        return out


class ImpalaCNN(BaseFeaturesExtractor):
    def __init__(self, observation_space, features_dim=512):
        super().__init__(observation_space, features_dim)
        n_channels = observation_space.shape[0]
        # Three Impala blocks with increasing channels
        self.network = nn.Sequential(
            ImpalaBlock(n_channels, 16),
            ImpalaBlock(16, 32),
            ImpalaBlock(32, 32),
            nn.Flatten()
        )
        # Compute output dim
        with torch.no_grad():
            sample = torch.zeros(1, *observation_space.shape)
            out_dim = self.network(sample).shape[1]
        self.linear = nn.Sequential(
            nn.Linear(out_dim, features_dim),
            nn.ReLU()
        )

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        x = self.network(observations)
        return self.linear(x)