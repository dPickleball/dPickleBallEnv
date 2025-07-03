import numpy as np
from collections import deque
import cv2
# Build a Python class for your solution, do preprocessing (image processing, frame stacking, etc) here.
# During competition, only the policy function is called at each time step, providing the observation and reward for that time step only.
# Your agent is expected to return actions to be executed.
class TeamY:
    # Define square directions
    square_directions = [
            np.array([1, 0, 0], dtype=np.int32),   # right
            np.array([0, 1, 0], dtype=np.int32),   # up
            np.array([2, 0, 0], dtype=np.int32),  # left
            np.array([0, 2, 0], dtype=np.int32)   # down
        ]
    steps_per_side = 30  # Number of env steps per side of the square
    step = 0

    def __init__(self, frame_stack=1):
        self.frames = deque(maxlen=frame_stack)
        
    
    # Your policy takes only visual representation as input, 
    # and reward is 1 when you score, -1 when your opponent scores
    # Your policy function returns actions
    def policy(self, observation, reward):
        # Implement your solution here

        # image processing
        obs = observation.transpose(1, 2, 0)
        obs = cv2.cvtColor(obs, cv2.COLOR_RGB2GRAY)  # (H, W)
        obs = np.expand_dims(obs, axis=0)  # (1, H, W)
        self.frames.append(observation)

        stacked_obs = np.concatenate(list(self.frames), axis=0)  # (stack, H, W)

        # square motion (you can replace with your own agent here)
        side = (self.step // self.steps_per_side) % 4
        action = self.square_directions[side]
        self.step+=1

        return action
        

