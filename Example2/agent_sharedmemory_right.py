from multiprocessing import shared_memory
import numpy as np
import time
import cv2
from collections import deque
from stable_baselines3 import PPO


obs_shape = (84, 168, 3)
rew_shape = (2,)
action_shape = (3,)

shm_obs = shared_memory.SharedMemory(name="shm_obs")
shm_rew = shared_memory.SharedMemory(name="shm_rew")
shm_step = shared_memory.SharedMemory(name="shm_step")
shm_action_right = shared_memory.SharedMemory(name="shm_action_right")

obs_array = np.ndarray(obs_shape, dtype=np.uint8, buffer=shm_obs.buf)
rew_array = np.ndarray(rew_shape, dtype=np.int32, buffer=shm_rew.buf)
step_array = np.ndarray((), dtype=np.int32, buffer=shm_step.buf)
action_array_right = np.ndarray(action_shape, dtype=np.int32, buffer=shm_action_right.buf)

obs_array.flags.writeable = False
rew_array.flags.writeable = False
step_array.flags.writeable = False

def _preprocess(obs):
    obs = cv2.flip(obs, 1)
    obs = obs[10:-10, 10:-10, :]
    # print(obs.shape)
    # print(obs.flags['C_CONTIGUOUS'])
    obs = np.ascontiguousarray(obs)
    # print(obs.flags['C_CONTIGUOUS'])
    img_size = (148, 64)
    obs = cv2.resize(obs, img_size, interpolation=cv2.INTER_AREA)
    grayscale = True
    if grayscale:
        obs = cv2.cvtColor(obs, cv2.COLOR_RGB2GRAY)  # (H, W)
        obs = np.expand_dims(obs, axis=0)  # (1, H, W)
    else:
        obs = obs.transpose(2, 0, 1)  # (C, H, W)

    if obs.max() >= 1.01:
        obs = obs.astype(np.float32) / 255.0
    return obs

# load your agent check point
#
model = PPO.load("D:\demo_dpickleball\Example2\ppo_sp_96000_steps.zip")
frame_stack = 12
frames = deque(maxlen=frame_stack)

last_step = -1
try:
    while True:
        current_step = int(step_array)
        if current_step != last_step:
            last_step = current_step
            obs = obs_array.copy()
            rew = rew_array.copy()

            # print(f"[Agent] Step {current_step}, rew={rew.tolist()}, obs_mean={obs.mean():.3f}")
            if np.sum(rew)>0:
                print("agent_rewards:", rew)

            # Visualization
            # img_uint8 = cv2.cvtColor(obs, cv2.COLOR_RGB2BGR)
            # cv2.imshow('Camera', img_uint8)
            # if cv2.waitKey(1) & 0xFF == ord('q'):
            #     print("Quitting display")
            #     break

            #preprocessing
            if len(frames) != frame_stack:
                # print("Adding frames")
                obs = _preprocess(obs)
                for _ in range(frame_stack):
                    frames.append(obs)
                assert len(frames) == frame_stack 
            else:
                obs = _preprocess(obs)
                frames.append(obs)

            # print(frames)

            # apply your policy network here
            # 

            # Send your action back to competition
            stacked_obs = np.concatenate(list(frames), axis=0)
            # print(frames)
            action, _states = model.predict(stacked_obs)
            
            # apply your policy network here
            # 
            
            # Send your action back to competition
            action_array_right[:] = action

        time.sleep(0.01)
        

except KeyboardInterrupt:
    print("[Agent] Interrupted.")
finally:
    print("[Agent] Cleaning up shared memory...")
    for shm in [shm_obs, shm_rew, shm_step, shm_action_right]:
        try:
            shm.close()
        except Exception as e:
            print(f"[Agent] Error closing {shm.name}: {e}")
