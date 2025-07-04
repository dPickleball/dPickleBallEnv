from multiprocessing import shared_memory
import numpy as np
import time
import cv2
from collections import deque

obs_shape = (84, 168, 3)
rew_shape = (2,)
action_shape = (3,)

shm_obs = shared_memory.SharedMemory(name="shm_obs")
shm_rew = shared_memory.SharedMemory(name="shm_rew")
shm_step = shared_memory.SharedMemory(name="shm_step")
shm_action = shared_memory.SharedMemory(name="shm_action")

obs_array = np.ndarray(obs_shape, dtype=np.uint8, buffer=shm_obs.buf)
rew_array = np.ndarray(rew_shape, dtype=np.int32, buffer=shm_rew.buf)
step_array = np.ndarray((), dtype=np.int32, buffer=shm_step.buf)
action_array = np.ndarray(action_shape, dtype=np.int32, buffer=shm_action.buf)

obs_array.flags.writeable = False
rew_array.flags.writeable = False
step_array.flags.writeable = False


# load your agent check point
#

frame_stack = 4
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
            obs = cv2.cvtColor(obs, cv2.COLOR_RGB2GRAY)  # (H, W)
            obs = np.expand_dims(obs, axis=0)  # (1, H, W)
            frames.append(obs)

            # print(frames)

            # apply your policy network here
            # 

            # Send your action back to competition
            action_array[:] = np.array([1, 0, 1], dtype=np.int32)

        time.sleep(0.01)
        

except KeyboardInterrupt:
    print("[Agent] Interrupted.")
finally:
    print("[Agent] Cleaning up shared memory...")
    for shm in [shm_obs, shm_rew, shm_step, shm_action]:
        try:
            shm.close()
        except Exception as e:
            print(f"[Agent] Error closing {shm.name}: {e}")
