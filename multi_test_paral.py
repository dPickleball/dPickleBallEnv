from mlagents_envs.environment import UnityEnvironment
from mlagents_envs.envs.unity_parallel_env import UnityParallelEnv
from mlagents_envs.envs.custom_side_channel import CustomDataChannel, StringSideChannel

import cv2
import numpy as np

NUM_ENVS = 5  
BASE_PORT = 5005  
STEPS_PER_SIDE = 30

# Define square directions
square_directions = [
    np.array([1, 0, 0], dtype=np.int32),  # right
    np.array([0, 1, 1], dtype=np.int32),  # up
    np.array([2, 0, 2], dtype=np.int32),  # left
    np.array([0, 2, 0], dtype=np.int32)   # down
]

class ParallelUnityManager:
    def __init__(self, num_envs):
        self.num_envs = num_envs
        self.envs = []
        self.reward_cum = []
        self.steps = []

    def make_env(self, worker_id):
    
        string_channel = StringSideChannel()
        custom_channel = CustomDataChannel()

        reward_init = [0, 0]
        custom_channel.send_data(serve=212, p1=reward_init[0], p2=reward_init[1])

        # remember change the unity_env: variable file_name to env path
        unity_env = UnityEnvironment(file_name=r"change to your env path",worker_id=worker_id,base_port=BASE_PORT,side_channels=[string_channel, custom_channel],no_graphics=False)

        env = UnityParallelEnv(unity_env)
        env.reset()

        return {
            "env": env,
            "unity_env": unity_env,
            "string_channel": string_channel,
            "custom_channel": custom_channel,
            "done": False
        }

    def start(self):
        print("Starting parallel Unity environments...")
        for i in range(self.num_envs):
            print(f"Creating env {i + 1} with worker_id={i}")
            env_dict = self.make_env(worker_id=i)
            self.envs.append(env_dict)
            self.reward_cum.append([0, 0])
            self.steps.append(0)
        print(f"Started {len(self.envs)} environments.")

    def step_all(self):
        for env_idx, env_pack in enumerate(self.envs): 
            if env_pack["done"]:
                continue

            env = env_pack["env"]

            if not env.agents:
                env_pack["done"] = True
                continue

            side = (self.steps[env_idx] // STEPS_PER_SIDE) % 4
            action = square_directions[side]

            actions = {
                env.agents[0]: action,
                env.agents[1]: action
            }

            observation, reward, done, info = env.step(actions) 

            self.reward_cum[env_idx][0] += reward[env.agents[0]]
            self.reward_cum[env_idx][1] += reward[env.agents[1]]

            print(f"[Env {env_idx + 1}] reward_cum={self.reward_cum[env_idx]} done={done}")

            if done[env.agents[0]] or done[env.agents[1]]:
                env_pack["done"] = True
                continue

            if env_idx == 0:
                obs = observation[env.agents[0]]['observation'][0]
                img = np.transpose(obs, (1, 2, 0))
                img_uint8 = cv2.cvtColor((img * 255).astype(np.uint8), cv2.COLOR_RGB2BGR)
                cv2.imshow("Camera_env0", img_uint8)

            self.steps[env_idx] += 1

    def all_done(self):
        return all(e["done"] for e in self.envs)

    def close(self):
        print("Closing environments")
        for env_pack in self.envs:
            try:
                env_pack["env"].close()
            except Exception as e:
                print("Error while closing env:", e)
        cv2.destroyAllWindows()


if __name__ == "__main__":
    print("Hello dPickleBall Trainer")

    multi_env = ParallelUnityManager(NUM_ENVS)

    try:
        multi_env.start()

        while not multi_env.all_done():
            multi_env.step_all()

            if cv2.waitKey(1) & 0xFF == ord('q'):
                print("Quitting display")
                break

    except KeyboardInterrupt:
        print("Interrupted by user")
    finally:
        multi_env.close()