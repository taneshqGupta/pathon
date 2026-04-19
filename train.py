import gymnasium as gym
from gymnasium import spaces
import numpy as np
import torch
import pathon_env  # the PyO3 module
from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.vec_env import DummyVecEnv, VecFrameStack
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.vec_env import VecNormalize
import json
import os

LATEST_REPLAY_JSON = "[]"

class PathonGymEnv(gym.Env):
    metadata = {"render_modes": ["console"]}

    def __init__(self):
        super(PathonGymEnv, self).__init__()
        self.env = pathon_env.GridEnv(300, 200)
        self.action_space = spaces.Discrete(8)

        # Continuous observation space encapsulating FOV, LiDAR, and Grad
        # FOV: 15x15 = 225, LiDAR: 8, Grad: 2 -> Total = 235
        self.observation_space = spaces.Dict(
            {
                "fov": spaces.Box(low=0, high=255, shape=(15, 15), dtype=np.uint8),
                "lidar": spaces.Box(low=0.0, high=30.0, shape=(8,), dtype=np.float32),
                "grad": spaces.Box(low=-1.0, high=1.0, shape=(2,), dtype=np.float32),
            }
        )

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        fov, lidar, grad = self.env.reset_env()
        obs = self._get_obs(fov, lidar, grad)
        return obs, {}

    def step(self, action):
        fov, lidar, grad, reward, terminated, reached_goal = self.env.step_action(
            int(action)
        )
        obs = self._get_obs(fov, lidar, grad)
        return obs, reward, terminated, False, {}

    def _get_obs(self, fov, lidar, grad):
        return {
            "fov": np.frombuffer(fov, dtype=np.uint8).reshape((15, 15)).copy(),
            "lidar": np.array(lidar, dtype=np.float32),
            "grad": np.array(grad, dtype=np.float32),
        }


class JsonReplayCallback(BaseCallback):
    def __init__(self, record_freq=50000, replay_dir="./replays", verbose=1):
        super().__init__(verbose)
        self.record_freq = record_freq
        self.replay_dir = replay_dir
        os.makedirs(self.replay_dir, exist_ok=True)

    def _on_step(self) -> bool:
        # Trigger recording every record_freq timesteps
        if self.n_calls % self.record_freq == 0:
            self.record_episode()
        return True

    def record_episode(self):
        global LATEST_REPLAY_JSON
        
        if self.verbose > 0:
            print(f"\n[Callback] Recording evaluation episode at step {self.num_timesteps}...")

        record_env = DummyVecEnv([lambda: PathonGymEnv()])
        record_env = VecFrameStack(record_env, n_stack=4)

        obs = record_env.reset()
        done = False
        frames = []
        rust_env = record_env.envs[0].env

        step_count = 0
        while not done and step_count < 2000:
            action, _ = self.model.predict(obs, deterministic=True)
            obs, reward, done, info = record_env.step(action)
            state_dict = json.loads(rust_env.get_state_json())
            frames.append(state_dict)
            step_count += 1

        # 1. Update the in-memory variable for the HTTP server
        LATEST_REPLAY_JSON = json.dumps(frames)

        # 2. Save to disk as usual
        filepath = os.path.join(self.replay_dir, f"replay_step_{self.num_timesteps}.json")
        with open(filepath, "w") as f:
            f.write(LATEST_REPLAY_JSON)

        if self.verbose > 0:
            print(f"[Callback] Replay saved: {filepath} ({len(frames)} frames)\n")

if __name__ == "__main__":
    env = DummyVecEnv([lambda: PathonGymEnv()])
    env = VecFrameStack(env, n_stack=4)
    env = VecNormalize(env, norm_obs=False, norm_reward=True, clip_reward=10.0)

    model = PPO("MultiInputPolicy", env, verbose=1)
    # Instantiate the callback (e.g., record every 50,000 steps)
    replay_callback = JsonReplayCallback(record_freq=50000)

    print("Training Agent...")
    # Pass the callback into learn()
    model.learn(total_timesteps=1000000, callback=replay_callback)

    model.save("ppo_pathon")

    # ONNX Export
    class OnnxWrapper(torch.nn.Module):
        def __init__(self, ppo_actor):
            super().__init__()
            self.actor = ppo_actor

        def forward(self, fov, lidar, grad):
            features = self.actor.features_extractor(
                {"fov": fov, "lidar": lidar, "grad": grad}
            )
            return torch.argmax(
                self.actor.action_net(self.actor.mlp_extractor.forward_actor(features)),
                dim=1,
            )

    onnx_model = OnnxWrapper(model.policy)
    # Using dummy batch of 1
    dummy_obs = {
        "fov": torch.zeros((1, 4, 15, 15), dtype=torch.float32),
        "lidar": torch.zeros((1, 32), dtype=torch.float32),
        "grad": torch.zeros((1, 8), dtype=torch.float32),
    }

    torch.onnx.export(
        onnx_model,
        (dummy_obs["fov"], dummy_obs["lidar"], dummy_obs["grad"]),
        "../backend/model.onnx",
        opset_version=14,
        input_names=["fov", "lidar", "grad"],
        output_names=["action"],
    )
    print("Successfully exported to ONNX.")
