from collections import OrderedDict

import xarm_base as base
import numpy as np
import gym
from gym.utils import seeding
from transformers import ViTImageProcessor, ViTModel

xarm_IP = '192.168.1.209'

class Reach(base.XYMovement):
    def __init__(self, IP):
        super().__init__(IP)

    def get_reward(self, obs):
        image, pointcloud, position, gripper_pos = obs
        return 0

    def is_success(self, obs):
        image, pointcloud, position, gripper_pos = obs
        return False

class XarmWrapper(gym.Wrapper):

    def __init__(self, env, obs_mode, image_size):
        super().__init__(env)
        self._env = env
        self.obs_mode = obs_mode
        self.image_size = image_size
        if self.obs_mode == 'state':
            self.observation_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(7,), dtype=np.float16)
        elif self.obs_mode == 'visual':
            self.observation_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(197, 768), dtype=np.float32)
        elif self.obs_mode == 'all':
            self.observation_space = gym.spaces.Dict(
				state=gym.spaces.Box(low=-np.inf, high=np.inf, shape=(7,), dtype=np.float16),
				visual=gym.spaces.Box(low=-np.inf, high=np.inf, shape=(197, 768), dtype=np.float32)
			)
        else:
            raise ValueError(f'Unknown obs_mode {obs_mode}. Must be one of [visual, all, state]')
        self.ViT_processor = ViTImageProcessor.from_pretrained('google/vit-base-patch16-224')
        self.ViT_model = ViTModel.from_pretrained('google/vit-base-patch16-224')
        xarm_cfg = {
              "initial_pos": np.array([201.5, 0, 140.5, -180, 0, 0]),
              "pos_hbound": np.array([301.5, 100, 240.5, -180, 0, 0]),
              "pos_lbound": np.array([101.5, -100, 40.5, -180, 0, 0]),
              "initial_gripper": 600,
              "gripper_hbound": 600,
              "gripper_lbound": 0
		}
        self._env.init(xarm_cfg)

    def transform_obs(self, obs):
        # postprocess should be implemented here
        image, pointcloud, position, gripper_pos = obs
        if self.obs_mode == "state":
            return np.concatenate([position, gripper_pos])
        else:
            inputs = self.ViT_processor(image, return_tensors="pt")
            outputs = self.ViT_model(inputs)
            outputs = outputs.squeeze().detach().numpy()
            if self.obs_mode == "visual":
                return outputs
            elif self.obs_mode == "all":
                return dict(
                    state=np.concatenate([position, gripper_pos]),
                    visual=outputs
                )

    def reset(self):
        return self.transform_obs(self._env.reset())
		
    def step(self, action):
        obs, reward, done, info = self._env.step(action)
        return self.transform_obs(obs), reward, done, info
        
    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

class TimeLimit(gym.Wrapper):
	# from gym

    def __init__(self, env, max_episode_steps=None):
        super(TimeLimit, self).__init__(env)
        if max_episode_steps is None and self.env.spec is not None:
            max_episode_steps = env.spec.max_episode_steps
        if self.env.spec is not None:
            self.env.spec.max_episode_steps = max_episode_steps
        self._max_episode_steps = max_episode_steps
        self._elapsed_steps = None

    def step(self, action):
        assert (
            self._elapsed_steps is not None
        ), "Cannot call env.step() before calling reset()"
        observation, reward, done, info = self.env.step(action)
        self._elapsed_steps += 1
        if self._elapsed_steps >= self._max_episode_steps:
            info["TimeLimit.truncated"] = not done
            done = True
        return observation, reward, done, info

    def reset(self, **kwargs):
        self._elapsed_steps = 0
        return self.env.reset(**kwargs)

TASKS = OrderedDict((
	('reach', {
		'env': Reach,
		'action_space': 'xy',
		'episode_length': 50,
		'description': 'Reach a target location with the end effector'
	}),
))

def make(task, obs_mode="visual", image_size=84, seed=1):
    if task not in TASKS:
        raise ValueError(f'Unknown task {task}. Must be one of {list(TASKS.keys())}')
    env = TASKS[task]['env'](xarm_IP)
    env = TimeLimit(env, TASKS[task]['episode_length'])
    env = XarmWrapper(env, obs_mode, image_size)
    env.seed(seed)
    return env