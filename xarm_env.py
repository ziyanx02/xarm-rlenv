from collections import OrderedDict

import xarm_base as base
import numpy as np
import gym
from gym.wrappers import TimeLimit
from gym.utils import seeding

xarm_IP = '192.168.1.113'

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
		"""if obs_mode == 'state':
			self.observation_space = env.observation_space
		elif obs_mode == 'rgb':
			self.observation_space = gym.spaces.Box(low=0, high=255, shape=(3, image_size, image_size), dtype=np.uint8)
		elif obs_mode == 'all':
			self.observation_space = gym.spaces.Dict(
				state=gym.spaces.Box(low=-np.inf, high=np.inf, shape=(4,), dtype=np.float32),
				rgb=gym.spaces.Box(low=0, high=255, shape=(3, image_size, image_size), dtype=np.uint8)
			)
		else:
			raise ValueError(f'Unknown obs_mode {obs_mode}. Must be one of [rgb, all, state]')"""

	def transform_obs(self, obs):
        # postprocess should be implemented here
		image, pointcloud, position, gripper_pos = obs
		return image

	def reset(self):
		return self.transform_obs(self._env.reset())
		
	def step(self, action):
		obs, reward, done, info = self._env.step(action)
		return self.transform_obs(obs), reward, done, info
        
	def seed(self, seed=None):
		self.np_random, seed = seeding.np_random(seed)
		return [seed]


TASKS = OrderedDict((
	('reach', {
		'env': Reach,
		'action_space': 'xy',
		'episode_length': 50,
		'description': 'Reach a target location with the end effector'
	}),
))

def make(task, obs_mode="rgb", image_size=84, seed=1):
    if task not in TASKS:
        raise ValueError(f'Unknown task {task}. Must be one of {list(TASKS.keys())}')
    env = TASKS[task]['env'](xarm_IP)
    env = TimeLimit(env, TASKS[task]['episode_length'])
    env = XarmWrapper(env, obs_mode, image_size)
    env.seed(seed)
    return env