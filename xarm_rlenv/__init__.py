from collections import OrderedDict

import gym
from gym.utils import seeding
import cv2
import numpy as np
from transformers import ViTImageProcessor, ViTModel
from xarm_rlenv.xarm_env import Reach, Lift, Base

default_IP = '192.168.1.209' # should be modified for each new xarm device

TASKS = OrderedDict((
	('reach', {
		'env': Reach,
		'action_space': 'xy',
		'episode_length': 50,
		'description': 'Reach a target location using end effector'
	}),
	('lift', {
		'env': Lift,
		'action_space': 'xyzg',
		'episode_length': 50,
		'description': 'Lift a cube using the end effector'
	}),
	('base', {
		'env': Base,
		'action_space': 'xyzrpyg',   # x y z roll pitch yaw gripper
		'episode_length': 200,
		'description': 'All dimension movement w/o task setting'
	}),
))

class XarmWrapper(gym.Wrapper):

    def __init__(self, env, obs_mode, use_ViT=False, image_size=84):
        super().__init__(env)
        self._env = env
        self.obs_mode = obs_mode
        self.use_ViT = use_ViT
        if self.obs_mode == "full":
            self.use_ViT = True
        self.image_size = image_size
        image_shape = (image_size, image_size, 3)
        if self.use_ViT:
            self.ViT_processor = ViTImageProcessor.from_pretrained('google/vit-base-patch16-224')
            self.ViT_model = ViTModel.from_pretrained('google/vit-base-patch16-224')
            image_shape = (197, 768)
        if self.obs_mode == 'state':
            self.observation_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(7,), dtype=np.float16)
        elif self.obs_mode == 'visual':
            self.observation_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=image_shape, dtype=np.float32)
        elif self.obs_mode == 'all':
            self.observation_space = gym.spaces.Dict(
				state=gym.spaces.Box(low=-np.inf, high=np.inf, shape=(7,), dtype=np.float16),
				visual=gym.spaces.Box(low=-np.inf, high=np.inf, shape=image_shape, dtype=np.float32)
			)
        elif self.obs_mode == 'full':
            print("\n---------------------------------------------------")
            print("current obs mode provides full obsevation including")
            print("[original image, output of ViT, 7-Dimension state]")
            print("---------------------------------------------------\n")
        else:
            raise ValueError(f'Unknown obs_mode {obs_mode}. Must be one of [visual, all, state, full]')

    def transform_obs(self, obs):
        # postprocess should be implemented here
        image, pointcloud, position, gripper_pos = obs
        if self.obs_mode == "full":
            input = self.ViT_processor(image, return_tensors="pt")
            output = self.ViT_model(input["pixel_values"]).last_hidden_state
            output = output.squeeze().detach().numpy()
            state = np.concatenate([position, gripper_pos])
            return image, output, state
        elif self.obs_mode == "state":
            return np.concatenate([position, gripper_pos])
        else:
            if self.use_ViT:
                input = self.ViT_processor(image, return_tensors="pt")
                output = self.ViT_model(input["pixel_values"]).last_hidden_state
                output = output.squeeze().detach().numpy()
            else:
                output = cv2.resize(image, (self.image_size, self.image_size))
            if self.obs_mode == "visual":
                return output
            elif self.obs_mode == "all":
                return dict(
                    state=np.concatenate([position, gripper_pos]),
                    visual=output
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

def make(task, obs_mode="visual", use_ViT=False, image_size=224, IP=default_IP, seed=1):
    if task not in TASKS:
        raise ValueError(f'Unknown task {task}. Must be one of {list(TASKS.keys())}')
    env = TASKS[task]['env'](IP)
    env = TimeLimit(env, TASKS[task]['episode_length'])
    env = XarmWrapper(env, obs_mode, use_ViT=use_ViT, image_size=image_size)
    env.seed(seed)
    return env