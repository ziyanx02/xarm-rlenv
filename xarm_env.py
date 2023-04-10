from collections import OrderedDict

import xarm_rlenv.xarm_base as base
import numpy as np
import gym
import cv2
from gym.utils import seeding
import random
from transformers import ViTImageProcessor, ViTModel

xarm_IP = '192.168.1.209' # should be modified for each new xarm device

class Reach(base.XYMovement):
    def __init__(self, IP):
        super().__init__(IP)
        xarm_cfg = {
              "initial_pos": np.array([206, 0, 20, -180, 0, 0]),
              "pos_hbound": np.array([500, 250, 200, -180, 0, 0]),
              "pos_lbound": np.array([180, -250, 0, -180, 0, 0]),
              "initial_gripper": 0,
              "gripper_hbound": 0,
              "gripper_lbound": 0
		}
        self.init(xarm_cfg)
        self.target_pos = np.array([468, 40, 20, -180, 0, 0])
        self.succeeded = False

    def get_reward(self, obs):
        image, pointcloud, position, gripper_pos = obs
        success = self.is_success(obs)
        reward = 100 if success and not self.succeeded else 0
        self.succeeded = self.succeeded or success
        return reward

    def is_success(self, obs):
        image, pointcloud, position, gripper_pos = obs
        distance = np.sum((position[:2] - self.target_pos[:2]) ** 2)
        return distance < 2500
    
    def reset(self):
        super().reset()
        random_movement = np.array([random.randint(-100, 100), random.randint(-300, 300)])
        return self.move(random_movement)

class Lift(base.XYZGMovement):
    def __init__(self, IP):
        super().__init__(IP)
        xarm_cfg = {
              "initial_pos": np.array([206, 0, 20, -180, 0, 0]),
              "pos_hbound": np.array([500, 250, 200, -180, 0, 0]),
              "pos_lbound": np.array([180, -250, 0, -180, 0, 0]),
              "initial_gripper": 600,
              "gripper_hbound": 800,
              "gripper_lbound": 0
		}
        self.init(xarm_cfg)

    def get_reward(self, obs):
        image, pointcloud, position, gripper_pos = obs
        return 0

    def is_success(self, obs):
        image, pointcloud, position, gripper_pos = obs
        return False
    
class Base(base.Base):
    def __init__(self, IP):
        super().__init__(IP)
        xarm_cfg = {
              "initial_pos": np.array([300, 0, 100, -180, 0, 0]),
              "pos_hbound": np.array([500, 250, 200, -150, 30, 30]),
              "pos_lbound": np.array([180, -250, 20, 150, -30, -30]),
              "initial_gripper": 600,
              "gripper_hbound": 800,
              "gripper_lbound": 0
		}
        self.init(xarm_cfg)

    def get_reward(self, obs):
        image, pointcloud, position, gripper_pos = obs
        return 0

    def is_success(self, obs):
        image, pointcloud, position, gripper_pos = obs
        return False
    
    def reset(self):
        super().reset()
        random_movement = np.array([
            random.randint(-100, 100),
            random.randint(-100, 100),
            random.randint(-100, 100),
            random.randint(-15, 15),
            random.randint(-15, 15),
            random.randint(-15, 15),
            random.randint(-300, 300),
        ])
        return self.move(random_movement)

class XarmWrapper(gym.Wrapper):

    def __init__(self, env, obs_mode, use_ViT=False, image_size=84):
        super().__init__(env)
        self._env = env
        self.obs_mode = obs_mode
        self.use_ViT = use_ViT
        if self.obs_mode == "full":
            self.use_ViT = True
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
        elif self.obs_mode == 'full':
            print("\n---------------------------------------------------")
            print("current obs mode provides full obsevation including")
            print("[original image, output of ViT, 7-Dimension state]")
            print("---------------------------------------------------\n")
        else:
            raise ValueError(f'Unknown obs_mode {obs_mode}. Must be one of [visual, all, state]')
        if self.use_ViT:
            self.ViT_processor = ViTImageProcessor.from_pretrained('google/vit-base-patch16-224')
            self.ViT_model = ViTModel.from_pretrained('google/vit-base-patch16-224')

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

def make(task, obs_mode="visual", seed=1, use_ViT=False, image_size=224):
    if task not in TASKS:
        raise ValueError(f'Unknown task {task}. Must be one of {list(TASKS.keys())}')
    env = TASKS[task]['env'](xarm_IP)
    env = TimeLimit(env, TASKS[task]['episode_length'])
    env = XarmWrapper(env, obs_mode, use_ViT=use_ViT, image_size=image_size)
    env.seed(seed)
    return env