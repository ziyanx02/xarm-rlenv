
import xarm_rlenv.xarm_base as base
import numpy as np
import random

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
