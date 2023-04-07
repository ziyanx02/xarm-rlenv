
from xarm.wrapper import XArmAPI
import numpy as np
import gym
from gym import spaces
import pyrealsense2 as rs
import time

action_dtype = np.float16

def alert_user(code, info):
    if code == 0:
        return
    else:
        print("\n-----------------------------")
        print("WARNING!!! return code {} when {}".format(code, info))
        print("check XArm API Code for detail")
        input("---press ENTER to contimue---\n")

def set_position(arm, position, info):
    # set xarm position
    code = 1
    while code != 0:
        code = arm.set_position(*position, timeout=3, wait=True)
        alert_user(code, info)

class Base(gym.Env):

    def __init__(self, IP):

        self._arm = XArmAPI(port=IP, is_radian=False)
        self._arm.motion_enable(enable=True)
        self._arm.set_mode(0)
        self._arm.set_state(0)
        alert_user(self._arm.set_gripper_mode(0), "enable gripper")

        self.pipeline = rs.pipeline()
        config = rs.config()
        config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
        config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
        self.profile = self.pipeline.start(config)

        self.movement_scale = np.array([50, 50, 50, 60, 60, 60])
        self.gripper_scale = 200
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(7,), dtype=action_dtype)
        # ----------------------------
        # should be defined in wrapper
        self.observation_space = None
        self.initial_pos = None
        self.pos_hbound = None
        self.pos_lbound = None
        # ----------------------------

    def init(self, cfg):
        self.initial_pos = cfg["initial_pos"]
        self.pos_hbound = cfg["pos_hbound"]
        self.pos_lbound = cfg["pos_lbound"]
        self.initial_gripper = cfg["initial_gripper"]
        self.gripper_hbound = cfg["gripper_hbound"]
        self.gripper_lbound = cfg["gripper_lbound"]
        return

    def reset(self):
        set_position(self._arm, self.initial_pos, "reset")
        alert_user(self._arm.set_gripper_position(0, timeout=3, wait=True), "reset gripper")
        alert_user(self._arm.set_gripper_position(self.initial_gripper, timeout=3, wait=True), "reset gripper")
        time.sleep(1)
        return self._get_obs()

    def _get_obs(self):
        self.last_pos = self._arm.position
        self.gripper_pos = self._arm.get_gripper_position()
        frames = self.pipeline.wait_for_frames()
        depth_frame = frames.get_depth_frame()
        color_frame = frames.get_color_frame()
        pc = rs.pointcloud()
        pc.map_to(color_frame)
        cloud = pc.calculate(depth_frame)
        image = np.asanyarray(color_frame.get_data())
        return image, cloud, self.last_pos, self.gripper_pos

    def get_reward(self, obs):
        raise NotImplementedError
    
    def is_success(self, obs):
        raise NotImplementedError
    
    def step(self, action):
        assert action.shape == (7,)
        next_pos = np.clip(self.movement_scale * action[:-1] + np.array(self.last_pos), self.pos_lbound, self.pos_hbound)
        gripper_pos = np.clip(self.gripper_scale * action[-1] + self.gripper_pos[1], self.gripper_lbound, self.gripper_hbound)
        set_position(self._arm, next_pos, "step arm pos")
        if action[-1] > 0.001:
            alert_user(self._arm.set_gripper_position(gripper_pos, timeout=3, wait=True), "step gripper pos")
        obs = self._get_obs()
        reward = self.get_reward(obs)
        done = False
        success = self.is_success(obs)
        info = {'is_success': success, 'success': success}
        return obs, reward, done, info
    
    def move(self, movement):
        assert movement.shape == (7,)
        next_pos = np.clip(movement[:-1] + np.array(self.last_pos), self.pos_lbound, self.pos_hbound)
        gripper_pos = np.clip(movement[-1] + self.gripper_pos[1], self.gripper_lbound, self.gripper_hbound)
        set_position(self._arm, next_pos, "move arm pos")
        if movement[-1] > 0.1:
            alert_user(self._arm.set_gripper_position(gripper_pos, timeout=3, wait=True), "move gripper pos")
        obs = self._get_obs()
        return obs

    @property
    def position(self):
        return self._arm.position
    
    @property
    def gripper_position(self):
        return self._arm.get_gripper_position()
    
    def get_frame(self):
        # return rgb and depth frame
        frames = self.pipeline.wait_for_frames()
        depth_frame = frames.get_depth_frame()
        color_frame = frames.get_color_frame()
        return color_frame, depth_frame
    
class XYMovement(Base):
    # only movement on XY is allowed

    def __init__(self, IP):
        super().__init__(IP)
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(2,), dtype=action_dtype)
        self.action_padding = np.array([0, 0, 0, 0, 0], dtype=action_dtype)
        self.movement_padding = np.array([0, 0, 0, 0, 0], dtype=action_dtype)

    def step(self, action):
        assert action.shape == (2,)
        assert self.action_space.contains(action), "%r (%s) invalid"%(action, type(action))
        full_action = np.concatenate([action, self.action_padding])
        return super().step(full_action)
    
    def move(self, movement):
        assert movement.shape == (2,)
        full_movement = np.concatenate([movement, self.movement_padding])
        return super().step(full_movement)

class XYZMovement(Base):
    # only movement on XYZ is allowed

    def __init__(self, IP):
        super().__init__(IP)
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(3,), dtype=action_dtype)
        self.action_padding = np.array([0, 0, 0, 0], dtype=action_dtype)
        self.movement_padding = np.array([0, 0, 0, 0], dtype=action_dtype)

    def step(self, action):
        assert action.shape == (3,)
        assert self.action_space.contains(action), "%r (%s) invalid"%(action, type(action))
        full_action = np.concatenate([action, self.action_padding])
        return super().step(full_action)
    
    def move(self, movement):
        assert movement.shape == (3,)
        full_movement = np.concatenate([movement, self.movement_padding])
        return super().step(full_movement)

class XYZGMovement(Base):
    # only movement on XYZ and gripper is allowed

    def __init__(self, IP):
        super().__init__(IP)
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(4,), dtype=action_dtype)
        self.action_padding = np.array([0, 0, 0], dtype=action_dtype)
        self.movement_padding = np.array([0, 0, 0], dtype=action_dtype)

    def step(self, action):
        assert action.shape == (4,)
        assert self.action_space.contains(action), "%r (%s) invalid"%(action, type(action))
        full_action = np.concatenate([action[:-1], self.action_padding, action[-1:]])
        return super().step(full_action)

    def move(self, movement):
        assert movement.shape == (4,)
        full_movement = np.concatenate([movement[:-1], self.movement_padding, movement[-1:]])
        return super().step(full_movement)