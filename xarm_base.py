
from xarm.wrapper import XArmAPI
from binghao_calib.cali_desktop import Recorder
import numpy as np
import gym
from gym import spaces

action_dtype = np.float32

def alert_user(code, info=None):
    if code == 0:
        return
    else:
        print("\n-----------------------------")
        print("WARNING!!! return code {} when {}".format(code, info))
        print("check XArm API Code for detail")
        input("---press ENTER to contimue---\n")

def postprocess_obs(image, pointcloud, position):
    # postprocess of 
    return image

class Base(gym.Env):

    def __init__(self, IP):
        self._arm = XArmAPI(port=IP, is_radian=False)
        self._arm.motion_enable(enable=True)
        self._arm.set_mode(0)
        self._arm.set_state(0)
        alert_user(self._arm.set_gripper_mode(0), "enable gripper")
        self.movement_scale = np.array([50, 50, 50, 20, 20, 20])
        self.gripper_scale = 0 # ???
        self._recorder = Recorder()
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(7,), dtype=action_dtype)
        self.observation_space = None # ???

    def reset(self):
        alert_user(self._arm.move_gohome(wait=True), "reset")
        obs = self._get_obs()
        self.last_pos = self._arm.position
        self.gripper_pos = self._arm.get_gripper_position()
        return obs

    def _get_obs(self):
        self.last_pos = self._arm.position
        self.gripper_pos = self._arm.get_gripper_position()
        rgb, pc = self._recorder.fetch_color_and_pc()
        return (rgb, pc, self.last_pos, self.gripper_pos)

    def get_reward(self, image, pointcloud, position, gripper_pos):
        raise NotImplementedError
    
    def is_success(self, image, pointcloud, position, gripper_pos):
        raise NotImplementedError
    
    def step(self, action):
        assert action.shape == (7,)
        assert self.action_space.contains(action), "%r (%s) invalid"%(action, type(action))
        next_pos = self.movement_scale * action[:-1] + np.array(self.last_pos)
        gripper_pos = self.gripper_scale * action[-1] + self.gripper_pos
        # pos limitation ???
        # gripper limitation ???
        alert_user(self._arm.set_position(*next_pos, timeout=3, wait=True), "set position")
        alert_user(self._arm.set_gripper_position(gripper_pos, timeout=1, wait=True), "set gripper pos")
        obs = self._get_obs()
        reward = self.get_reward(obs)
        done = False
        success = self.is_success(obs)
        info = {'is_success': success, 'success': success}
        return obs, reward, done, info
    
    @property
    def position(self):
        return self._arm.position
    
    @property
    def gripper_position(self):
        return self._arm.get_gripper_position()
    
    def get_frame(self):
        rgb, _ = self._recorder.fetch_color_and_pc()
        return rgb
    
class XYMovement(Base):

    def __init__(self, IP):
        super().__init__(IP)
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(2,), dtype=action_dtype)
        self.action_padding = np.array([0, 0, 0, 0, 0], dtype=action_dtype)

    def step(self, action):
        assert action.shape == (2,)
        assert self.action_space.contains(action), "%r (%s) invalid"%(action, type(action))
        full_action = np.concatenate(action, self.action_padding)
        return super().step(full_action)

class XYZMovement(Base):

    def __init__(self, IP):
        super().__init__(IP)
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(3,), dtype=action_dtype)
        self.action_padding = np.array([0, 0, 0, 0], dtype=action_dtype)

    def step(self, action):
        assert action.shape == (3,)
        assert self.action_space.contains(action), "%r (%s) invalid"%(action, type(action))
        full_action = np.concatenate(action, self.action_padding)
        return super().step(full_action)

class XYZGMovement(Base):

    def __init__(self, IP):
        super().__init__(IP)
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(4,), dtype=action_dtype)
        self.action_padding = np.array([0, 0, 0], dtype=action_dtype)

    def step(self, action):
        assert action.shape == (4,)
        assert self.action_space.contains(action), "%r (%s) invalid"%(action, type(action))
        full_action = np.concatenate(action, self.action_padding)
        return super().step(full_action)
