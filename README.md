# XArm Reinforcement Learning Environment
Operating XArm in a gym-style API.

## Dependencies
[xArm-Python-SDK](https://github.com/xArm-Developer/xArm-Python-SDK) and [realsense2](https://pypi.org/project/pyrealsense2/) are required for building ```xarm_rlenv```.

Ufactory xArm robot and Intel RealSense camera are needed for using ```xarm_rlenv```.

## Usage
Initialize environment by:
```
import xarm_rlenv
env = xarm_rlenv.make(task[, obs_mode, use_ViT, image_size, IP, seed])
```
```task``` can be defined in ```xarm_env.py``` (should be added to \_\_init__.py) by defining following attributes:
```
xarm_cfg: safety boundary for movement and initial position after reset
```
```
def get_reward(self, obs):
Define reward function using observation including image, pointcloud, state
```
```
def get_done(self, obs):
Not implemented yet
```
```
def get_info(self, obs):
Not implemented yet
```
```
def reset(self):
Calling function reset of superclass will reset the xarm to the initial position defined in xarm_cfg. 
Random initialization can be implemented by calling self.move(movement) which directly changes the position of xarm.
```
### Action
Action is specified by ```task```:

```
  task   | action  |  size
---------|---------|--------
 "reach" |   xy    |  (2,)
 "lift"  |  xyzg   |  (4,)
 "Base"  | xyzrpyg |  (7,)
```
Actions should be a ```np.ndarray``` [with ```dtype=np.float16```] and limited in (-1.0, 1.0).

The projection rate from action to movement of xarm can be modified in ```xarm_base.Base```. The limitation of target position (safety boundary of expected position after action) should be limited in each environment by calling  ```xarm_base.Base.init(cfg)``` where ```cfg``` includes the safety boundary as well as initial position after reset.

### Observation
Observation is specified by ```obs_mode, use_ViT, image_size```.

If ```use_ViT=True```:
```
 obs_mode |   observation    |                 observation shape
----------|------------------|------------------------------------------------------
 "state"  |      state       |                       (7,)
 "visual" |      visual      |           (image_size, image_size, 3)
  "all"   |   visual+state   | {"visual":(image_size, image_size, 3), "state":(7,)}
  "full"  | visual+ViT+state |   [(image_size, image_size, 3), (197, 768), (7,)]
```
If ```use_ViT=True```, all visual outputs will be preprocessed by a fixed pretrained [ViT](https://huggingface.co/google/vit-base-patch16-224) and in shape of (197, 768) except when ```obs_mode="full"```.

## Acknowledgements
This repository is based on work by [Nicklas Hansen](https://nicklashansen.github.io/) as the author of simxarm environment.