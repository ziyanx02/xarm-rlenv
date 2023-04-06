import xarm_env
import numpy as np
import cv2

env = xarm_env.make()
image0 = env.reset()

image1 = env.step(np.array([1.0, 1.0], dtype=np.float16))