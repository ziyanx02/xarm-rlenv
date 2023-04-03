import xarm_env as xarm
import numpy as np
import cv2

env = xarm.make("reach")

image0 = env.reset()
cv2.imwrite("./test_img/image0.jpg", image0)

image1 = env.step(np.array([1.0, 1.0], dtype=np.float16))
cv2.imwrite("./test_img/image1.jpg", image1)