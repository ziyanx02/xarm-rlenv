import xarm_env
import numpy as np
import cv2

env = xarm_env.make("lift", obs_mode="visual", use_ViT=False)
obs = env.reset()
cv2.imwrite("initial.jpg", obs)
obs, _, _, _ = env.step(np.array([1.0, 0, 0, 0], dtype=np.float16))
obs, _, _, _ = env.step(np.array([0, 1.0, 0, 0], dtype=np.float16))
obs, _, _, _ = env.step(np.array([0, 0, 1.0, 0], dtype=np.float16))
obs, _, _, _ = env.step(np.array([0, 0, 0, 1.0], dtype=np.float16))
cv2.imwrite("final.jpg", obs)