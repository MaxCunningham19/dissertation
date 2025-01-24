import gymnasium as gym
import mo_gymnasium as mo_gym
from moviepy.video.io.ImageSequenceClip import ImageSequenceClip
from gym.wrappers.record_video import RecordVideo
import numpy as np

gym.pprint_registry()

# The highway reward vec is [speed, right lane, collision]
# the rward space is [(0,1),(0,1),(-1,0)]
env = mo_gym.make("mo-highway-v0", render_mode="rgb_array", max_episode_steps=100)
env = RecordVideo(env, "videos/demo", episode_trigger=lambda e: e % 10 == 0, name_prefix="video")
# this is needed so that the video records smoothly :shrug:
env.unwrapped.set_record_video_wrapper(env)
print(env.action_space)
for i in range(5):
    obs, info = env.reset()
    done = truncated = False
    while not (done or truncated):
        obs, vec_reward, done, truncated, info = env.step(env.action_space.sample())
        np_vec = np.array(vec_reward)

env.close()
