import time
import pyautogui
from snake_env import SnakeGameEnv, Direction, Point

# OpenCV allows frame processing
import cv2
import numpy as np
# Tesseract - OCR for "Game Over" extraction
from matplotlib import pyplot as plt
import time
from gym import Env
from gym.spaces import Box, Discrete

# Import os for file path management
import os 
# Import Base Callback for saving models
from stable_baselines3.common.callbacks import BaseCallback
# Check Environment    
from stable_baselines3.common import env_checker

# Build DQN and Train
from stable_baselines3 import DQN
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecFrameStack

# Log folder
CURRENT_DIR = os.getcwd()
CHECKPOINT_DIR = os.path.join(CURRENT_DIR, 'train')
LOG_DIR = os.path.join(CURRENT_DIR, 'logs')

class TrainAndLoggingCallback(BaseCallback):
    def __init__(self, check_freq, save_path, verbose=1):
        super(TrainAndLoggingCallback, self).__init__(verbose)
        self.check_freq = check_freq
        self.save_path = save_path

    def _init_callback(self):
        if self.save_path is not None:
            os.makedirs(self.save_path, exist_ok=True)

    def _on_step(self):
        if self.n_calls % self.check_freq == 0:
            model_path = os.path.join(self.save_path, 'best_model_{}'.format(self.n_calls))
            self.model.save(model_path)
        return True
    
 # Environment Init
env = SnakeGameEnv()
env.play_step()
env_checker.check_env(env) # check if environment is compatible with OpenAI gym

# callback = TrainAndLoggingCallback(check_freq=10000, save_path=CHECKPOINT_DIR)
# model = DQN('CnnPolicy', env, tensorboard_log=LOG_DIR, verbose=1, buffer_size=100000, learning_starts=5000, learning_rate=0.0001)

# model.load('ocr_agent/train/best_model_100000')
# model.learn(total_timesteps=100000, callback=callback)
# for episode in range(5):
#     observation = env.reset()
#     done = False
#     total_reward = 0
#     while not done: 
#         action, _ = model.predict(observation)
#         observation, reward, done, info = env.step(int(action))
#         # time.sleep(0.01)
#         total_reward += reward
#     print('Total Reward for episode {} is {}'.format(episode, total_reward))
#     time.sleep(0.3)
    

   
 
