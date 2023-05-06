import time
import pyautogui
from snake_game_ai import SnakeGameAI, Direction, Point

# AI Imports
# MSS used for screen capture
from mss import mss

# Sending commands
import pydirectinput

# OpenCV allows frame processing
import cv2
import numpy as np
# Tesseract - OCR for "Game Over" extraction
import pytesseract
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


class Agent(Env):

    def __init__(self):
        super().__init__()
        self.observation_space = Box(low=0, high=255, shape=(1,83,100), dtype=np.uint8)
        self.action_space = Discrete(3)
        self.reward_range = (-float('inf'), float('inf')) # This is needed as part of gym.Env interface
        # Capture game frames
        self.screenshot = mss()
        # Get the size of the primary monitor
        screen_width, screen_height = pyautogui.size()
        # Faster Key Presses
        pydirectinput.PAUSE=0
        # Calculate the center of the screen
        center_x = screen_width // 2
        center_y = screen_height // 2
        game_location_w = 640
        game_location_h = 480
        done_location_w = 130
        done_location_h = 30

        # Calculate the game location based on the center of the screen
        self.game_location = {'top': center_y - game_location_h//2, 'left': center_x - game_location_w//2, 'width': game_location_w, 'height': game_location_h}
        self.done_location = {'top': center_y - done_location_h//2, 'left': center_x - done_location_w//2, 'width': done_location_w, 'height': done_location_h}
    
    def step(self, action):
        # action = [straight, right, left]
        total_moves = [Direction.RIGHT, Direction.DOWN, Direction.LEFT, Direction.UP]
        action_map = ['right','down','left','up']
        index = total_moves.index(snake.direction)
        if action==0:
            new_dir = action_map[index] # no change
        elif action==1:
            next_index = (index + 1) % 4
            new_dir = action_map[next_index] # right turn (r > d > l > u)
        else: # action==2:
            next_index = (index - 1) % 4
            new_dir = action_map[next_index] # left turn (r > u > l > d)

        if action !=0:
            pydirectinput.press(new_dir) 

        reward, done, score = snake.play_step()
        observation = self.get_observation()
        info = {} # This is required as part of OpenAI's gym
        return observation, reward, done, info
    
    def reset(self):
        # time.sleep(1)
        # pydirectinput.click(x=150, y=150)
        # pydirectinput.press('space')
        snake.reset()
        return self.get_observation() 

    def get_observation(self):
        raw = np.array(self.screenshot.grab(self.game_location))[:,:,:3].astype(np.uint8) # reshape to get 3 channels instead of 4. mss produces 4 channels for some reason. alpha channel?
        # Grayscale
        gray = cv2.cvtColor(raw, cv2.COLOR_BGR2GRAY)
        # Resize - Compress the 640x480 down to 83x100
        resized = cv2.resize(gray, (100,83))
        # Add channels first to suit stable_baseline format
        channel = np.reshape(resized, (1,83,100))
        return channel
    
    def get_done(self):
        # Game Over Text Capture
        done_cap = np.array(self.screenshot.grab(self.done_location))
        # Accepted Done Texts
        done_strings = ['GAME', 'GAHE']

        # OCR
        done = False
        res = pytesseract.image_to_string(done_cap)[:4] # Fetch first 4 characters
        if res in done_strings:
            done = True
        return done, done_cap

    def print_observation(self):
        plt.imshow(cv2.cvtColor(self.get_observation()[0], cv2.COLOR_GRAY2BGR))
        plt.show()

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
    
if __name__ == '__main__':
    # Log folder
    CHECKPOINT_DIR = 'ocr_agent/train/'
    LOG_DIR = 'ocr_agent/logs/'

    # Environment Init
    snake = SnakeGameAI()
    snake.play_step()
    env = Agent()
    time.sleep(2)
    env.print_observation()
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