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
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecFrameStack

# Args to be passed from terminal execution
import argparse

class Agent(Env):

    def __init__(self):
        super().__init__()
        self.snake = SnakeGameAI()
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
        index = total_moves.index(self.snake.direction)
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

        reward, done, score = self.snake.play_step()
        observation = self.get_observation()
        info = {} # This is required as part of OpenAI's gym
        return observation, reward, done, info
    
    def reset(self):
        # time.sleep(1)
        # pydirectinput.click(x=150, y=150)
        # pydirectinput.press('space')
        self.snake.reset()
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

# Get the absolute path of the script directory
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
# Join the current directory with the desired subdirectories for checkpoints and logs
CHECKPOINT_DIR = os.path.join(SCRIPT_DIR, 'train')
LOG_DIR = os.path.join(SCRIPT_DIR, 'logs')

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
#   _______ _____            _____ _   _       __  __  ____  _____  ______ _      
#  |__   __|  __ \     /\   |_   _| \ | |     |  \/  |/ __ \|  __ \|  ____| |     
#     | |  | |__) |   /  \    | | |  \| |     | \  / | |  | | |  | | |__  | |     
#     | |  |  _  /   / /\ \   | | | . ` |     | |\/| | |  | | |  | |  __| | |     
#     | |  | | \ \  / ____ \ _| |_| |\  |     | |  | | |__| | |__| | |____| |____ 
#     |_|  |_|  \_\/_/    \_\_____|_| \_|     |_|  |_|\____/|_____/|______|______|

# Environment Init
def train_model(best_model=None):
    callback = TrainAndLoggingCallback(check_freq=10000, save_path=CHECKPOINT_DIR)
    # Single Agent
    env = Agent()
    # load previous checkpoint if available
    if best_model:
        model = PPO('CnnPolicy', env, tensorboard_log=LOG_DIR, verbose=1, learning_rate=0.0001)
        # model = DQN.load(os.path.join(CHECKPOINT_DIR, best_model), env)
    else:
        model = PPO('CnnPolicy', env, tensorboard_log=LOG_DIR, verbose=1, learning_rate=0.0001)
        # model = DQN('MlpPolicy', env, tensorboard_log=LOG_DIR, verbose=1, buffer_size=50000, learning_starts=10000, learning_rate=0.001)
    # start/resume training
    model.learn(total_timesteps=10000000, callback=callback)

def test_model(best_model):
    # env = SnakeGameAI(speed=50000)
    env = Agent()
    model = PPO.load(os.path.join(CHECKPOINT_DIR, best_model), env)
    iterations = 10
    for episode in range(iterations):
        observation = env.reset()
        done = False
        total_reward = 0
        while not done: 
            action, _ = model.predict(observation)
            observation, reward, done, info = env.step(int(action))
            # time.sleep(0.01)
            total_reward += reward
        print('Total Reward for episode {} is {}'.format(episode, total_reward))
        time.sleep(0.1)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Custom Environment Agent')
    parser.add_argument('command', type=str, choices=['train', 'evaluate'], help='Command to execute')
    parser.add_argument('--best_model', type=str, default='', help='Filename of the best model (default: "")')

    args = parser.parse_args()

    if args.command == 'train':
        if args.best_model == "":
            print("...Training from scratch...")
        else: print("...Training with '"+args.best_model+"' ...")
        train_model(args.best_model)
    elif args.command == 'evaluate':
        # Specify the best model to test
        if args.best_model == "":
            best_model = 'best_model'
        else: best_model = args.best_model
        test_model(best_model)