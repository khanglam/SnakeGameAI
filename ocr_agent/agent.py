import time
import pyautogui

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

class Agent:
    
    def __init__(self):
        super().__init__()
        self.observation_space = Box(low=0, high=255, shape=(1,83,100), dtype=np.uint8)
        self.action_space = Discrete(3)
        # Capture game frames
        self.screenshot = mss()
        # Get the size of the primary monitor
        screen_width, screen_height = pyautogui.size()

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
        pass

    def get_observation(self):
        raw = np.array(self.screenshot.grab(self.game_location))[:,:,:3].astype(np.uint8) # reshape to get 3 channels instead of 4. mss produces 4 channels for some reason. alpha channel?
        # Grayscale
        gray = cv2.cvtColor(raw, cv2.COLOR_BGR2GRAY)
        # Resize - Compress the 640x480 down to 83x100
        resized = cv2.resize(gray, (100,83))
        # Add channels first to suit stable_baseline format
        channel = np.reshape(resized, (1,83,100))
        return raw
    
    def get_done(self):
        done_cap = np.array(self.screenshot.grab(self.done_location))
        done_strings = ['GAME', 'GAHE']
        # done = False
        # res = pytesseract.image_to_string(done_cap)[:4]
        # if res in done_strings:
        #     done = True
        # return done, done_cap
        return done_cap

    def print_observation(self):
        plt.imshow(cv2.cvtColor(self.get_observation()[0], cv2.COLOR_GRAY2BGR))
        plt.show()

if __name__ == '__main__':
    # time.sleep(1) # So that I have time to alt tab to screenshot
    env = Agent()
    # done, done_cap = env.get_done()
    # print(done)
    plt.imshow(env.get_done())
    plt.show()
    # env.print_observation()
   