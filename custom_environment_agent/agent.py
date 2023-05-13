from snake_env import SnakeGameEnv, Direction, Point

# OpenCV allows frame processing
import cv2
import numpy as np
# Import os for file path management
import os 
# Import Base Callback for saving models
from stable_baselines3.common.callbacks import BaseCallback
# Check Environment    
from stable_baselines3.common import env_checker

# Build DQN and Train
from stable_baselines3 import DQN

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


def train_model():
    env = SnakeGameEnv(speed=100000)
    callback = TrainAndLoggingCallback(check_freq=50000, save_path=CHECKPOINT_DIR)
    model = DQN('MlpPolicy', env, tensorboard_log=LOG_DIR, verbose=1, buffer_size=50000, learning_starts=10000, learning_rate=0.00005, exploration_fraction=0.5, exploration_final_eps=0.1)
    # model.load(os.path.join(CHECKPOINT_DIR, 'best_model_350000'))
    model.learn(total_timesteps=1000000, callback=callback)

#   _______ ______  _____ _______     __  __  ____  _____  ______ _      
#  |__   __|  ____|/ ____|__   __|   |  \/  |/ __ \|  __ \|  ____| |     
#     | |  | |__  | (___    | |      | \  / | |  | | |  | | |__  | |     
#     | |  |  __|  \___ \   | |      | |\/| | |  | | |  | |  __| | |     
#     | |  | |____ ____) |  | |      | |  | | |__| | |__| | |____| |____ 
#     |_|  |______|_____/   |_|      |_|  |_|\____/|_____/|______|______|

def test_model(best_model):
    env = SnakeGameEnv()
    model = DQN('MlpPolicy', env, tensorboard_log=LOG_DIR, verbose=1, buffer_size=50000, learning_starts=10000, learning_rate=0.00005, exploration_fraction=0.5, exploration_final_eps=0.1)
    model.load(os.path.join(CHECKPOINT_DIR, best_model))
    for episode in range(1000):
        observation = env.reset()
        done = False
        total_reward = 0
        while not done: 
            action, _ = model.predict(observation)
            observation, reward, done, info = env.step(int(action))
            # time.sleep(0.01)
            total_reward += reward
        print('Total Reward for episode {} is {}'.format(episode, total_reward))
    

#   _____  ______ ____  _    _  _____ 
#  |  __ \|  ____|  _ \| |  | |/ ____|
#  | |  | | |__  | |_) | |  | | |  __ 
#  | |  | |  __| |  _ <| |  | | | |_ |
#  | |__| | |____| |_) | |__| | |__| |
#  |_____/|______|____/ \____/ \_____|
                                    
# print(env.get_observation())
# print(env.observation_space.sample())
# observation, reward, done, info = env.step(int(env.action_space.sample()))
# env_checker.check_env(env) # check if environment is compatible with OpenAI gym   
 
# train_model()
test_model("best_model_1000000")