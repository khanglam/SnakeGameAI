from snake_env import SnakeGameEnv
# Import os for file path management
import os 
# Import Base Callback for saving models
from stable_baselines3.common.callbacks import BaseCallback
# Check Environment    
from stable_baselines3.common import env_checker

# RL Algorithms
from stable_baselines3 import DQN
from stable_baselines3 import PPO

# RayTune for parallelized training
# import ray
# from ray import tune
# from ray.tune.schedulers import ASHAScheduler
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.callbacks import CheckpointCallback

# Args to be passed from terminal execution
import argparse

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
    callback = TrainAndLoggingCallback(check_freq=100000, save_path=CHECKPOINT_DIR)
    # Single Agent
    env = SnakeGameEnv(speed=10000)
    # load previous checkpoint if available
    if best_model:
        model = PPO.load(os.path.join(CHECKPOINT_DIR, best_model), env, learning_rate=0.0005)
        # model = DQN.load(os.path.join(CHECKPOINT_DIR, best_model), env)
    else:
        model = PPO('MlpPolicy', env, tensorboard_log=LOG_DIR, verbose=1, learning_rate=0.0005)
        # model = DQN('MlpPolicy', env, tensorboard_log=LOG_DIR, verbose=1, buffer_size=50000, learning_starts=10000, learning_rate=0.001)
    # start/resume training
    model.learn(total_timesteps=10000000, callback=callback)

def train_multi_agent(config, checkpoint_dir=None, checkpoint=None):
    # Create a vectorized environment with 2 instances of your environment
    env = SubprocVecEnv([new_env(10000) for _ in range(2)])

    # create a PPO agent
    agent = PPO('MlpPolicy', env, verbose=1, learning_rate=config['learning_rate'], 
                n_steps=config['n_steps'], batch_size=config['batch_size'], 
                gamma=config['gamma'], gae_lambda=config['gae_lambda'], 
                clip_range=config['clip_range'], ent_coef=config['ent_coef'],
                tensorboard_log="./logs")

    # load previous checkpoint if available
    if checkpoint_dir:
        agent = PPO.load(checkpoint_dir)

    # set up checkpoint saving
    callback = TrainAndLoggingCallback(check_freq=50000, save_path=CHECKPOINT_DIR)

    # train the agent
    agent.learn(total_timesteps=10000000, callback=callback)
#   _______ ______  _____ _______     __  __  ____  _____  ______ _      
#  |__   __|  ____|/ ____|__   __|   |  \/  |/ __ \|  __ \|  ____| |     
#     | |  | |__  | (___    | |      | \  / | |  | | |  | | |__  | |     
#     | |  |  __|  \___ \   | |      | |\/| | |  | | |  | |  __| | |     
#     | |  | |____ ____) |  | |      | |  | | |__| | |__| | |____| |____ 
#     |_|  |______|_____/   |_|      |_|  |_|\____/|_____/|______|______|
# Create a function to instantiate your environment
def new_env(speed):
    return SnakeGameEnv(speed)

def test_model(best_model):
    env = SnakeGameEnv(speed=50000)
    avg_reward = 0
    avg_score = 0
    iterations = 10
    model = PPO.load(os.path.join(CHECKPOINT_DIR, best_model), env)
    for episode in range(iterations):
        observation = env.reset()
        done = False
        total_reward = 0
        while not done: 
            action, _ = model.predict(observation)
            observation, reward, done, info = env.step(int(action))
            # time.sleep(0.01)
            total_reward += reward
        print('Episode {}: Total Reward: {}. Score: {}'.format(episode, total_reward, env.score))
        avg_reward += total_reward
        avg_score += env.score
    avg_reward /= iterations
    avg_score /= iterations
    print('Average Reward: {} Average Score: {}'.format(avg_reward, avg_score))
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
# train_model('best_model')
# test_model("best_model")

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
