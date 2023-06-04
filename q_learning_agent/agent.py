import torch
import random
import numpy as np
from snake_game_ai import SnakeGameAI, Direction, Point
from collections import deque
from model import Linear_QNet, QTrainer
from helper import plot
import os
# Args to be passed from terminal execution
import argparse

# Hyperparameters
MAX_MEMORY = 100_000
BATCH_SIZE = 1000
LR = 0.0001
INPUT_SIZE = 11
HIDDEN_SIZE = 256
OUTPUT_SIZE = 3
EXPLORATION = 80

DEBUG=False

class Agent:
    def __init__(self):
        self.n_games = 0
        self.epsilon = 0 # randomness
        self.gamma = 0.9 # discount rate (< 1)
        self.memory = deque(maxlen=MAX_MEMORY) # popleft()
        self.model = Linear_QNet(INPUT_SIZE, HIDDEN_SIZE, OUTPUT_SIZE)
        self.trainer = QTrainer(self.model, lr=LR, gamma=self.gamma)

    def get_state(self, game):
        head = game.snake[0]
        point_l = Point(head.x - 20, head.y)
        point_r = Point(head.x + 20, head.y)
        point_u = Point(head.x, head.y - 20)
        point_d = Point(head.x, head.y + 20)
        
        dir_l = game.direction == Direction.LEFT
        dir_r = game.direction == Direction.RIGHT
        dir_u = game.direction == Direction.UP
        dir_d = game.direction == Direction.DOWN

        state = [
            # Danger straight
            (dir_r and game.is_collision(point_r)) or 
            (dir_l and game.is_collision(point_l)) or 
            (dir_u and game.is_collision(point_u)) or 
            (dir_d and game.is_collision(point_d)),

            # Danger right
            (dir_u and game.is_collision(point_r)) or 
            (dir_d and game.is_collision(point_l)) or 
            (dir_l and game.is_collision(point_u)) or 
            (dir_r and game.is_collision(point_d)),

            # Danger left
            (dir_d and game.is_collision(point_r)) or 
            (dir_u and game.is_collision(point_l)) or 
            (dir_r and game.is_collision(point_u)) or 
            (dir_l and game.is_collision(point_d)),
            
            # Move direction
            dir_l,
            dir_r,
            dir_u,
            dir_d,
            
            # Food location 
            game.food.x < game.head.x,  # food left
            game.food.x > game.head.x,  # food right
            game.food.y < game.head.y,  # food up
            game.food.y > game.head.y  # food down
            ]

        return np.array(state, dtype=int)

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done)) # popleft if MAX_MEMORY is reached

    def train_long_memory(self):
        if len(self.memory) > BATCH_SIZE:
            mini_sample = random.sample(self.memory, BATCH_SIZE) # list of tuples
        else:
            mini_sample = self.memory

        states, actions, rewards, next_states, dones = zip(*mini_sample)
        self.trainer.train_step(states, actions, rewards, next_states, dones)
        # for state, action, reward, next_state, done in mini_sample:
        #     self.trainer.train_step(state, action, reward, next_state, done)
    
    def train_short_memory(self, state, action, reward, next_state, done):
        self.trainer.train_step(state, action, reward, next_state, done)

    def get_action(self, state):
        # random moves: tradeoff exploration / exploitation
        # self.epsilon = max(EXPLORATION - self.n_games, 3)
        self.epsilon = EXPLORATION - self.n_games
        final_move = [0,0,0]
        if random.randint(0,200) < self.epsilon:
            if(DEBUG): 
                print("EXPLORE")
            move = random.randint(0,2)
            final_move[move] = 1
        else:
            if(DEBUG): 
                print("EXPLOIT")
            state0 = torch.tensor(state, dtype=torch.float)
            prediction = self.model(state0)
            move = torch.argmax(prediction).item()
            final_move[move] = 1

        return final_move
    
    # ADDED
    def get_play(self, state):
        self.model.eval()
        action = [0, 0, 0]
        state0 = torch.tensor(state, dtype=torch.float)
        prediction = self.model.forward(state0)
        move = torch.argmax(prediction).item()
        action[move] = 1

        return action
    
def train(best_model=None):
    plot_scores = []
    plot_mean_scores = []
    total_score = 0
    high_score = 0
    agent = Agent()
    # if(EXPLORATION == 0):
    if(best_model):
        agent.model.load(best_model)
    game = SnakeGameAI()
    while True:
        # get old state
        state_old = agent.get_state(game)

        # get move
        final_move = agent.get_action(state_old)

        # perform move and get new state
        reward, done, score = game.play_step(final_move)
        state_new = agent.get_state(game)

        # train short memory
        agent.train_short_memory(state_old, final_move, reward, state_new, done)

        # remember
        agent.remember(state_old, final_move, reward, state_new, done)

        if done:
            # train long memory (replay memory), plot result
            game.reset()
            agent.n_games += 1
            agent.train_long_memory()

            # Fetch High Score
            try:
                with open("high_score.txt", "r") as f:
                    high_score = int(f.read())
            except FileNotFoundError:
                high_score = 0

            if score > high_score:
                high_score = score
                with open("high_score.txt", "w") as f:
                    f.write(str(high_score))

                print('High Score Beat. Saving Model...')
                agent.model.save()

            print('Game:', agent.n_games, 'Score:', score, 'Best Score:', high_score)

            plot_scores.append(score)
            total_score += score
            mean_score = total_score / agent.n_games
            plot_mean_scores.append(mean_score)
            plot(plot_scores, plot_mean_scores)

def main():
    agent = Agent()
    agent.model.load()
    game = SnakeGameAI()
    while True:
        state_old = agent.get_state(game)
        action = agent.get_play(state_old)
        reward, game_over, score = game.play_step(action)
        if game_over:
            game.reset()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Custom Environment Agent')
    parser.add_argument('command', type=str, choices=['train', 'evaluate'], help='Command to execute')
    parser.add_argument('--best_model', type=str, default='', help='Filename of the best model (default: "")')

    args = parser.parse_args()

    if args.command == 'train':
        if args.best_model == "":
            print("...Training from scratch...")
        else: print("...Training with '"+args.best_model+"' ...")
        train(args.best_model)
    elif args.command == 'evaluate':
        # Specify the best model to test
        if args.best_model == "":
            best_model = 'best_model'
        else: best_model = args.best_model
        main(best_model)