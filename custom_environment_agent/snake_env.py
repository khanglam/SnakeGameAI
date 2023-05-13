import pygame
import random
from enum import Enum
from collections import namedtuple
import keyboard
import numpy as np
# Sending commands
from gym import Env
from gym.spaces import Box, Discrete
from collections import deque

pygame.init()
font = pygame.font.SysFont('arial', 25)

class Direction(Enum):
    RIGHT = 1
    LEFT = 2
    UP = 3
    DOWN = 4
    
Point = namedtuple('Point', 'x, y')

# rgb colors
WHITE = (255, 255, 255)
RED = (200,0,0)
BLUE1 = (0, 0, 255)
BLUE2 = (0, 100, 255)
BLACK = (0,0,0)

BLOCK_SIZE = 20

SNAKE_SIZE_GOAL=30
class SnakeGameEnv(Env):
    
    def __init__(self, w=640, h=480, speed=100):
        super().__init__()
        self.SPEED = speed
        self.observation_space = Box(low=-w, high=w, shape=(5+SNAKE_SIZE_GOAL,), dtype=np.float32)
        self.action_space = Discrete(3)
        self.w = w
        self.h = h
        # init display
        self.display = pygame.display.set_mode((self.w, self.h))
        pygame.display.set_caption('Snake Game')
        self.clock = pygame.time.Clock()
        self.reset()
        
    def reset(self):
        # init game state
        self.direction = Direction.RIGHT
        
        self.head = Point(self.w/2, self.h/2)
        self.snake = [self.head, 
                    Point(self.head.x-BLOCK_SIZE, self.head.y),
                    Point(self.head.x-(2*BLOCK_SIZE), self.head.y)]
                    # Point(self.head.x-(3*BLOCK_SIZE), self.head.y)]
        
        self.score = 0
        self.reward = 0
        self.game_over = False
        self.food = None
        self._place_food()
        self.frame_iteration = 0
        # Calculate Observation
        food_delta_x = self.head.x - self.food.x
        food_delta_y = self.head.y - self.food.y
        snake_length = len(self.snake)
        self.prev_actions = deque(maxlen=SNAKE_SIZE_GOAL)
        for _ in range(SNAKE_SIZE_GOAL):
            self.prev_actions.append(-1)

        self.observation = [self.head.x, self.head.y, food_delta_x, food_delta_y, snake_length] + list(self.prev_actions)
        self.observation=np.array(self.observation)
        return self.observation

    def get_observation(self):
        food_delta_x = self.head.x - self.food.x
        food_delta_y = self.head.y - self.food.y
        snake_length = len(self.snake)

        self.observation = [self.head.x, self.head.y, food_delta_x, food_delta_y, snake_length] + list(self.prev_actions)
        self.observation=np.array(self.observation)
        return self.observation

    def step(self, action):
        self.frame_iteration += 1
        # 1. collect user input
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                quit()
        
        # 2. move and get observation
        self._move(action) # update the head
        self.snake.insert(0, self.head)
        self.prev_actions.append(action) # Add action to deque for memory

        observation = self.get_observation()
        info={} # Not needed, but need to return to comply with Open AI Gym
        # print(observation)
        # 3. check if game over
        # if self.is_collision() or self.frame_iteration > 100*len(self.snake):
        if self.is_collision():
            self.game_over = True
            self.reward -= 1000
            return observation, self.reward, self.game_over, info
            
        # 4. eat and place new food or move
        if self.head == self.food:
            self.score += 1
            self.reward += 1000 * self.score # Add bonus for reaching food and increase over time
            self._place_food()
        else:
            euclidean_distance_to_food = np.linalg.norm(np.array(self.head) - np.array(self.food))
            self.reward += ((250-euclidean_distance_to_food)+1000 * self.score)/100
            # self.total_reward = ((250 - euclidean_distance_to_food) + apple_reward)/100
            self.snake.pop()

        # # Add a positive reward for each step taken without collision
        # self.reward = 1

        # 5. update ui and clock
        self._update_ui()
        self.clock.tick(self.SPEED)
        # 6. return game over and score
        return observation, self.reward, self.game_over, info

    def _place_food(self):
        x = random.randint(0, (self.w-BLOCK_SIZE )//BLOCK_SIZE )*BLOCK_SIZE 
        y = random.randint(0, (self.h-BLOCK_SIZE )//BLOCK_SIZE )*BLOCK_SIZE
        self.food = Point(x, y)
        if self.food in self.snake:
            self._place_food()

    def is_collision(self, point=None):
        if point is None:
            point = self.head
        # hits boundary
        if point.x > self.w - BLOCK_SIZE or point.x < 0 or point.y > self.h - BLOCK_SIZE or point.y < 0:
            return True
        # hits itself
        if point in self.snake[1:]:
            return True
        
        return False
        
    def _update_ui(self):
        self.display.fill(BLACK)
        
        for pt in self.snake:
            pygame.draw.rect(self.display, BLUE1, pygame.Rect(pt.x, pt.y, BLOCK_SIZE, BLOCK_SIZE))
            pygame.draw.rect(self.display, BLUE2, pygame.Rect(pt.x+4, pt.y+4, 12, 12))
            
        pygame.draw.rect(self.display, RED, pygame.Rect(self.food.x, self.food.y, BLOCK_SIZE, BLOCK_SIZE))
        
        text = font.render("Score: " + str(self.score), True, WHITE)
        self.display.blit(text, [0, 0])
        pygame.display.flip()

    def _update_game_over_ui(self):
        self.display.fill(BLACK)
        text = font.render("GAME OVER", True, WHITE)
        self.display.blit(text, [self.w//2 - text.get_width()//2, self.h//2 - text.get_height()//2])
        pygame.display.flip()
        
    def _move(self, action):
        # action = [straight, right, left]
        total_moves = [Direction.RIGHT, Direction.DOWN, Direction.LEFT, Direction.UP]
        index = total_moves.index(self.direction)
        if action==0:
            new_dir = total_moves[index] # no change
        elif action==1:
            next_index = (index + 1) % 4
            new_dir = total_moves[next_index] # right turn (r > d > l > u)
        else: # action==2:
            next_index = (index - 1) % 4
            new_dir = total_moves[next_index] # left turn (r > u > l > d)

        self.direction = new_dir

        x = self.head.x
        y = self.head.y
        if self.direction == Direction.RIGHT:
            x += BLOCK_SIZE
        elif self.direction == Direction.LEFT:
            x -= BLOCK_SIZE
        elif self.direction == Direction.DOWN:
            y += BLOCK_SIZE
        elif self.direction == Direction.UP:
            y -= BLOCK_SIZE
            
        self.head = Point(x, y)
            
if __name__ == '__main__':
    game = SnakeGameEnv()
    
    # game loop
    while True:
        obs, reward, game_over, info= game.step(game.action_space.sample())
        print("Reward: ", reward)
        if game_over == True:
            print('Final Score', game.score)
            game._update_game_over_ui()
            print("Press any key to restart... Esc to quit")
            key = keyboard.read_event().name
            if(key=='esc'):
                pygame.quit()
                quit()
            else: game.reset()