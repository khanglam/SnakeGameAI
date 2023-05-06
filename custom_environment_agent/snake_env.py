import pygame
import random
from enum import Enum
from collections import namedtuple
import keyboard
import numpy as np
# Sending commands
import pydirectinput
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
SPEED = 20
SNAKE_SIZE_GOAL=30
class SnakeGameEnv(Env):
    
    def __init__(self, w=640, h=480):
        super().__init__()
        self.observation_space = Box(low=0, high=w, shape=(5+SNAKE_SIZE_GOAL,), dtype=np.float32)
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
                    Point(self.head.x-(2*BLOCK_SIZE), self.head.y),
                    Point(self.head.x-(3*BLOCK_SIZE), self.head.y)]
        
        self.score = 0
        self.food = None
        self._place_food()
        self.frame_iteration = 0


    def get_observation(self):
        food_delta_x = self.head.x - self.food.x
        food_delta_y = self.head.y - self.food.y

        snake_length = len(self.snake)   
        self.prev_actions = deque(maxlen=SNAKE_SIZE_GOAL)
        for _ in range(SNAKE_SIZE_GOAL):
            self.prev_actions.append(-1)
        self.observation_space = [self.head.x, self.head.y, food_delta_x, food_delta_y, snake_length] + list(self.prev_actions)
        self.observation_space=np.array(self.observation_space) 
        return self.observation_space

    def _place_food(self):
        x = random.randint(0, (self.w-BLOCK_SIZE )//BLOCK_SIZE )*BLOCK_SIZE 
        y = random.randint(0, (self.h-BLOCK_SIZE )//BLOCK_SIZE )*BLOCK_SIZE
        self.food = Point(x, y)
        if self.food in self.snake:
            self._place_food()
        
    def play_step(self, action):
        self.frame_iteration += 1
        # 1. collect user input
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                quit()
        
        # 2. move
        self._move(action) # update the head
        self.snake.insert(0, self.head)
        
        # 3. check if game over
        reward = 0
        game_over = False
        if self.is_collision() or self.frame_iteration > 100*len(self.snake):
            game_over = True
            reward = -10
            return reward, game_over, self.score
            
        # 4. place new food or just move
        if self.head == self.food:
            self.score += 1
            reward = 10
            self._place_food()
        else:
            self.snake.pop()
        
        # 5. update ui and clock
        self._update_ui()
        self.clock.tick(SPEED)
        # 6. return game over and score
        return reward, game_over, self.score
    
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
        # [straight, right, left]
        clock_wise = [Direction.RIGHT, Direction.DOWN, Direction.LEFT, Direction.UP]
        index = clock_wise.index(self.direction)
        if np.array_equal(action, [1,0,0]):
            new_dir = clock_wise[index] # no change
        elif np.array_equal(action, [0,1,0]):
            next_index = (index + 1) % 4
            new_dir = clock_wise[next_index] # right turn (r > d > l > u)
        else: # [0,0,1]
            next_index = (index - 1) % 4
            new_dir = clock_wise[next_index] # left turn (r > u > l > d)

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
    game = SnakeGameAI()
    
    # game loop
    while True:
        reward, game_over, score = game.play_step()
        if game_over == True:
            print('Final Score', score)
            game._update_game_over_ui()
            print("Press any key to restart... Esc to quit")
            key = keyboard.read_event().name
            if(key=='esc'):
                pygame.quit()
                quit()
            else: game.reset()