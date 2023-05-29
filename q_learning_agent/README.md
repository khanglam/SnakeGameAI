# Q Learning Agent

This repository contains a Q Learning agent implemented to play the Snake Game. The agent learns to navigate the game environment using the Q-learning algorithm and a deep neural network model.

## Game Environment

The Snake Game environment is a classic game where a snake moves around a grid, aiming to eat the food while avoiding collisions with the walls and its own body. The game provides a visual representation of the environment and allows the agent to interact with it.

### Observation Space

The agent observes the game environment through an 11-dimensional state vector. This state vector represents the relevant information necessary for the agent to make decisions. It includes the following components:

- Danger straight: Indicates whether there is a collision risk in the direction the snake is currently facing. This information helps the agent avoid running into walls or its own body.
- Danger right: Indicates whether there is a collision risk to the right of the snake's head. This information helps the agent avoid turning right into walls or its own body.
- Danger left: Indicates whether there is a collision risk to the left of the snake's head. This information helps the agent avoid turning left into walls or its own body.
- Move direction: Represents the current direction of the snake (left, right, up, down). This information allows the agent to maintain its heading and make informed decisions about turning directions.
- Food location: Indicates the relative position of the food in relation to the snake's head (left, right, up, down). This information helps the agent navigate towards the food.

This state representation allows the agent to perceive the game environment and make decisions based on the observed information.

### Action Space

The agent can choose from three possible actions at each time step:

1. Move straight: The snake continues moving in its current direction.
2. Turn left: The snake turns 90 degrees counterclockwise from its current direction.
3. Turn right: The snake turns 90 degrees clockwise from its current direction.

These actions determine the direction in which the snake moves in the next time step.

### Rewards

The agent receives rewards based on its actions and the current state of the game. The following reward scheme is used:

- Eating food: When the snake successfully eats the food, it receives a positive reward. This encourages the agent to actively seek and consume food.
- Collision with walls or body: If the snake collides with the walls or its own body, it receives a negative reward. This discourages the agent from making detrimental moves that lead to collisions.

The goal of the agent is to maximize its cumulative reward over time by learning to navigate the game environment effectively.

## Algorithms

The Q Learning agent employs the Q-learning algorithm, which is a model-free reinforcement learning algorithm. Q-learning learns an action-value function, known as the Q-function, to estimate the expected cumulative rewards for each state-action pair.

The Q-function is updated iteratively using the following formula:
Q(s, a) = Q(s, a) + α * (r + γ * max(Q(s', a')) - Q(s, a))

where:
- `Q(s, a)` is the estimated value (Q-value) of taking action `a` in state `s`.
- `α` is the learning rate, determining the weight of new information. It controls how much the agent should learn from each update.
- `r` is the received reward for taking action `a` in state `s`.
- `γ` is the discount factor, balancing immediate and future rewards. It determines the importance of future rewards in the agent's decision-making process.
- `s'` is the next state.
- `a'` is the action that maximizes the Q-value for the next state.

By updating the Q-values based on observed rewards and transitioning to new states, the agent gradually learns to make better decisions and improve its performance in the game environment.

## Training Model

The training process consists of the following steps:

1. Initialize the Q-learning agent with the required hyperparameters, such as the maximum memory capacity, batch size, learning rate, discount factor, and exploration rate.
2. Create a Snake Game environment for the agent to interact with.
3. Repeat the following steps for multiple game episodes:
   1. Get the current state of the game environment.
   2. Choose an action based on the current state using an exploration-exploitation strategy.
   3. Perform the chosen action in the game environment.
   4. Receive the reward for the action and observe the new state.
   5. Store the experience (state, action, reward, next state) in the agent's memory.
   6. Update the agent's Q-function by sampling a mini-batch of experiences from the memory and performing a Q-learning update step.
   7. Repeat the above steps until the game episode is completed.
4. After each episode, evaluate the agent's performance by measuring the score achieved in the game.
5. Keep track of the highest score achieved and save the model if a new high score is reached.
6. Repeat the training process for a sufficient number of episodes to allow the agent to learn and improve its performance.

## Evaluate Model

After training the Q Learning agent, you can evaluate its performance by running the trained model on the Snake Game environment. The evaluation process is similar to the training process, but without updating the Q-values.

1. Load the trained model.
2. Create a Snake Game environment.
3. Repeat the following steps for multiple game episodes:
   1. Get the current state of the game environment.
   2. Use the trained model to select the best action based on the current state.
   3. Perform the chosen action in the game environment.
   4. Receive the reward for the action and observe the new state.
   5. Repeat the above steps until the game episode is completed.
4. Measure the agent's performance by calculating the score achieved in each episode.
5. Analyze the performance and compare it with the training results.

## Requirements

The following dependencies are required to run the Q Learning agent:

- Python (>=3.6)
- PyTorch (>=1.9.0)
- NumPy (>=1.19.5)
- Matplotlib (>=3.4.3)

You can install the required packages by running the following command:

pip install -r requirements.txt


## References

Here are some resources that were helpful in the development of this Q Learning agent:

- [Sutton, R. S., & Barto, A. G. (2018). Reinforcement Learning: An Introduction](https://web.stanford.edu/class/psych209/Readings/SuttonBartoIPRLBook2ndEd.pdf)
- [PyTorch Documentation](https://pytorch.org/docs/stable/index.html)
- [NumPy Documentation](https://numpy.org/doc/)
- [Matplotlib Documentation](https://matplotlib.org/stable/contents.html)

Feel free to explore these references to gain a deeper understanding of the Q-learning algorithm and its implementation in this Snake Game agent.

