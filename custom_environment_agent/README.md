# Custom Environment Agent

This project implements a reinforcement learning agent to play the Snake game. The agent is trained using Stable Baselines3 library with the Proximal Policy Optimization (PPO) algorithm. The goal is to teach the agent to navigate the game board and collect as much food as possible without hitting the boundaries or its own body.

## Game Environment

The Snake game environment (`snake_env.py`) provides an interface for interacting with the game. It contains the game logic, action space, and observation space. The environment is compatible with OpenAI Gym and can be easily integrated with RL algorithms.

### Observation Space

The observation space of this agent consists of the following elements:

- **Danger Straight**: A boolean value indicating whether there is an obstacle directly ahead of the snake's head in its current direction.
- **Danger Right**: A boolean value indicating whether there is an obstacle to the right of the snake's head.
- **Danger Left**: A boolean value indicating whether there is an obstacle to the left of the snake's head.
- **Move Direction**: A one-hot encoded vector representing the current direction of the snake. It has four elements:
  - `dir_l`: 1 if the snake is moving left, 0 otherwise.
  - `dir_r`: 1 if the snake is moving right, 0 otherwise.
  - `dir_u`: 1 if the snake is moving up, 0 otherwise.
  - `dir_d`: 1 if the snake is moving down, 0 otherwise.
- **Food Location**: A one-hot encoded vector representing the relative position of the food with respect to the snake's head. It has four elements:
  - `food_left`: 1 if the food is to the left of the snake's head, 0 otherwise.
  - `food_right`: 1 if the food is to the right of the snake's head, 0 otherwise.
  - `food_up`: 1 if the food is above the snake's head, 0 otherwise.
  - `food_down`: 1 if the food is below the snake's head, 0 otherwise.

The state space also includes the previous actions taken by the snake, encoded as a list of integers.

Overall, the state space is a vector representation of the game state, combining information about obstacles, movement direction, and food location. It provides the agent with the necessary information to make decisions and navigate the game board.

### Action Space

The action space represents the possible actions that the agent can take in the Snake environment. In this implementation, the action space consists of three actions:

- `Action 0`: Move straight - The snake continues moving in its current direction.
- `Action 1`: Turn right - The snake makes a 90-degree turn to the right from its current direction (right > down > left > up).
- `Action 2`: Turn left - The snake makes a 90-degree turn to the left from its current direction (right > up > left > down).

The agent selects an action from the action space, and it is passed to the `_move` function. Based on the selected action, the snake's direction and position are updated accordingly. The `_move` function determines the new direction and calculates the new position of the snake's head based on the current direction and the chosen action.

### Rewards

The agent receives rewards based on its actions and the resulting game state. The reward model is as follows:

- `+10`: When the agent collects food
- `-100`: When the agent hits the boundaries or its own body. Heavy punishment to discourage the snake from dying. Dying is bad!
- `-1/+1`: Calculates the Euclidean Distance from snake to food. +1 if this distance is closer to food than previous step and vice versa.

## Algorithms

The agent is trained using the Proximal Policy Optimization (PPO) algorithm, implemented in the Stable Baselines3 library. PPO is a policy optimization algorithm that uses a surrogate objective function to update the policy in an iterative manner. It strikes a balance between exploration and exploitation to find an optimal policy.

## Training Model

The training process is responsible for training the Snake AI agent using reinforcement learning. In this implementation, the training is performed using the Proximal Policy Optimization (PPO) algorithm.

To train the model, the `train_model` function is provided. This function takes an optional parameter `best_model`, which allows you to specify the name of a previously trained model checkpoint to load and resume training from. If no `best_model` is provided, the training will start from scratch.

The `train_model` function follows the following steps:

1. Setting up the callback: The training process uses a `TrainAndLoggingCallback` to monitor the training progress and save checkpoints. The `check_freq` parameter specifies the frequency at which the callback is triggered, and the `save_path` parameter determines the directory where the checkpoints will be saved.

2. Creating the environment: The Snake game environment, `SnakeGameEnv`, is created with a specified speed. This environment will be used for training the Snake AI agent.

3. Loading or initializing the model: If a `best_model` checkpoint is provided, the model is loaded using the `PPO.load` method (or `DQN.load` for DQN algorithm) from the specified checkpoint file. Otherwise, a new model is initialized using the `PPO` (or `DQN`) class, with the desired policy, environment, and other parameters such as the learning rate and tensorboard logging directory.

4. Starting or resuming the training: The model's `learn` method is called to begin the training process. The `total_timesteps` parameter determines the total number of timesteps the training will run for. During training, the model will interact with the environment, optimize its policy, and update its parameters based on the PPO algorithm.

The `train_model` function provides a convenient way to train the Snake AI agent and can be customized as needed. You can modify the hyperparameters, change the algorithm (e.g., PPO or DQN), or adjust the training settings based on your requirements.

### Examples

Here are some examples of the agent in action:

**Note**: The speed of the snake has been significantly reduced to for demo purposes. In real training, the speed is set to a much higher rate to speed up training.

#### This is what training looks like if started from scratch:

```bash
python .\agent.py train
```

![train_from_scratch](https://github.com/khanglam/SnakeGameAI/assets/7472121/9e4b5621-ec9c-4e1f-9de2-1a0a930f07df) ![step_info from scratch](https://github.com/khanglam/SnakeGameAI/assets/7472121/ba25dfef-c1ac-4cf5-a363-4767125a966f)

#### This is what training would look like if resumed from my pretrained model (~60+ million steps):

```bash
python .\agent.py train --best_model best_model
```

![train_from_best_model](https://github.com/khanglam/SnakeGameAI/assets/7472121/e4efb29d-7fbd-4a96-85db-b9f69a1f3719) ![step info from best_model](https://github.com/khanglam/SnakeGameAI/assets/7472121/98b2ba9b-55f2-4f86-862d-3f9d86cbbf8e)

**Highlight**: Notice the `ep_rew_mean` and `value_loss` difference between the two.

## Evaluate Model

The evaluation process allows you to test the performance of a trained Snake AI agent using a specified checkpoint. The `test_model` function is provided for this purpose.

To evaluate the model, follow these steps:

1. Specifying the best model: The `best_model` parameter is required, which should be the name of the checkpoint file containing the trained model.

2. Creating the environment: The Snake game environment, `SnakeGameEnv`, is created with a specified speed. This environment will be used for evaluating the Snake AI agent.

3. Initializing variables: Several variables are initialized to keep track of the average reward, average score, and the number of iterations to run the evaluation. You can customize these variables based on your evaluation requirements.

4. Loading the model: The trained model is loaded using the `PPO.load` method (or `DQN.load` for DQN algorithm) from the specified checkpoint file. This ensures that the evaluation is performed using the trained policy.

5. Running the evaluation: The evaluation process consists of several episodes. For each episode, the environment is reset, and the agent interacts with the environment using the learned policy. The agent's actions are determined by calling `model.predict` to get the action prediction based on the observation. The agent then takes the predicted action, and the environment updates accordingly. The episode continues until the termination condition is met.

6. Displaying the results: After each episode, the total reward and score achieved by the agent are printed. Additionally, the average reward and average score across all episodes are calculated and displayed at the end of the evaluation.

The `test_model` function provides a convenient way to evaluate the performance of a trained Snake AI agent. You can adjust the evaluation settings, such as the number of iterations, and analyze the agent's performance based on the displayed results.

Please note that this is a general explanation of the `test_model` function. If you have any specific details or additional information to include, please let me know, and I'll be happy to assist you further.

#### To evaluate the model, execute either of these command. It will execute the agent for `iterations` and take the average reward and score:

```bash
python .\agent.py evaluate
```

or

```bash
python .\agent.py evaluate --best_model best_model
```

![evaluate](https://github.com/khanglam/SnakeGameAI/assets/7472121/953ae439-2818-4b96-b8e2-e9596d8ca2b4)

In this case, after `5` iterations, calculate the averages:

![evaluate_info](https://github.com/khanglam/SnakeGameAI/assets/7472121/3de33996-736f-4aea-bac4-cc193e49c9f8)

## Tensorboard

Tensorboard is a powerful tool that allows you to visualize and analyze the training logs of the Snake AI agent. You can monitor various metrics, such as the training loss, rewards, and policies, to gain insights into the agent's learning progress.

To view the training logs in Tensorboard, follow these steps:

1. Open a command prompt or terminal.

2. Navigate to the directory that contains `logs` directory. In this case, `SnakeGameAI/custom_environment_agent`

3. Run the following command:

   ```shell
   tensorboard --logdir=logs/
   ```
This will start TensorBoard and you can access it in your web browser at `http://localhost:6006`. You will be able to see various training metrics and monitor the agent's learning progress.

## Requirements

- Python 3.x
- NumPy
- OpenAI Gym
- Stable Baselines3
- TensorBoard

## References

- [Stable Baselines3 Documentation](https://stable-baselines3.readthedocs.io/)
- [Proximal Policy Optimization Paper](https://arxiv.org/abs/1707.06347)
- [Snake Played by a Deep Reinforcement Learning Agent](https://towardsdatascience.com/snake-played-by-a-deep-reinforcement-learning-agent-53f2c4331d36)
