# SnakeGameAI

SnakeGameAI is a Python implementation of the classic Snake game with different types of AI agents that attempt to learn to play the game using reinforcement learning.
<p align="center">
  <img src="https://github.com/khanglam/SnakeGameAI/assets/7472121/8520dc8b-6486-4231-bf39-33311cafb378" alt="Snake Game AI Demo">
</p>

## Features

- Classic Snake game implementation
- Different types of AI agents for learning to play the game:
  - `custom_environment_agent`: Creates a custom environment using OpenAI's Gym
  - `ocr_agent`: Uses computer vision techniques with OpenCV and Tesseract for game analysis
  - `q_learning_agent`: Implements a Deep Q Learning Network from scratch
- Customizable game environment and agent parameters
- Training statistics and visualization with Tensorboard
- Pretrained models for resumed learnings/evaluations

The goal of this project is to explore various ways of implementing reinforcement learning and determine which approach yields the best convergence and performance in learning to play the Snake game.

## Dependencies

Make sure you have the following dependencies installed:

- Python (>= 3.6)
- Pygame (>= 2.0.1)
- Stable Baselines3 (>= 1.2.0)
- OpenAI Gym (>= 0.20.0)
- OpenCV (>= 4.5.1)
- Tesseract OCR (>= 4.1.1)

## Installation

1. Clone the SnakeGameAI repository:
   ```bash
   git clone https://github.com/khanglam/SnakeGameAI.git
   ```
2. Navigate to the SnakeGameAI directory:
   ```bash
    cd SnakeGameAI
   ```
3. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Custom Environment Agent

This method uses PPO (Proximal Policy Optimization) algorithm with MlpPolicy to train the agent.

Note: You can modify the script to use a different RL algorithm or adjust the training parameters.

Bonus: I've spent a lot of time training this agent (~60+ million steps), and committed this pretrained model under `custom_environment_agent/train` directory.

## OCR Agent

This method also uses PPO (Proximal Policy Optimization) but with CnnPolicy to train the agent.

This OCR agent combines OpenCV and Tesseract to capture the snake game frame by frame and passing it to Stable-Baseline3 to create its observational space.

## Q-Learning Agent

This script implements a Q-Learning Network from scratch. Purpose of this is to get a deeper understanding of a Reinforcement Learning architecture (Q-Net, Exploration vs Exploitation, etc...) and to explore ways of creating an agent without the help of stable-baseline3. Of course, this agent is most flexible of the three and you can definitely modify the agent architecture and adjust the training parameters in the code.

## Usage

Navigate to the agent you desire (custom, ocr, q-learning) and run the script with the following command and options. The available commands are:

- `train`: Trains the agent.
- `evaluate`: Evaluates the agent using a pre-trained model.
  This is the command format:

```bash
python agent.py <command> --best_model <filename>
```

- `command`: Required. Specify the command to execute (train or evaluate).
- `--best_model` <filename>: Optional. Specify the filename of the best model to use. If not provided, it will default to an empty string, meaning it will begin training from scratch.

Example usage:

Train the agent from scratch:

```bash
python agent.py train
```

Train the agent using a pre-trained model:

```bash
python agent.py train --best_model best_model
```

Evaluate the agent using the default model named "best_model":

```bash
python agent.py evaluate
```

Evaluate the agent using a specific best model:

```bash
python agent.py evaluate --best_model best_model
```

## Pretrained Models

The SnakeGameAI repository includes pretrained models that you can use to quickly start playing or continue training the Snake AI. The models are located in the `train` directory of each agent.

For both `custom_environment_agent` and `ocr_agent`, you will find that they will have `train` and `logs` folders.

- `train` - All models will be saved here. How frequent it saves will be determined by the parameter you set in the code.
- `logs` - All PPO/DQN logs will be saved here. You can use Tensorboard to open and analyze the performance of each training session.

For q_learning agent, you will only find `train` folder. The code uses matplotlib to help plot mean_score and visualize performance.

- `train` - Maintains all best_model.pth files. The model will get overridden if you decide to load and resume training. Otherwise if you start training from scratch, it will generate another best_model_2.pth etc... If you want to change this functionality, you can modify the code to your liking.

## Contributing

Please note that this project is not open for external contributions as it is intended for my individual learning and experimentation. You may however, feel free to clone this project to explore and modify the code as per your requirements.

If you encounter any issues, have suggestions for improvements, or want to discuss ideas related to this project, you are welcome to open an issue to initiate a discussion. However, please understand that there might be limited support available for external contributions.

Thank you for your understanding.

## Credits

This project was developed and maintained by [Khang Lam](https://github.com/khanglam/).

Special thanks to:

- [Patrick Loeber](https://www.youtube.com/@patloeber) for their informative YouTube videos on Python and AI, more specifically the assistance with understanding how to build Q_Learning agent.
- The contributors of the [Stable Baselines3](https://github.com/DLR-RM/stable-baselines3) library for providing a powerful reinforcement learning framework.

Thank you for their valuable contributions and resources!
