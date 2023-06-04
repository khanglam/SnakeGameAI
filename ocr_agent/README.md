# OCR Agent for SnakeGameAI

This repository contains the OCR Agent for the SnakeGameAI project. The OCR Agent utilizes Optical Character Recognition (OCR) techniques to capture frame-by-frame gameplay of the classic Snake game and train an AI agent based on visual information.

## Game Environment

The OCR Agent operates in the Snake game environment, where the goal is to control the snake to eat food and avoid colliding with walls or its own body. The game is played on a grid-based board.

### Observation Space

The observation space for the OCR Agent includes the current state of the game board as an image captured from the gameplay. The OCR Agent uses OCR techniques to extract relevant information such as the snake's position, food location, and obstacles from the game board image. The observation space is captured via `mss` screenshots. It then gets processed, resized and grayscaled for faster image processing. When working with image processing, it's important to keep in mind that some information is inevitably lost during the image compression process. However, this trade-off allows us to perform computations more quickly, which is crucial for RL.

```shell
 def get_observation(self):
    raw = np.array(self.screenshot.grab(self.game_location))[:,:,:3].astype(np.uint8) # reshape to get 3 channels instead of 4. mss produces 4 channels for some reason. alpha channel?
    # Grayscale
    gray = cv2.cvtColor(raw, cv2.COLOR_BGR2GRAY)
    # Resize - Compress the 640x480 down to 83x100
    resized = cv2.resize(gray, (100,83))
    # Add channels first to suit stable_baseline format
    channel = np.reshape(resized, (1,83,100))
    return channel
```

#### What we see vs. What the agent sees (frame by frame):

<img src="https://github.com/khanglam/SnakeGameAI/assets/7472121/8c343510-5eda-4232-a854-8fba8e246626" alt="What we see" width="500"> <img src="https://github.com/khanglam/SnakeGameAI/assets/7472121/e06cc32e-7bd0-49a4-8116-f1344deb541a" alt="What Computer Vision Sees" width="500">

### Reset

For this Agent, initially, to reset the `observation_space`, I've implemented the "GAME OVER" text to display after each collision. Then using OCR (Optical Character Recognition - Tesseract) to extract the digits and determine the `done_state` like so:

<p align="center">
  <img src="https://github.com/khanglam/SnakeGameAI/assets/7472121/0e4a99c7-da54-4f1c-b5d1-04e02dc68266" alt="Game Over">
</p>

When called, OCR will extract the done_location and specifically look for 'GAME' or 'GAHE'

```shell
done_strings = ['GAME', 'GAHE']
```

Even though in theory this would work, I find that this is unnecessarily causing dramatic slowdowns in training. So I've resorted to reusing the same reset function as `custom_environment_agent` and `q_learning_agent`, which is to re-initiate the board everytime the snake dies. This means that we can now remove OCR entirely and this algorithm would still work, depending on which reset function you'd like to use.

### Action Space

The action space consists of a set of discrete actions that the OCR Agent can take in the game. These actions include moving the snake in different directions: up, down, left, or right. The agent makes this action by simulating key presses with `pydirectinput`, based on what it sees. The caveat of this method is that, the user must maintain window focus on Snake Game during its entire training, otherwise the agent will not be able to perform any key presses on the game, causing the snake to run straight in whichever direction it was left at.

### Rewards

The OCR Agent receives rewards based on its performance in the game. It receives a positive reward when it successfully eats food and a negative reward when it collides with walls or its own body. The goal of the OCR Agent is to maximize its cumulative reward over time.

- `+1`: When the agent collects food
- `-1`: When the agent hits the boundaries or its own body. Heavy punishment to discourage the snake from dying. Dying is bad!

## Model Architecture

The agent is also trained using the Proximal Policy Optimization (PPO) algorithm, but with CnnPolicy (Convolutional Neural Network Policy). To learn more about this RL algorithm, I've attached the source to read more on it in **Reference** section.

The model is trained using the captured game board processed images by frame as input and the corresponding actions and rewards as the target. The architecture of the model can be customized based on the specific requirements of the game environment.

## Training Process

The OCR Agent is trained using a combination of OCR techniques, image processing, and reinforcement learning algorithms. The agent learns to make decisions based on the captured game board images and the extracted information. The training process involves iteratively playing the game, capturing frames, extracting features using OCR, and updating the model based on the observed rewards.

During the training process, the OCR Agent interacts with the game environment, captures frames, extracts features using OCR, and updates its model based on the observed rewards. The training process continues iteratively until the OCR Agent achieves a satisfactory level of performance.

## Evaluation

To evaluate the performance of the OCR Agent, it can be tested in the game environment using unseen game scenarios. The agent's ability to capture frames accurately, extract relevant information using OCR, and make intelligent decisions can be assessed. The evaluation can be based on metrics such as the agent's average score, the number of successful food captures, and its ability to avoid collisions.

#### Drawback:

Since the `observation_space` is an image, the stored models take up dramatically more storage compared to the other two agents. Learning also takes much longer to converge and I concluded that this is not ideal for something as simple as the Snake Game.

## Tensorboard

Tensorboard is a powerful tool that allows you to visualize and analyze the training logs of the Snake AI agent. You can monitor various metrics, such as the training loss, rewards, and policies, to gain insights into the agent's learning progress.

To view the training logs in Tensorboard, follow these steps:

1. Open a command prompt or terminal.

2. Navigate to the directory that contains `logs` directory. In this case, `SnakeGameAI/ocr_agent`

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
