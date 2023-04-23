# Teach AI To Play Snake! Reinforcement Learning With PyTorch and Pygame

In this Python Reinforcement Learning Tutorial series we teach an AI to play Snake! We build everything from scratch using Pygame and PyTorch. The tutorial consists of 4 parts:

You can find all tutorials on my channel: [Playlist](https://www.youtube.com/playlist?list=PLqnslRFeH2UrDh7vUmJ60YrmWd64mTTKV)

- Part 1: I'll show you the project and teach you some basics about Reinforcement Learning and Deep Q Learning.
- Part 2: Learn how to setup the environment and implement the Snake game.
- Part 3: Implement the agent that controls the game.
- Part 4: Implement the neural network to predict the moves and train it.

Libraries Used:
pip install mss pydirectinput pytesseract
mss: This library provides an easy way to capture screenshots and record screen activities in Python. It's a cross-platform library that can be used on Windows, macOS, and Linux.

pydirectinput: This library provides an easy way to simulate keyboard and mouse events in Python. It's a cross-platform library that can be used on Windows, macOS, and Linux.

pytesseract: This library provides an easy way to perform Optical Character Recognition (OCR) on images using Python. It's a wrapper for Google's Tesseract-OCR Engine, which is an open-source OCR engine that can recognize text in over 100 languages.

pip install stable-baselines3[extra] protobuf==3.20.\*
stable-baselines3 is a set of high-quality implementations of reinforcement learning algorithms in Python, built on top of the TensorFlow 2 library. It provides an easy-to-use interface for creating and training agents that can learn to solve complex tasks by interacting with their environment.

The extra part of the command installs some optional dependencies that may be needed for certain features or algorithms provided by stable-baselines3.

protobuf is a Python implementation of Google's Protocol Buffers, a language-agnostic data serialization format. It is used in stable-baselines3 to encode and decode messages that are exchanged between the different components of a reinforcement learning algorithm during training and evaluation. The version 3.20._ specifies a range of compatible versions to be installed, where the third digit _ indicates any patch version.
