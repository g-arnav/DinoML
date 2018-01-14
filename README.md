# DinoML
An implementation of deep learning to play the google chrome no internet dinosaur game.

![giphy](https://user-images.githubusercontent.com/31298849/34912125-9ee2cf30-f88d-11e7-8e19-3de9e1faf5c2.gif)

## Overview

DinoMl uses a convolutional neural network that takes an input of the game screen and outputs whether it should or should not jump at that specific moment. It is able to capture the screen, evaluate it, and output its calculations in real time to play the game.

## Requirements
* OS X
* Tensorflow (pip install tensorflow)
* Numpy      (pip install numpy)
* cv2        (pip install opencv-python)
* pynput     (pip install pynput)
* keyboard   (pip install keyboard)

## Runinng DinoML

1. Download and unpack the .zip for this repository.
2. Open terminal and cd into the folder where you saved it.
3. Open the dinosaur game by starting google chrome, turning off the wifi, and searching for something.
4. Run findArea.py in terminal.

```
$ python findArea.py
```
Move your google chrome window around until the game fits into the displayed window like so:
<img width="600" alt="holder" src="https://user-images.githubusercontent.com/31298849/34912316-62c0946e-f893-11e7-9b5d-3176f3dac43d.png">

5. Run runSupervised.py in terminal and then click onto your google chrome window.

```
$ python runSupervised.py
```
If the dinosuar is unable to jump over any cacti this may be because of a significant difference in frames per second between your system and mine, and you may have to retrain the network.

## Retraining DinoML
