# DinoML
An implementation of deep learning to play the google chrome no internet dinosaur game.

![giphy](https://user-images.githubusercontent.com/31298849/34912125-9ee2cf30-f88d-11e7-8e19-3de9e1faf5c2.gif)

## Overview

DinoMl uses a convolutional neural network that takes an input of the game screen and outputs whether it should or should not jump at that specific moment. It is able to capture the screen, evaluate it, and output its calculations in real time to play the game.

## Requirements

* OS X
* python 2.6 or 2.7
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

>Move your google chrome window around until the game fits into the displayed window like so:
<img width="600" alt="holder" src="https://user-images.githubusercontent.com/31298849/34912316-62c0946e-f893-11e7-9b5d-3176f3dac43d.png">

5. Run runSupervised.py in terminal and then click onto your google chrome window.

```
$ python runSupervised.py
```

>If the dinosuar is unable to jump over any cacti this may be because of a significant difference in frames per second between your system and mine, and you may have to retrain the network.

## Retraining DinoML

1. Make sure the program is looking at the correct area of the screen using findArea.py

```
$ python findArea.py
```
2. Run GetData.py with the number of games you have played and quickly start your game. Make sure to only use the up arrow key (no down or spacebar). If you lose before the night time or start very late, quickly stop the program, restart, and use the same game number again. You may have to run with sudo if you see an error right before it starts capturing data.

```
$ sudo python GetData.py 1
```
>I recommend capturing at least 10 games so there is enough data for the network to learn how to play without overfitting to the data set. Also, be certain not to skip any numbers as this will cause errors when you go to train the model.

3. Train the network by running trainSupervised.py

```
$ python trainSupervised.py
```
>This will take a while so I recommend allowing this to run overnight or for at least a couple hours depending on the speed of your computer. When you are satisfied with the accuracy (at least 95% of it will not work very well), hit Ctr + C and wait for a couple of seconds to allow the model to save. Make sure not to use Ctr + Z as this will delete the model.

4. Run the model using runSupervised.py

```
$ python runSupervised.py
```
>If you are not satisfied with its performance, you can either: retrain the netowrk entirely, add more games using GetData.py, or let the model train for longer.

## Next Steps

There are a lot of improvements that can be made to improve this model. Some easy ones could be gathering more training data and allowing it to train for longer. Also, a time input could be added to the network that is not considered in the convolutions so it can adapt to how the game gradually increases in speed. This would remedy how the bot always jumps early at first and, as the game goes on, gets slower and slower at jumping until it loses. 

The architecture of the neural network could also be changed to be deeper and have larger hidden layers. However, I do not think this will lead to much of an improvement since it is only looking for basic shapes and does not even necessarily have to dirrentiate between them, only to recognize their position. Also, it would not need many more fully connected layers since again, the relation is pretty direct between the position of the shapes in the game and the decision of jumping.

Another possibillity that I would be excited to try would be using methods other than supervised learning. This could be unsupervised learning through the use of Reinforcement Learning and DQN. This method would likely have to run for much longer in order to become comparable to supervised learning, but I beleive it could develop much better tactics for playing the game and become noticeably better than an average human player if given enough time. Another option could be to use a genetic algorithm. While this is not exactly machine learning, I beleive this method would still work just as well if not better. Its progression would likely be similar to that of Reinforcement Learning. This would probably take much longer, however, since it requires testing every randomly generating network on the game in realtime. This process could be sped up heavily if each run was simulated instead of run in realtime.
