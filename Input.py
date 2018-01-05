from pynput.keyboard import Key
import time

def Up(keyboard):
    keyboard.press(Key.up)

def Wait(keyboard):
    keyboard.release(Key.up)
    keyboard.release(Key.down)

def Reset(keyboard):
    Up(keyboard)
    keyboard.release(Key.up)
    time.sleep(4)

def Reload(keyboard):
    keyboard.release(Key.up)
    keyboard.press(Key.cmd)
    keyboard.press('r')
    keyboard.release(Key.cmd)
    keyboard.release('r')
    time.sleep(1)
    Reset(keyboard)

def Act(keyboard, action):
    if action == 1:
        Up(keyboard)
    else:
        Wait(keyboard)