from pynput import keyboard
import time
import numpy as np


LEFT, RIGHT, FORE, BACK, SHIFT, CTRL, QUIT \
    = 65295, 65296, 65297, 65298, 65306, 65307, ord('Q')


class Keyboard:
    def __init__(self):

        self.currently_pressed = set()

        self.listener = keyboard.Listener(
            on_press=self._on_press,
            on_release=self._on_release
        )

    def __enter__(self):
        self.listener.start()

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.listener.stop()

    def _on_press(self, key):
        self.currently_pressed.add(key)

    def _on_release(self, key):
        try:
            self.currently_pressed.remove(key)
        except KeyError:
            pass

    def get_state(self):

        pose = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])

        if keyboard.Key.ctrl in self.currently_pressed:
            ids = [4, 5]
        elif keyboard.Key.shift in self.currently_pressed:
            ids = [2, 3]
        else:
            ids = [0, 1]

        if keyboard.Key.up in self.currently_pressed:
            pose[ids[0]] = 1.0
        if keyboard.Key.down in self.currently_pressed:
            pose[ids[0]] = -1.0
        if keyboard.Key.right in self.currently_pressed:
            pose[ids[1]] = 1.0
        if keyboard.Key.left in self.currently_pressed:
            pose[ids[1]] = -1.0

        return np.array(pose)


if __name__ == '__main__':

    keyboard_listener = Keyboard()
    with keyboard_listener:
        for i in range(1000):
            print(keyboard_listener.get_state())
            time.sleep(0.01)
