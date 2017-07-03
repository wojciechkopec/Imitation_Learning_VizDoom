from __future__ import print_function
from pykeyboard import PyKeyboardEvent


class KeyMonitor(PyKeyboardEvent):
    def __init__(self, keys, keypress_handler):
        PyKeyboardEvent.__init__(self)
        self.keypress_handler = keypress_handler
        self.keys = set(keys)

    def tap(self, keycode, character, press):
        if character in self.keys:
            self.keypress_handler(character, press)
