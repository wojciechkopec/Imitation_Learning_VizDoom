#!/usr/bin/env python

#####################################################################
# This script presents SPECTATOR mode. In SPECTATOR mode you play and
# your agent can learn from it.
# Configuration is loaded from "../../examples/config/<SCENARIO_NAME>.cfg" file.
# 
# To see the scenario description go to "../../scenarios/README.md"
#####################################################################

from __future__ import print_function

from time import sleep
from vizdoom import *
import pickle
import sys, tty, termios
from random import randint

game = DoomGame()

# Choose scenario config file you wish to watch.
# Don't load two configs cause the second will overrite the first one.
# Multiple config files are ok but combining these ones doesn't make much sense.

# game.load_config("../../scenarios/basic.cfg")
# game.load_config("../../scenarios/simpler_basic.cfg")
# game.load_config("../../scenarios/rocket_basic.cfg")
# game.load_config("config/deadly_corridor.cfg")
#game.load_config("../../scenarios/deathmatch.cfg")
game.load_config("config/defend_the_center.cfg")
# game.load_config("config/cig.cfg")
#game.load_config("../../scenarios/defend_the_line.cfg")
#game.load_config("../../scenarios/health_gathering.cfg")
# game.load_config("../../scenarios/my_way_home.cfg")
#game.load_config("../../scenarios/predict_position.cfg")
#game.load_config("../../scenarios/take_cover.cfg")


# Enables freelook in engine
# game.add_game_args("+freelook 1")

game.set_screen_resolution(ScreenResolution.RES_640X480)

# Enables spectator mode, so you can play. Sounds strange but it is the agent who is supposed to watch not you.
game.set_window_visible(True)
game.set_mode(Mode.PLAYER)
game.set_screen_format(ScreenFormat.GRAY8)
memory = []
game.init()

episodes = 3


def get_action():
    fd = sys.stdin.fileno()
    old_settings = termios.tcgetattr(fd)
    try:
        tty.setraw(sys.stdin.fileno())
        move = sys.stdin.read(1)
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)
    if move == 'j':
        return [1, 0, 0]
    if move == 'l':
        return [0, 1, 0]
    if move == 'a':
        return [0, 0, 1]
    return [0, 0, 0]

for i in range(episodes):

    print("Episode #" + str(i + 1))

    game.new_episode()
    while not game.is_episode_finished():
        state = game.get_state()

        game.set_action(get_action())
        for i in range(4):
            game.advance_action()
        next_state = game.get_state()
        last_action = game.get_last_action()
        reward = game.get_last_reward()
        isterminal = game.is_episode_finished()

        print("State #" + str(state.number))
        print("Game variables: ", state.game_variables)
        print("Action:", last_action)
        print("Reward:", reward)
        print("=====================")
        memory.append((state.screen_buffer,last_action, next_state.screen_buffer, reward, isterminal))

    print("Episode finished!")
    print("Total reward:", game.get_total_reward())
    print("************************")
    sleep(2.0)

game.close()
with open('recorder_episode.pkl', 'wb') as f:
    pickle.dump(memory, f, 2)
