#!/usr/bin/env python
#Following file contains modified version of VizDoom's spectator.py program.

#Original disclaimer as follows:


#####################################################################
# This script presents SPECTATOR mode. In SPECTATOR mode you play and
# your agent can learn from it.
# Configuration is loaded from "../../examples/config/<SCENARIO_NAME>.cfg" file.
# 
# To see the scenario description go to "../../scenarios/README.md"
#####################################################################

from __future__ import print_function

import pickle
import sys
import termios
import tty
from time import sleep

from vizdoom import *

game = DoomGame()

scenario = "health_gathering"
output_file = "recorder_episode.pkl"
episodes = 3
if len(sys.argv) > 1:
    scenario = sys.argv[1]
if len(sys.argv) > 2:
    output_file = sys.argv[2]
if len(sys.argv) > 3:
    episodes = int(sys.argv[3])

if len(sys.argv) == 1:
    print("Usage: python [scenario, for example health_gathering or defend_the_center] [trajectories output file] [number of episodes to play, for example 3]")
    exit(0)

print("Running " + str(episodes) + " episodes of " + scenario + " with output to " + output_file)

game.load_config("config/" + scenario + ".cfg")

game.set_screen_resolution(ScreenResolution.RES_640X480)

# Enables spectator mode, so you can play. Sounds strange but it is the agent who is supposed to watch not you.
game.set_window_visible(True)
game.set_mode(Mode.SPECTATOR)
game.set_screen_format(ScreenFormat.GRAY8)
memory = []
game.init()




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
    if move == 'i':
        return [0, 0, 1]
    return [0, 0, 0]

for i in range(episodes):

    print("Episode #" + str(i + 1))

    game.new_episode()
    while not game.is_episode_finished():
        state = game.get_state()
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
with open(output_file, 'wb') as f:
    pickle.dump(memory, f, 2)
