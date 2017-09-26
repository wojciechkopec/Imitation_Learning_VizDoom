__author__ = 'Wojciech Kopec'
import os
import sys
from os import listdir
from os.path import isfile, join

from experiments_runner import run as run
from sources.actions_estimator_tf import ActionsEstimator
from sources.q_estimator_tf import QEstimator as tfQEstimator


class ExpertConfig:
    def __init__(self, feed_memory=[], reward_for_another_action=-1, frames_limit=sys.maxint, switch_expert_mode='expert_call'):
        self.reward_for_another_action = reward_for_another_action
        self.feed_memory = feed_memory
        self.frames_limit = frames_limit
        self.switch_expert_mode = switch_expert_mode


class ExperimentConfig:
    def __init__(self, epochs=20, learning_steps_per_epoch=2000, test_episodes_per_epoch=100
                 , frame_repeat=2, resolution=(30, 45), config_file_path="./config/simpler_basic.cfg",
                 play_agent=False, store_trajectory=False, explore_whole_episode=False, initial_eps=1, dest_eps=0.1, expert_config = None):
        self.expert_config = expert_config
        self.dest_eps = dest_eps
        self.initial_eps = initial_eps
        self.explore_whole_episode = explore_whole_episode
        self.store_trajectory = store_trajectory
        self.epochs = epochs
        self.learning_steps_per_epoch = learning_steps_per_epoch
        self.test_episodes_per_epoch = test_episodes_per_epoch
        self.frame_repeat = frame_repeat
        self.resolution = resolution
        self.config_file_path = config_file_path
        self.play_agent = play_agent

    def get_scenario(self):
        return os.path.basename(self.config_file_path).replace(".cfg", "")

    def jsonable(self):
        result = self.__dict__.copy()
        result['expert_config'] = result['expert_config'].__dict__.copy()
        return result




agents = {}
agents['simpleTFAgent'] = lambda actions, config, dump_file_name: tfQEstimator(len(actions), config.resolution,
                                                                               dump_file_name=dump_file_name,
                                                                               store_trajectory=config.store_trajectory)

agents['simpleActionsTFAgent'] = lambda actions, config, dump_file_name: ActionsEstimator(actions,
                                                                                          config.resolution,
                                                                                          config.expert_config,
                                                                                          dump_file_name=dump_file_name,
                                                                                          store_trajectory=config.store_trajectory)

map_config = "health_gathering"
trajectories = "presenting_expert_trajectories/*"
epochs = 0
frames_limit = 6000
if len(sys.argv) == 1:
    print "Usage: python [scenario, for example health_gathering or defend_the_center] [learning frames limit, for example 3000 or 12000] [directory with trajectories or list of files with trajectories created with spectator.py]"
    exit(0)
if len(sys.argv) > 1:
    map_config = sys.argv[1]
if len(sys.argv) > 2:
    frames_limit = int(sys.argv[2])
if len(sys.argv) > 3:
    trajectories = sys.argv[3:]

if len(trajectories) == 1 and os.path.isdir(trajectories[0]):
    trajectories = [trajectories[0] + os.sep + f for f in listdir(trajectories[0]) if isfile(join(trajectories[0], f))]
memory = trajectories
print "Running agent with " + str(epochs) + " epochs of DAgger on " + map_config + " map, taught on " + str(
    frames_limit) + " frames from " + str(memory)

run('simpleActionsTFAgent',
    ExperimentConfig(store_trajectory=False, explore_whole_episode=True, play_agent=True, resolution=(106, 80),
                     config_file_path="./config/" + map_config + ".cfg", epochs=epochs, test_episodes_per_epoch=20,
                     initial_eps=0, expert_config=ExpertConfig(memory, -0.01, frames_limit=frames_limit)), 10,
    agents)
