__author__ = 'wojtek'
from sources.multi_q_estimator import MultiQEstimator
from sources.q_estimator import QEstimator
from experiments_runner import ExperimentsRunner
from experiments_runner import run as run
import os


class ExperimentConfig:
    def __init__(self, epochs=20, learning_steps_per_epoch=2000, test_episodes_per_epoch=100
                 , frame_repeat=12, resolution=(30, 45), config_file_path="./config/basic.cfg", playAgent=False):
        self.epochs = epochs
        self.learning_steps_per_epoch = learning_steps_per_epoch
        self.test_episodes_per_epoch = test_episodes_per_epoch
        self.frame_repeat = frame_repeat
        self.resolution = resolution
        self.config_file_path = config_file_path
        self.playAgent = playAgent

    def get_scenario(self):
        return os.path.basename(self.config_file_path).replace(".cfg", "")


agents = {}
agents['exampleAgent'] = lambda actions, config, dump_file_name: QEstimator(len(actions), config.resolution,
                                                                            dump_file_name=dump_file_name)
agents['bdqnAgentK5p09'] = lambda actions, config, dump_file_name: MultiQEstimator(len(actions), config.resolution, 5,
                                                                                   0.9, False,
                                                                                   dump_file_name=dump_file_name)

chosenAgent = 'exampleAgent'
# ExperimentsRunner(chosenAgent,ExperimentConfig(config_file_path="./config/defend_the_center.cfg"), agents[chosenAgent]).run()
run(chosenAgent,
    ExperimentConfig(config_file_path="./config/defend_the_center.cfg", epochs=2, learning_steps_per_epoch=200), 2,
    agents)

run(chosenAgent,
    ExperimentConfig(config_file_path="./config/basic.cfg", epochs=2, learning_steps_per_epoch=200), 2,
    agents)
