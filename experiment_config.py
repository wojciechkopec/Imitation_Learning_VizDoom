__author__ = 'wojtek'
from sources.multi_q_estimator import MultiQEstimator
from sources.q_estimator import QEstimator
from sources.q_estimator_tf import QEstimator as tfQEstimator
from experiments_runner import ExperimentsRunner
from experiments_runner import run as run
from experiments_runner import play as play
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
agents['exampleTFAgent'] = lambda actions, config, dump_file_name: tfQEstimator(len(actions), config.resolution,
                                                                            dump_file_name=dump_file_name,store_trajectory=False)
agents['bdqnAgentK5p09'] = lambda actions, config, dump_file_name: MultiQEstimator(len(actions), config.resolution, 5,
                                                                                   0.9, False,
                                                                                   dump_file_name=dump_file_name)

chosenAgent = 'bdqnAgentK5p09'
# ExperimentsRunner(chosenAgent,ExperimentConfig(config_file_path="./config/defend_the_center.cfg"), agents[chosenAgent]).run()
# run(chosenAgent, ExperimentConfig(playAgent=True,epochs=1), 1, agents)
play(chosenAgent, ExperimentConfig(playAgent=True,config_file_path="./config/defend_the_center.cfg"), "out/", agents)


