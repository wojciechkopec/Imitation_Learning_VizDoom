__author__ = 'wojtek'
from sources.multi_q_estimator import MultiQEstimator
from sources.q_estimator import QEstimator
from experiments_runner import ExperimentsRunner


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

agents = {}
agents['exampleAgent'] = lambda actions, config: QEstimator(len(actions), config.resolution)
agents['bdqnAgentK5p09'] = lambda actions, config: MultiQEstimator(len(actions), config.resolution, 5, 0.9, False)

chosenAgent = 'exampleAgent'
ExperimentsRunner(chosenAgent,ExperimentConfig(epochs=1,learning_steps_per_epoch=200), agents[chosenAgent]).run()

