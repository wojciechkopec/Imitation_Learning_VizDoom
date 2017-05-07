__author__ = 'wojtek'
from sources.q_estimator_tf import QEstimator as tfQEstimator
from sources.multi_q_estimator_tf import QEstimator as MultiTfQEstimator
from experiments_runner import ExperimentsRunner
from experiments_runner import run as run
from experiments_runner import play as play
import os


class ExperimentConfig:
    def __init__(self, epochs=20, learning_steps_per_epoch=2000, test_episodes_per_epoch=100
                 , frame_repeat=12, resolution=(30, 45), config_file_path="./config/simpler_basic.cfg",
                 play_agent=False, store_trajectory=False, explore_whole_episode=False, initial_eps=1, dest_eps=0.1):
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


agents = {}
agents['simpleTFAgent'] = lambda actions, config, dump_file_name: tfQEstimator(len(actions), config.resolution,
                                                                                dump_file_name=dump_file_name,
                                                                                store_trajectory=config.store_trajectory)


agents['bdqnAgentK5p1'] = lambda actions, config, dump_file_name: MultiTfQEstimator(len(actions), config.resolution,
                                                                                     subnets=5,
                                                                                     incl_prob=1,
                                                                                     dump_file_name=dump_file_name,
                                                                                     store_trajectory=config.store_trajectory)

agents['bdqnAgentK5p075'] = lambda actions, config, dump_file_name: MultiTfQEstimator(len(actions), config.resolution,
                                                                                     subnets=5,
                                                                                     incl_prob=0.75,
                                                                                     dump_file_name=dump_file_name,
                                                                                     store_trajectory=config.store_trajectory)


chosenAgent = 'simpleTFAgent'
# run(chosenAgent, ExperimentConfig(playAgent=True,config_file_path="./config/defend_the_center.cfg", epochs=20), 1, agents)
# run(chosenAgent, ExperimentConfig(playAgent=True,config_file_path="./config/deadly_corridor.cfg", epochs=20), 10, agents)
# run(chosenAgent, ExperimentConfig(store_trajectory=True, explore_whole_episode=True, play_agent=False, config_file_path="./config/basic.cfg", epochs=10), 5, agents)
# run('simpleTFAgent', ExperimentConfig(store_trajectory=False, explore_whole_episode=True, play_agent=False,
#                                       config_file_path="./config/health_gathering.cfg", epochs=10), 10, agents)
run('simpleTFAgent', ExperimentConfig(store_trajectory=True, explore_whole_episode=True, play_agent=True,
                                      config_file_path="./config/defend_the_center.cfg", epochs=1, initial_eps=0.1), 1,
    agents)
# run('bdqnAgentK5p1', ExperimentConfig(store_trajectory=False, explore_whole_episode=True, play_agent=False,
#                                       config_file_path="./config/health_gathering.cfg", epochs=10), 10, agents)
# run('bdqnAgentK5p1', ExperimentConfig(playAgent=False,config_file_path="./config/basic.cfg", epochs=20), 10, agents)
# run('bdqnAgentK5p075_store', ExperimentConfig(playAgent=False,config_file_path="./config/basic.cfg", epochs=20), 10, agents)
# run('bdqnAgentK5p1_store', ExperimentConfig(playAgent=False,config_file_path="./config/basic.cfg", epochs=20), 10, agents)
#play(chosenAgent, ExperimentConfig(playAgent=True), "out/", agents)


