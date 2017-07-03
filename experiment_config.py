__author__ = 'wojtek'
from sources.q_estimator_tf import QEstimator as tfQEstimator
from sources.bootstrap_q_estimator_tf import QEstimator as MultiTfQEstimator
from sources.dropout_q_estimator_tf import QEstimator as DOTfQEstimator
from sources.actions_estimator_tf import ActionsEstimator
from experiments_runner import ExperimentsRunner
from experiments_runner import run as run
from experiments_runner import play as play
import os
import pickle


class ExpertConfig:
    def __init__(self, feed_memory=[], reward_for_another_action=-1):
        self.reward_for_another_action = reward_for_another_action
        self.feed_memory = feed_memory


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
        result['expert_config'] = []
        return result


# memory = ['recorder_episode_defend_the_center_4.pkl','recorder_episode_defend_the_center_3.pkl','recorder_episode_defend_the_center_2.pkl']
# memory = ['recorder_episode_health_gathering_couch_2_1.pkl']
memory = ['recorder_episode_health_gathering_2.pkl']
#
# with open('recorder_episode_defend_the_center_4.pkl', 'rb') as f:
#     memory = pickle.load(f)

# with open('recorder_episode_defend_the_center_3.pkl', 'rb') as f:
#     for frame in pickle.load(f):
#         memory.append(frame)

agents = {}
agents['simpleTFAgent'] = lambda actions, config, dump_file_name: tfQEstimator(len(actions), config.resolution,
                                                                               dump_file_name=dump_file_name,
                                                                               store_trajectory=config.store_trajectory)

agents['simpleActionsTFAgent'] = lambda actions, config, dump_file_name: ActionsEstimator(actions,
                                                                                          config.resolution,
                                                                                          config.expert_config,
                                                                                          dump_file_name=dump_file_name,
                                                                                          store_trajectory=config.store_trajectory)

agents['doTFAgent'] = lambda actions, config, dump_file_name: DOTfQEstimator(len(actions), config.resolution, calls=10, dropout=0.9,
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


chosenAgent = 'bdqnAgentK5p1'
# run(chosenAgent, ExperimentConfig(playAgent=True,config_file_path="./config/defend_the_center.cfg", epochs=20), 1, agents)
# run(chosenAgent, ExperimentConfig(playAgent=True,config_file_path="./config/deadly_corridor.cfg", epochs=20), 10, agents)
# run(chosenAgent, ExperimentConfig(store_trajectory=True, explore_whole_episode=True, play_agent=False, config_file_path="./config/basic.cfg", epochs=10), 5, agents)
# run('simpleTFAgent', ExperimentConfig(store_trajectory=False, explore_whole_episode=True, play_agent=False,
#                                       config_file_path="./config/health_gathering.cfg", epochs=10), 10, agents)
run('simpleActionsTFAgent',
    ExperimentConfig(store_trajectory=False, explore_whole_episode=True, play_agent=True, resolution=(90, 60),
                     config_file_path="./config/health_gathering.cfg", epochs=5, test_episodes_per_epoch=30,
                     initial_eps=0,
                     expert_config=ExpertConfig(memory, -0.01)), 1,
    agents)
# run('bdqnAgentK5p1', ExperimentConfig(store_trajectory=False, explore_whole_episode=True, play_agent=False,
#                                       config_file_path="./config/health_gathering.cfg", epochs=10), 10, agents)
# run('bdqnAgentK5p1', ExperimentConfig(playAgent=False,config_file_path="./config/basic.cfg", epochs=20), 10, agents)
# run('bdqnAgentK5p075_store', ExperimentConfig(playAgent=False,config_file_path="./config/basic.cfg", epochs=20), 10, agents)
# run('bdqnAgentK5p1_store', ExperimentConfig(playAgent=False,config_file_path="./config/basic.cfg", epochs=20), 10, agents)
#play(chosenAgent, ExperimentConfig(playAgent=True), "out/", agents)


