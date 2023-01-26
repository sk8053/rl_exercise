import glob
import gym, ray
from ray import air

from ray.rllib.algorithms import ppo
import numpy as np
from env import Env as MyEnv
from ray.rllib.agents.trainer import with_common_config
from ray import tune
from utils import read_file_list
import ray
import ray.rllib.agents.ppo as ppo
from ray.tune.logger import pretty_print
from ray.rllib.algorithms.td3 import TD3Config
config = TD3Config()
print(config)

# https://docs.ray.io/en/releases-1.11.0/rllib/rllib-training.html#common-parameters
file_list = read_file_list()


ray.init()
config = config.training(lr=0.0001, twin_q = True, tau = tune.grid_search(
    [0.005, 0.0005, 0.00005]))
# Set the config object's env.
config.environment(env=MyEnv, env_config= {'file_list':file_list, 'n_prev_t':20})
# Use to_dict() to get the old-style python config dict
# when running with tune.
'''
tune.Tuner(
    "TD3",
    run_config=air.RunConfig(stop={"episode_reward_mean": 500}),
    param_space=config.to_dict(),

).fit()

'''
tune.run(
    "TD3",
    stop={"episode_reward_mean": 600},

    config={
        "env_config": {'file_list':file_list, 'n_prev_t':20},
        "replay_buffer_config":{'capacity':10000,'type':'MultiAgentPrioritizedReplayBuffer'},
        "env": MyEnv,
        "num_gpus": 0,
        "num_workers": 1,
        "num_steps_sampled_before_learning_starts": 2000,
        "noisy": True,  # Whether to use noisy network
        "sigma0": 0.1,  # control the initial value of noisy nets
        "dueling": True,  # Use dueling network
        "double_q": True,  # Use double Q-learning
        "tau":tune.grid_search([0.001, 0.0001, 0.00001]),
        #"parameter_noise": True,
        "disable_env_checking":True,
        "sample_batch_size": 100,


        "lr": 0.0001, #
    },
)

'''
config = ppo.DEFAULT_CONFIG.copy()
config["num_gpus"] = 0
config["num_workers"] = 1
config['env_config']['file_list'] = file_list
config['env_config']['n_prev_t'] = 15
trainer = ppo.PPOTrainer(config=config, env=MyEnv)


for i in range(1000):
   # Perform one iteration of training the policy with PPO
   result = trainer.train()
   print(pretty_print(result))

   if i % 100 == 0:
       checkpoint = trainer.save()
       print("checkpoint saved at", checkpoint)
# Also, in case you have trained a model outside of ray/RLlib and have created
# an h5-file with weight values in it, e.g.
# my_keras_model_trained_outside_rllib.save_weights("model.h5")
# (see: https://keras.io/models/about-keras-models/)

# ... you can load the h5-weights into your Trainer's Policy's ModelV2
# (tf or torch) by doing:
trainer.import_model("my_weights.h5")
# NOTE: In order for this to work, your (custom) model needs to implement
# the `import_from_h5` method.
# See https://github.com/ray-project/ray/blob/master/rllib/tests/test_model_imports.py
# for detailed examples for tf- and torch trainers/models.

 "prioritized_replay": True,

        # Alpha parameter for prioritized replay buffer.
        "prioritized_replay_alpha": 0.6,
        # Beta parameter for sampling from prioritized replay buffer.
        "prioritized_replay_beta": 0.4,
        # Fraction of entire training period over which the beta parameter is
        # annealed
        "beta_annealing_fraction": 0.2,
        # Final value of beta
        "final_prioritized_replay_beta": 0.4,
        # Epsilon to add to the TD errors when updating priorities.
        "prioritized_replay_eps": 1e-6,
'''
ray.shutdown()

print("done!")
#while True:
#    print(algo.train())
#print(len(file_list))