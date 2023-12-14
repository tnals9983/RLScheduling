filename = 'RLScheduling'
from gym.spaces import Box, Dict, Discrete


from ray.tune.suggest.hyperopt import HyperOptSearch
from ray.tune.suggest.ax import AxSearch
from ray.tune.schedulers import AsyncHyperBandScheduler
from ray.tune.stopper import MaximumIterationStopper

import tensorflow as tf
from gym import spaces
from or_gym.utils import create_env

from ray.rllib.models.tf.fcnet import FullyConnectedNetwork
from ray.rllib.models import ModelCatalog
from ray.rllib.models.tf.tf_modelv2 import TFModelV2
from ray.rllib.models.torch.torch_modelv2 import TorchModelV2
from ray.rllib.models.torch.fcnet import FullyConnectedNetwork as TorchFC
from ray.rllib.utils.framework import try_import_tf, try_import_torch
# from ray.rllib.utils.torch_utils import FLOAT_MIN
import time
tf1, tf, tfv = try_import_tf()
torch, nn = try_import_torch()
from ray.rllib.agents.dqn.distributional_q_tf_model import DistributionalQTFModel
import gym
# Ray rllib
import ray
from ray.rllib import agents
from ray import tune
from ray.tune import Stopper
# Custom Environment (\
from scheduling_env import SingleStage as env
from or_gym.utils import create_env

start = time.time()

class ActionMaskModel(TFModelV2):
    """Model that handles simple discrete action masking.
    This assumes the outputs are logits for a single Categorical action dist.
    Getting this to work with a more complex output (e.g., if the action space
    is a tuple of several distributions) is also possible but left as an
    exercise to the reader.
    """

    def __init__(
        self, obs_space, action_space, num_outputs, model_config, name, **kwargs
    ):
        orig_space = getattr(obs_space, "original_space", obs_space)
        assert isinstance(orig_space, Dict) and \
               "action_mask" in orig_space.spaces and \
               "observations" in orig_space.spaces

        super().__init__(obs_space, action_space, num_outputs, model_config, name)

        self.internal_model = FullyConnectedNetwork(
            orig_space["observations"],
            action_space,
            num_outputs,
            model_config,
            name + "_internal",
        )

        # disable action masking --> will likely lead to invalid actions
        # self.no_masking = model_config["custom_model_config"].get("no_masking", False)

    def forward(self, input_dict, state, seq_lens):
        # Extract the available actions tensor from the observation.
        action_mask = input_dict["obs"]["action_mask"]

        # Compute the unmasked logits.
        logits, _ = self.internal_model({"obs": input_dict["obs"]["observations"]})

        # If action masking is disabled, directly return unmasked logits
        # if self.no_masking:
        #     return logits, state

        # Convert action_mask into a [0.0 || -inf]-type mask.
        inf_mask = tf.maximum(tf.math.log(action_mask), tf.float32.min)
        masked_logits = logits + inf_mask

        # Return masked logits.
        return masked_logits, state

    def value_function(self):
        return self.internal_model.value_function()


ModelCatalog.register_custom_model('schedule_mask', ActionMaskModel)


def register_env(env_name, env_config={}):
    # env = create_env(env_name)
    tune.register_env(env_name,
                      lambda env_name: env(env_name, env_config=env_config))

env_name = 'SingleStage'
env_config = {}

trainer_config = dict(
        env=env_name,
        lr=2.3138816562230318e-06, #tune.loguniform(1e-6, 1e-4),
        num_workers=0,
        clip_param = 0.3,#tune.choice([0.1,0.2,0.3]),
        env_config=env_config,
        num_sgd_iter = 25,
        gamma = 1,
        model = dict(custom_model = 'schedule_mask'
                    ),
        exploration_config = dict(
            type = 'Curiosity',
            eta = 1.0,
            lr = 1.2432503232651561e-05,#tune.loguniform(1e-5,1e-3),
            feature_net_config = dict(fcnet_hiddens = [], fcnet_activation = 'relu'),
            beta = 0.2,#tune.choice([0.1,0.2,0.3]),
            sub_exploration = dict(type = "StochasticSampling")
        )

        )


analysis = tune.run('PPO',
                    config=trainer_config,
                    verbose=1,
                    local_dir=f'~/ray_results/{filename}',
                    metric = 'episode_reward_mean',
                    checkpoint_freq=5,

                    )


ray.shutdown()



