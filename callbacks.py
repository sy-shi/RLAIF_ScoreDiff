from typing import Dict

import os
import shutil
from ray.rllib.algorithms.callbacks import DefaultCallbacks
from ray.tune.callback import Callback
from ray.rllib.env import BaseEnv
from ray.rllib.evaluation import Episode, RolloutWorker
from ray.rllib.policy import Policy
from ray.rllib.policy.sample_batch import DEFAULT_POLICY_ID, SampleBatch


class CustomCheckpointCallback(Callback):
    def __init__(self, checkpoint_dir):
        self.checkpoint_dir = checkpoint_dir

    def on_checkpoint(self, iteration, trials, trial, checkpoint, **info):
        # Checkpoint is a Checkpoint object which contains metadata
        checkpoint_path = checkpoint.dir_or_data
        if checkpoint_path:
            # Construct new checkpoint path
            new_checkpoint_path = os.path.join(self.checkpoint_dir, os.path.basename(checkpoint_path))
            # Copy or move the checkpoint to the new location
            shutil.copytree(checkpoint_path, new_checkpoint_path)
            print(f"Checkpoint saved to {new_checkpoint_path}")


class LoggingCallbacks(DefaultCallbacks):
    def on_episode_end(
        self,
        *,
        worker: RolloutWorker,
        base_env: BaseEnv,
        policies: Dict[str, Policy],
        episode: Episode,
        env_index: int,
        **kwargs
    ):
        pass
        # advised_policy = policies[DEFAULT_POLICY_ID]

        # episode.custom_metrics["action_advice"] = advised_policy.action_info["action_advice"]
        # episode.custom_metrics["action_student"] = advised_policy.action_info["action_student"]
        # episode.custom_metrics["action_introspection"] = advised_policy.action_info["action_introspection"]

    # def on_learn_on_batch(
    #     self, *, policy: Policy, train_batch: SampleBatch, result: dict, **kwargs
    # ) -> None:
    #     """Called at the beginning of Policy.learn_on_batch().

    #     Note: This is called before 0-padding via
    #     `pad_batch_to_sequences_of_same_size`.

    #     Also note, SampleBatch.INFOS column will not be available on
    #     train_batch within this callback if framework is tf1, due to
    #     the fact that tf1 static graph would mistake it as part of the
    #     input dict if present.
    #     It is available though, for tf2 and torch frameworks.

    #     Args:
    #         policy: Reference to the current Policy object.
    #         train_batch: SampleBatch to be trained on. You can
    #             mutate this object to modify the samples generated.
    #         result: A results dict to add custom metrics to.
    #         kwargs: Forward compatibility placeholder.
    #     """
    #     policy.update_teacher(train_batch)