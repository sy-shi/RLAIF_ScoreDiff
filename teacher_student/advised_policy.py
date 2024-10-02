import gym
import os
from rewardModel import RewardModel, RandomNetwork
import model.introspection_model as introspection_model
import model.model_wrapper as model_wrapper
import numpy as np
import torch
from typing import List, Type, Union, Dict, Tuple, Optional

from . import advise
from . import util

from ray.rllib.evaluation.postprocessing import Postprocessing
from ray.rllib.models.action_dist import ActionDistribution
from ray.rllib.models.modelv2 import restore_original_dimensions, ModelV2
from ray.rllib.policy.sample_batch import SampleBatch
from ray.rllib.policy.view_requirement import ViewRequirement
from ray.rllib.utils.numpy import convert_to_numpy
from ray.rllib.utils.torch_utils import (
    convert_to_torch_tensor,
    explained_variance,
    sequence_mask,
    warn_if_infinite_kl_divergence,
)
from ray.rllib.evaluation.postprocessing import (
    Postprocessing,
    discount_cumsum
)
from ray.rllib.utils.typing import ModelWeights, TensorType
from ray.rllib.evaluation.episode import Episode
from ray.rllib.policy.policy import Policy
import csv, traceback

def get_advised_policy(base_class):

    class AdvisedPolicy(base_class):
        def __init__(self, observation_space, action_space, config):
            self.advice_mode = config.get("advice_mode", False)
            self.smooth_mode = config.get("smooth_mode")
            self.teacher_model_path = config.get("teacher_model_path")
            self.teacher_model_config = config.get("teacher_model_config")
            self.max_steps = config.get("max_steps")
            self.reward_clipping_func = config.get("reward_clipping_func")
            self.max_reward_scale = config.get("max_reward_scale")
            self.score_mean = config.get("score_mean")
            self.score_std = config.get("score_std")
            self.const_penalty = config.get("constant_penalty")
            self.random_network_path = config.get("random_network_path")
            self.action_log = {}
            self.teacher_model = None
            self.random_network = None
            self.teacher_initialized = False
            super().__init__(observation_space, action_space, config)

            self.device = next(self.model.parameters()).device

            if self.advice_mode == 'llm' or self.advice_mode == 'baseline':
                self.teacher_model = RewardModel(self.teacher_model_config)
                self.teacher_model.to(self.device)
                assert os.path.exists(self.teacher_model_path)
                self.teacher_model.load_state_dict(torch.load(self.teacher_model_path, map_location=next(self.teacher_model.parameters()).device))
                self.teacher_model.eval()
                if self.reward_clipping_func == "random":
                    self.random_network = RandomNetwork(self.teacher_model_config, file_path=self.random_network_path)
                self.teacher_initialized = True

        def postprocess_trajectory(self, sample_batch, other_agent_batches=None, episode=None):
            # print('*************************')
            # print('the old shape of sample_batch[SampleBatch.VF_PREDS]',sample_batch[SampleBatch.VF_PREDS].shape)


            # #sample_batch = PPOTorchPolicy.postprocess_trajectory(sample_batch, other_agent_batches, episode)
            # sample_batch = super().postprocess_trajectory(sample_batch, other_agent_batches, episode)
            # # print('have get the sample batch')
            # # sample_batch[SampleBatch.VF_PREDS] = self.teacher_model.model.value_function(
            # #     convert_to_torch_tensor(
            # #         sample_batch[SampleBatch.CUR_OBS], self.device
            # #     )
            # # ).cpu().detach().numpy()

            # sample_batch = self._lazy_tensor_dict(sample_batch)
            # self.teacher_model.model(sample_batch)
            # vf_preds = self.teacher_model.model.value_function()

            # sample_batch[SampleBatch.VF_PREDS] = vf_preds

            # sample_batch = convert_to_numpy(sample_batch)

            # print('the new shape of sample_batch[SampleBatch.VF_PREDS]',sample_batch[SampleBatch.VF_PREDS].shape)
            # print('*************************')
            # return sample_batch
            if True:

                with torch.no_grad():
                    return self.compute_gae_for_sample_batch(
                        self, sample_batch, other_agent_batches, episode
                    )
            else:
                batch = super().postprocess_trajectory(sample_batch, other_agent_batches, episode)
                return batch
        
        def compute_advantages(self,
            rollout: SampleBatch,
            last_r: float,
            value_diff,
            gamma: float = 0.9,
            lambda_: float = 1.0,
            use_gae: bool = True,
            use_critic: bool = True,
        ):
            """Given a rollout, compute its value targets and the advantages.

            Args:
                rollout: SampleBatch of a single trajectory.
                last_r: Value estimation for last observation.
                gamma: Discount factor.
                lambda_: Parameter for GAE.
                use_gae: Using Generalized Advantage Estimation.
                use_critic: Whether to use critic (value estimates). Setting
                    this to False will use 0 as baseline.

            Returns:
                SampleBatch with experience from rollout and processed rewards.
            """

            # assert (
            #     SampleBatch.VF_PREDS in rollout or not use_critic
            # ), "use_critic=True but values not found"
            assert use_critic or not use_gae, "Can't use gae without using a value function"
            """
                Advice with Teacher Value Function
            """

            # Compute the value estimates for each observation
            sample = self._lazy_tensor_dict(rollout.copy())

            # print('*****************************')
            # print(sample)
            # print(sample['obs'])
            # print(sample['new_obs'])
            # print(sample[SampleBatch.DONES])
            # print('*****************************')

            # print(sample['obs'].shape)
            rm_eval = False
            if (self.advice_mode == 'llm' or self.advice_mode == 'baseline') and self.teacher_initialized:
                try:
                    with torch.no_grad():
                        x1 = torch.tensor([convert_to_numpy(o[5:]) for o in sample['obs'].float()]).reshape(-1,13,9,3)

                        for i in range(x1.shape[0]):
                            obs_r = x1[i]
                            for cn in range(9):
                                for lm in range(13):
                                    obj_type = obs_r[lm,cn,0]
                                    if obj_type == 10:
                                        x1[i][lm, cn, 2] = 0

                        # Reorder the state from (NHWC) to (NCHW).
                        # Most images put channel last, but pytorch conv expects channel first.
                        x1 = x1.permute(0, 3, 1, 2)

                        value_estimates = (self.teacher_model(x1).squeeze())


                        # current new state
                        x2 = torch.tensor([convert_to_numpy(o[5:]) for o in sample['new_obs'].float()]).reshape(-1,13,9,3)

                        for i in range(x2.shape[0]):
                            obs_r = x2[i]
                            for cn in range(9):
                                for lm in range(13):
                                    obj_type = obs_r[lm,cn,0]
                                    if obj_type == 10:

                                        x2[i][lm, cn, 2] = 0

                        x2 = x2.permute(0, 3, 1, 2)

                        value_estimates_ = (self.teacher_model(x2).squeeze())

                        rm_eval = True


                except Exception as e:
                    print("An error occurred:")
                    traceback.print_exc()
                
                try:
                    if self.reward_clipping_func == "tanh":
                        value_diff = self.max_reward_scale * torch.tanh(value_estimates_ - value_estimates).cpu()
                    elif self.reward_clipping_func == "sigmoid":
                        value_diff = self.max_reward_scale * (2 * torch.sigmoid(value_estimates_ - value_estimates) - 1).cpu()
                    elif self.reward_clipping_func == "constant":
                        diff = value_estimates_ - value_estimates
                        value_diff = self.max_reward_scale * (torch.where(diff > 0, torch.ones_like(diff), torch.where(diff == 0, torch.zeros_like(diff), -torch.ones_like(diff))))
                    elif self.reward_clipping_func == "random":
                        rand_rwd = torch.rand(value_estimates.shape)
                        with torch.no_grad():
                            rand_rwd = self.random_network(x1).squeeze()
                        diff = value_estimates_ - value_estimates
                        value_diff = self.max_reward_scale * rand_rwd * (torch.where(diff > 0, torch.ones_like(diff), torch.where(diff == 0, torch.zeros_like(diff), -torch.ones_like(diff))))
                    else:
                        value_diff = (value_estimates_ - value_estimates).cpu()

                except:
                    print('\n\n\n\n\n')
                    print('error occurs')
                    print('x with shape', x1.shape)
                    print(value_estimates)
                    print('\n\n\n\n\n')
                    exit()
                #value_diff = torch.cat((torch.tensor([0]), value_diff), dim=0)
                
                # if self.advice_mode == 'baseline':
                #     if self.reward_clipping_func == 'normalize':
                #         value_estimates = self.max_reward_scale * (value_estimates - self.score_mean) / (self.score_std + 1e-10)
                prev_eps_steps = -1
                temp_rewards = 0
                crt_r_given = 0
                rs_given = []
                ends = []

                win_len = 1

                try:
                    teacher_rewards = np.zeros(value_diff.shape)
                    for i in range(value_diff.size(0)):

                        if (self.advice_mode == 'llm' or self.advice_mode == 'baseline') and self.teacher_initialized:
                            # if value_diff[i] < 0:
                            #     value_diff[i] = torch.tensor(0)
                            # if i > 0 and rollout[SampleBatch.REWARDS][i-1] > 0: # get rewards at the last step
                            #     value_diff[i] = torch.tensor(0)


                            # if value_diff[i] > 0:
                            #     teacher_rewards[i] = value_diff[i] - 0.99 * (i - prev_eps_steps) / max_steps
                            # if abs(value_diff[i]) < 0.005:
                            #     teacher_rewards[i] = - 0.99 * (i - prev_eps_steps) / max_steps
                            if self.advice_mode == 'llm':
                                value_diff[i] = (value_diff[i] - self.score_mean) / (self.score_std+1e-8) - self.const_penalty
                            elif self.advice_mode == 'baseline':
                                value_diff[i] = (value_estimates[i] - self.score_mean) / (self.score_std+1e-8) - self.const_penalty
                            # if value_diff[i] == 0:
                            #     value_diff[i] = - 0.99 * (i - prev_eps_steps) / self.max_steps
                            # else:
                            #     if self.advice_mode == 'baseline':
                            #         value_diff[i] = value_estimates[i]

                            #teacher_rewards[i] = value_diff[i] - 0.99 * (i - prev_eps_steps) / max_steps

                            #value_diff[i] += rollout[SampleBatch.REWARDS][i]

                            if rollout[SampleBatch.DONES][i] or (i - prev_eps_steps >= self.max_steps):
                                prev_eps_steps = i
                                teacher_rewards[i] = value_diff[i]

                            if i - prev_eps_steps >= win_len:
                                teacher_rewards[i] = np.mean(convert_to_numpy(value_diff[(i - win_len + 1) : (i + 1)]))

                        elif self.advice_mode == 'tfs' and self.teacher_initialized:
                            if i > 0 and rollout[SampleBatch.REWARDS][i-1] > 0: # get rewards at the last step
                                value_diff[i] = torch.tensor(0)

                            temp_rewards += value_diff[i]
                            if rollout[SampleBatch.REWARDS][i] > 0 or rollout[SampleBatch.DONES][i]:
                                if value_diff[i] <= 0:
                                    value_diff[i] = torch.tensor(0.3)

                                if self.smooth_mode == 'cache':
                                    #FAIL: teacher_rewards[i] = np.sum(convert_to_numpy(value_diff[nxt_reward_time:(i+1)])) - 0.9 * (i - prev_eps_steps) / 100
                                    teacher_rewards[i] = np.max([temp_rewards - 0.9 * (i - prev_eps_steps) / self.max_steps, rollout[SampleBatch.REWARDS][i]-crt_r_given, 0.1])

                                    #FAIL: teacher_rewards[i] = np.max([rollout[SampleBatch.REWARDS][i]-crt_r_given, 0.1])
                                    temp_rewards = 0
                                    crt_r_given = 0
                                    #nxt_reward_time = i + 1

                            elif self.smooth_mode == 'cache' and (i - prev_eps_steps >= self.max_steps):
                                teacher_rewards[i] = temp_rewards - 0.9 * (i - prev_eps_steps) / self.max_steps
                                temp_rewards = 0

                            elif self.smooth_mode == 'cache' and abs(temp_rewards) >= 0.5:
                                teacher_rewards[i] = temp_rewards
                                crt_r_given += temp_rewards
                                temp_rewards = 0

                            if self.smooth_mode == 'win' and i - prev_eps_steps >= win_len:
                                teacher_rewards[i] = np.mean(convert_to_numpy(value_diff[(i - win_len + 1) : (i + 1)]))

                            if (i % win_len != 0 or i // win_len < 1) and (rollout[SampleBatch.REWARDS][i] <= 0):
                                teacher_rewards[i] = 0

                        if rollout[SampleBatch.DONES][i] or (i - prev_eps_steps >= self.max_steps):
                            prev_eps_steps = i
                            crt_r_given = 0
                            ends.append(True)
                        else:
                            ends.append(False)

                        rs_given.append(crt_r_given)
                except:
                    teacher_rewards = np.zeros(1)
                    if self.advice_mode == 'llm':
                        teacher_rewards[0] = (value_diff - self.score_mean) / (self.score_std+1e-8) - self.const_penalty
                    elif self.advice_mode == 'baseline':
                        teacher_rewards[0] = (value_estimates - self.score_mean) / (self.score_std+1e-8) - self.const_penalty
                            
                    # teacher_rewards[0] = value_diff - 0.9 / self.max_steps

            if use_gae:
                vpred_t = np.concatenate([rollout[SampleBatch.VF_PREDS], np.array([last_r])])
                if self.advice_mode and rm_eval:
                    delta_t = convert_to_numpy(teacher_rewards) + gamma * vpred_t[1:] - vpred_t[:-1]
                else:
                    delta_t = rollout[SampleBatch.REWARDS] + gamma * vpred_t[1:] - vpred_t[:-1]
                delta_t = convert_to_numpy(delta_t)
                # This formula for the advantage comes from:
                # "Generalized Advantage Estimation": https://arxiv.org/abs/1506.02438
                rollout[Postprocessing.ADVANTAGES] = discount_cumsum(delta_t, gamma * lambda_)
                rollout[Postprocessing.VALUE_TARGETS] = (
                    rollout[Postprocessing.ADVANTAGES] + rollout[SampleBatch.VF_PREDS]
                    #rollout[Postprocessing.ADVANTAGES] + value_estimates.cpu().numpy()
                ).astype(np.float32)
            else:
                if self.advice_mode and rm_eval:
                    rewards_plus_v = np.concatenate(
                        [convert_to_numpy(teacher_rewards), np.array([last_r])]
                    )
                else:
                    rewards_plus_v = np.concatenate(
                        [rollout[SampleBatch.REWARDS], np.array([last_r])]
                    )
                discounted_returns = discount_cumsum(rewards_plus_v, gamma).astype(
                    np.float32
                )

                if use_critic:
                    rollout[Postprocessing.ADVANTAGES] = (
                        discounted_returns - rollout[SampleBatch.VF_PREDS]
                        #discounted_returns - value_estimates
                    )
                    rollout[Postprocessing.VALUE_TARGETS] = discounted_returns
                else:
                    rollout[Postprocessing.ADVANTAGES] = discounted_returns
                    rollout[Postprocessing.VALUE_TARGETS] = np.zeros_like(
                        rollout[Postprocessing.ADVANTAGES]
                    )

            rollout[Postprocessing.ADVANTAGES] = rollout[Postprocessing.ADVANTAGES].astype(
                np.float32
            )

            return rollout


        # @override(PPOTorchPolicy)
        def compute_gae_for_sample_batch(self,
            policy: Policy,
            sample_batch: SampleBatch,
            other_agent_batches = None,
            episode: Optional[Episode] = None,
        ) -> SampleBatch:
            """Adds GAE (generalized advantage estimations) to a trajectory.

            The trajectory contains only data from one episode and from one agent.
            - If  `config.batch_mode=truncate_episodes` (default), sample_batch may
            contain a truncated (at-the-end) episode, in case the
            `config.rollout_fragment_length` was reached by the sampler.
            - If `config.batch_mode=complete_episodes`, sample_batch will contain
            exactly one episode (no matter how long).
            New columns can be added to sample_batch and existing ones may be altered.

            Args:
                policy: The Policy used to generate the trajectory (`sample_batch`)
                sample_batch: The SampleBatch to postprocess.
                other_agent_batches: Optional dict of AgentIDs mapping to other
                    agents' trajectory data (from the same episode).
                    NOTE: The other agents use the same policy.
                episode: Optional multi-agent episode object in which the agents
                    operated.

            Returns:
                The postprocessed, modified SampleBatch (or a new one).
            """

            # Trajectory is actually complete -> last r=0.0.
            if sample_batch[SampleBatch.DONES][-1]:
                last_r = 0.0
            # Trajectory has been truncated -> last r=VF estimate of last obs.
            else:
                # Input dict is provided to us automatically via the Model's
                # requirements. It's a single-timestep (last one in trajectory)
                # input_dict.
                # Create an input dict according to the Model's requirements.
                input_dict = sample_batch.get_single_step_input_dict(
                    policy.model.view_requirements, index="last"
                )
                last_r = policy._value(**input_dict)
                # input_dict['obs'] = {'image': torch.from_numpy(input_dict['obs'][:,:90].reshape(1,5,9,2)),\
                #                     'action_mask': torch.from_numpy(input_dict['obs'][:,90:])}

                # last_r = self.teacher_model.model.value_function(input_dict)

            # Adds the policy logits, VF preds, and advantages to the batch,
            # using GAE ("generalized advantage estimation") or not.
            batch = self.compute_advantages(
                sample_batch,
                last_r,
                policy.config["gamma"],
                policy.config["lambda"],
                use_gae=policy.config["use_gae"],
                use_critic=policy.config.get("use_critic", True),
            )

            return batch

    return AdvisedPolicy
