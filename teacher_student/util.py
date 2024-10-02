from teacher_student import advised_policy
from teacher_student import advised_trainer

from ray.rllib.agents import ppo
from ray.rllib.evaluation.postprocessing import Postprocessing
from ray.rllib.policy.sample_batch import SampleBatch
from ray.rllib.utils.numpy import convert_to_numpy

import numpy as np
import scipy


def get_trainer(alg_type):
    trainer = None

    if alg_type == "ppo":
        AdvisedPPOPolicy = advised_policy.get_advised_policy(ppo.PPOTorchPolicy)
        trainer = advised_trainer.get_advised_trainer(ppo.PPOTrainer, AdvisedPPOPolicy)
        # trainer = advised_trainer.get_advised_trainer(ppo.PPOTrainer, ppo.PPOTorchPolicy)
    else:
        raise("Unknown algorithm type: {}".format(alg_type))

    # Force the trainer to allow unknown configuration options.
    # Needed to define top-level configs like "alg" which are used by this code but not RLlib.
    trainer._allow_unknown_configs = True

    return trainer


def discount_cumsum(x: np.ndarray, gamma: float) -> np.ndarray:
    """Calculates the discounted cumulative sum over a reward sequence `x`.
    y[t] - discount*y[t+1] = x[t]
    reversed(y)[t] - discount*reversed(y)[t-1] = reversed(x)[t]
    Args:
        gamma: The discount factor gamma.
    Returns:
        The sequence containing the discounted cumulative sums
        for each individual reward in `x` till the end of the trajectory.
    Examples:
        >>> x = np.array([0.0, 1.0, 2.0, 3.0])
        >>> gamma = 0.9
        >>> discount_cumsum(x, gamma)
        ... array([0.0 + 0.9*1.0 + 0.9^2*2.0 + 0.9^3*3.0,
        ...        1.0 + 0.9*2.0 + 0.9^2*3.0,
        ...        2.0 + 0.9*3.0,
        ...        3.0])
    """
    return scipy.signal.lfilter([1], [1, float(-gamma)], x[::-1], axis=0)[::-1]


def compute_advantages(
    rollout: SampleBatch,
    last_r: float,
    gamma: float = 0.9,
    lambda_: float = 1.0,
    use_gae: bool = True,
    use_critic: bool = True,
    vf_pred_col = SampleBatch.VF_PREDS,
    adv_col = Postprocessing.ADVANTAGES,
    value_target_col = Postprocessing.VALUE_TARGETS,
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

    assert (
        vf_pred_col in rollout or not use_critic
    ), "use_critic=True but values not found"
    assert use_critic or not use_gae, "Can't use gae without using a value function"

    if use_gae:
        vpred_t = np.concatenate([rollout[vf_pred_col], np.array([last_r])])
        delta_t = rollout[SampleBatch.REWARDS] + gamma * vpred_t[1:] - vpred_t[:-1]
        # This formula for the advantage comes from:
        # "Generalized Advantage Estimation": https://arxiv.org/abs/1506.02438
        rollout[adv_col] = discount_cumsum(delta_t, gamma * lambda_)
        rollout[value_target_col] = (
            rollout[adv_col] + rollout[vf_pred_col]
        ).astype(np.float32)
    else:
        rewards_plus_v = np.concatenate(
            [rollout[SampleBatch.REWARDS], np.array([last_r])]
        )
        discounted_returns = discount_cumsum(rewards_plus_v, gamma)[:-1].astype(
            np.float32
        )

        if use_critic:
            rollout[adv_col] = (
                discounted_returns - rollout[vf_pred_col]
            )
            rollout[value_target_col] = discounted_returns
        else:
            rollout[adv_col] = discounted_returns
            rollout[value_target_col] = np.zeros_like(
                rollout[adv_col]
            )

    rollout[adv_col] = rollout[adv_col].astype(
        np.float32
    )

    return rollout


def compute_gae_for_sample_batch(
    policy,
    sample_batch,
    vf_fn,
    other_agent_batches = None,
    episode = None,
    vf_pred_col = SampleBatch.VF_PREDS,
    adv_col = Postprocessing.ADVANTAGES,
    value_target_col = Postprocessing.VALUE_TARGETS,
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
        last_r = vf_fn(**input_dict)

    # Adds the policy logits, VF preds, and advantages to the batch,
    # using GAE ("generalized advantage estimation") or not.
    batch = compute_advantages(
        sample_batch,
        last_r,
        policy.config["gamma"],
        policy.config["lambda"],
        use_gae=policy.config["use_gae"],
        use_critic=policy.config.get("use_critic", True),
        vf_pred_col=vf_pred_col,
        adv_col=adv_col,
        value_target_col=value_target_col,
    )

    return batch