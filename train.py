import argparse
import config.config_loader
import environment.env_loader
import model.model_wrapper
from callbacks import CustomCheckpointCallback
import model.util
import numpy as np
import pickle
import ray
import teacher_student.util
import tempfile
import torch
import os
import csv

from rewardModel import RewardModel

from datetime import datetime

from ray import tune
from ray.rllib.policy.sample_batch import DEFAULT_POLICY_ID
from ray.tune.logger import pretty_print, UnifiedLogger
from ray.tune.result import DEFAULT_RESULTS_DIR
from ray.rllib.utils.torch_utils import convert_to_torch_tensor
from ray.rllib.utils.numpy import convert_to_numpy
from generate_ranking_data import *

from matplotlib import animation
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser()
parser.add_argument(
    "--name",
    type = str,
)
parser.add_argument(
    "--mode",
    default="train",
    choices=["train", "debug", "evaluate", "export", "experiments", "info", "rollout_rm"],
    help="Running mode. Train to train a new model from scratch. Debug to run a minimal overhead training loop for debugging purposes."
         "Evaluate to rollout a trained pytorch model. Export to export a trained pytorch model from a checkpoint."
)
parser.add_argument(
    "--config",
    type=str,
    default="pacman",
    help="Configuration file to use."
)
parser.add_argument(
    "--checkpoint-dir",
    type=str,
    default=None,
    help="Path to the directory containing an RLlib checkpoint. Used with the 'export' mode.",
)
parser.add_argument(
    "--model_path",
    type=str,
    default=None
)
parser.add_argument(
    "--import-path",
    type=str,
    default=None,
    help="Path to a pytorch saved model. Used with the 'evaluate' mode.",
)
parser.add_argument(
    "--export-path",
    type=str,
    default=None,
    help="Path to export a pytorch saved model. Used with the 'export' mode."
)
parser.add_argument(
    "--eval-episodes",
    type=int,
    default=1,
    help="Number of episodes to rollout for evaluation."
)
parser.add_argument(
    "--save-eval-rollouts",
    action="store_true",
    help="Whether to save (state, action, importance) triplets from evaluation rollouts.",
)
parser.add_argument(
    "--model-type",
    default="torch",
    choices=["torch", "tree"],
    help="The type of model to be imported. Options are 'torch' for a pytorch model or 'tree' for an sklearn tree classifier."
)
parser.add_argument(
    "--hpo",
    action="store_true",
    help="Whether to perform HPO during training."
)
parser.add_argument(
    "--max-steps",
    type=int,
    default=400000,
    help="Maximum number of training iterations."
)
parser.add_argument(
    "--num-samples",
    type=int,
    default=1,
    help="Number of samples to run per configuration."
)
parser.add_argument(
    "--max-concurrent",
    type=int,
    default=0,
    help="Maximum number of concurrent trials."
)


# A default logger to escape out ALE's stupid new ALE namespace which is forbidden by the windows file system
# def default_logger(config):
#     timestr = datetime.today().strftime("%Y-%m-%d_%H-%M-%S")
#     logdir_prefix = "{}_{}_{}".format(str(config["alg"]), config["env"].replace("/", "_"), timestr)

#     logdir = tempfile.mkdtemp(prefix=logdir_prefix, dir=DEFAULT_RESULTS_DIR)

#     def default_logger_creator(config):
#         """Creates a Unified logger with the default prefix."""
#         return UnifiedLogger(config, logdir, loggers=None)

#     return default_logger_creator

def default_logger(config, args):
    "create logger to a customized path"
    timestr = datetime.today().strftime("%Y-%m-%d_%H-%M-%S")
    logdir_prefix = "{}_{}_{}".format(str(config["alg"]), config["env"].replace("/", "_"), timestr)
    dir = "./data/env3_exp1_run"
    if not os.path.exists(dir):
        os.makedirs(dir, exist_ok=True)
    logdir = tempfile.mkdtemp(prefix=logdir_prefix, dir=dir)
    def default_logger_creator(config):
        """Creates a Unified logger with the default prefix."""
        return UnifiedLogger(config, logdir, loggers=None)
    return default_logger_creator


def add_new_ranks(ranks, rank):
    if rank == Ranking.GREATER:
        ranks.append(ranks[-1] + 1)
    elif rank == Ranking.LESSER:
        ranks.append(ranks[-1] - 1)
    if rank == Ranking.EQUAL:
        ranks.append(ranks[-1])


def rollout_rm_episode(loaded_model, env,render_mode=True, render_name=None, teacher_model=None):
    obs = env.reset() #TODO
    
    done = False
    frames = []


    step_idx = 0
    value_estimates, vf_ranks, true_ranks = [], [0], [0]
    landmarks = [[],[]]
    #lst_pos = [None, None]
    total_steps = 0
    accu_vf = 0
    lst_pos = None
    action = None
    thre = 0
    #actions = [[3,1,3,2,3,2,3,4,3,2,3,2,3,1,3,2,3,2,3,1,5,2,3,2,3,2,3,2,3,3,3,2,3,2,2,3,1,3,3,1,3,2,3,3,2,3], [4,2,3,3,1,3,1,3,1,3,1,3,5,3,3,1,3,1,3,1,3,1,3,3,3,1,3,2,3,2,3,3,3,2,3,2,3,1,3,3,2,3,0,4,5,0]]

    while not done:
    # while min(total_steps) < 100:
        dir = obs['image'][env.unwrapped.agent_pos[0], env.unwrapped.agent_pos[1], 2]
        obs['image'][env.unwrapped.agent_pos[0], env.unwrapped.agent_pos[1], 2] = 0
        
        # print(lst_obs[i]['image'][lst_pos[i][0], lst_pos[i][1], 2])
        # print('*********')
        # print(obs[i]['image'][env.env.agents[i].pos[0], env.env.agents[i].pos[1], 2])
        
        # vf ranking
        with torch.no_grad():
            x = torch.tensor([convert_to_numpy(obs["image"])], dtype=torch.float32)

            # Reorder the state from (NHWC) to (NCHW).
            # Most images put channel last, but pytorch conv expects channel first.
            x = x.permute(0, 3, 1, 2)
            model = teacher_model

            value_estimates.append(float(model(x).squeeze().numpy()))

        obs['image'][env.unwrapped.agent_pos[0], env.unwrapped.agent_pos[1], 2] = dir
        # vf ranking eval
        if (step_idx > 0) and not np.all(lst_pos == env.unwrapped.agent_pos) or \
            action == MiniGridEnv.Actions.pickup or action == MiniGridEnv.Actions.toggle:
            lst_obs['image'][lst_pos[0], lst_pos[1], 2] = obs['image'][env.unwrapped.agent_pos[0], env.unwrapped.agent_pos[1], 2]
            # evaluation
            # env.unwrapped.print_obs(lst_obs['image'])
            # env.unwrapped.print_obs(obs['image'])
            ground_truth = rank_states(lst_obs["image"], obs["image"], action, env)
            # print('GT', ground_truth)

            total_steps += 1
            add_new_ranks(true_ranks, ground_truth)

            if value_estimates[-1] < (value_estimates[-2] * (1-thre)):
                vf_rank = Ranking.LESSER
            elif value_estimates[-1] > (value_estimates[-2] * (1+thre)):
                vf_rank = Ranking.GREATER
            else:
                vf_rank = Ranking.EQUAL

            # print('vf rank', vf_rank, value_estimates[-2], value_estimates[-1])
            # print("\n\n\n\n===========================")
            add_new_ranks(vf_ranks, vf_rank)

            if str(ground_truth) == str(vf_rank):# or abs(value_estimates[i][-2] - value_estimates[i][-1]) <= 0.05:
                accu_vf += 1
            else:
                env.unwrapped.print_obs(lst_obs['image'])
                env.unwrapped.print_obs(obs['image'])
                print('GT', ground_truth)
                print('vf rank', vf_rank, value_estimates[-2], value_estimates[-1])
                print("\n\n\n===================================")
        # elif np.all(lst_pos[i] == env.env.agents[i].pos) and reward_dict[i] <= 0:
        #     #bias[i] += value_estimates[i][-1] - value_estimates[i][-2]
        #     value_estimates[i][-1] = value_estimates[i][-2]

        lst_pos = deepcopy(env.unwrapped.agent_pos)

        if render_name is not None:
            img = env.render()
            frames.append(img)

        #print(step_idx,obs)
        lst_obs = deepcopy(obs)
        action, _, _ = loaded_model.get_action(obs)
        obs, reward, done, info = env.step(action)

        # for i in range(n_agents):
        #     if reward_dict[i] > 0:
        #         landmarks[i].append(step_idx)

        done = done
        step_idx += 1

    loggings = [os.getcwd()+'/videos/' + render_name + '/teacher_vals1.csv', os.getcwd()+'/videos/' + render_name + '/true_ranks1.csv',\
                os.getcwd()+'/videos/' + render_name + '/teacher_vals0.csv', os.getcwd()+'/videos/' + render_name + '/true_ranks0.csv',]
    for f in loggings:
        if os.path.exists(f):
            # Delete the file
            os.remove(f)


    file_path = os.getcwd()+'/videos/' + render_name + '/teacher_vals'+'.csv'
    file1 = open(file_path, 'w')
    # with open(file_path, 'a', newline='') as file:
    writer = csv.writer(file1)
    writer.writerow(value_estimates)

    file_path = os.getcwd()+'/videos/' + render_name + '/true_ranks'+'.csv'
    file2 = open(file_path, 'w')
    # with open(file_path, 'a', newline='') as file:
    writer = csv.writer(file2)
    writer.writerow(true_ranks)


    env.reset()

    print('total steps', len(frames))
    print('total ranking steps', total_steps)
    print('vf accuracy', accu_vf / total_steps)
    # print(steps_per_stage)

    return frames, accu_vf, total_steps

def rollout_episode(loaded_model, env, max_steps = 200, flatten_obs = True, obs_save_label = "image", save_rollouts = False, teacher_model=None):
    obs = env.reset()
    done = False

    states = []

    step_idx = 0
    total_reward = 0
    while not done:
        action, _, importance = loaded_model.get_action(obs)

        if save_rollouts:
            if isinstance(obs, dict):
                if obs_save_label in obs:
                    obs = obs[obs_save_label]
                else:
                    raise("Observation key word not in observation dict to save. Change keyword or disable flattening.")


            if flatten_obs:
                obs = obs.flatten()

            states.append([obs, action, importance])

        obs, reward, done, info = env.step(action)
        step_idx += 1
        total_reward += reward
        env.render(pause = 0.5)
        if step_idx > max_steps:
            break

    env.reset()

    return states, step_idx, total_reward


def rollout_episodes(loaded_model, env, num_episodes = 1, save_rollouts = False, max_steps = 500, teacher_model=None):
    all_episode_states = []
    num_steps = []
    rewards = []

    for _ in range(num_episodes):
        states, steps, reward = rollout_episode(loaded_model, env, max_steps = max_steps, save_rollouts = save_rollouts)
        all_episode_states.append(states)
        num_steps.append(steps)
        rewards.append(reward)

    if save_rollouts:
        with open('model_trajectories.pickle', 'wb') as f:
            pickle.dump(all_episode_states, f)

    return np.mean(rewards), np.mean(num_steps)

def save_frames_as_gif(frames, path='./', filename='gym_animation.gif'):

    #Mess with this to change frame size
    plt.figure(figsize=(frames[0].shape[1] / 72.0, frames[0].shape[0] / 72.0), dpi=72)

    patch = plt.imshow(frames[0])
    plt.axis('off')

    def animate(i):
        patch.set_data(frames[i])

    anim = animation.FuncAnimation(plt.gcf(), animate, frames = len(frames), interval=50)
    anim.save(path + filename, writer='imagemagick', fps=20)


def rollout_rm_episodes(loaded_model, env, num_episodes = 1, save_rollouts = False, max_steps = 500, render_name = None, teacher_model=None):
    all_episode_states = []
    num_steps = []
    accu_vfs = []
    frames = []
    for _ in range(num_episodes):
        frame, accu_vf, total_steps = rollout_rm_episode(loaded_model, env, render_mode=True, render_name=render_name, teacher_model=teacher_model)
        frames.append(frame)
        # all_episode_states.append(states)
        num_steps.append(total_steps)
        accu_vfs.append(accu_vf)

    # if save_rollouts:
    #     with open('model_trajectories.pickle', 'wb') as f:
    #         pickle.dump(all_episode_states, f)
    if render_name is not None:
        dir = 'videos/' + render_name + '/frames/'
        if not os.path.exists(dir):
            os.makedirs(dir, exist_ok=True)

        print("saving rollout video ...")
        for i in range(len(frames)):
            file_name = "eval_episode_" + str(i) + ".gif"
            save_frames_as_gif(frames[i], dir, file_name)
    return np.mean(accu_vfs), np.mean(num_steps)


def rollout_steps(loaded_model, env, num_steps = 100, max_steps = 500, flatten_obs = True, save_rollouts = True):
    steps_collected = 0

    all_episode_states = []

    while steps_collected < num_steps:
        states, steps, _ = rollout_episode(loaded_model, env, max_steps = max_steps, flatten_obs = flatten_obs, save_rollouts = save_rollouts)
        all_episode_states.extend(states)
        steps_collected += steps

    all_episode_states = all_episode_states[:num_steps]

    return all_episode_states


def export_model(checkpoint_dir, export_path, exp_config):
    trainer = teacher_student.util.get_trainer(exp_config["alg"])(config=exp_config, logger_creator=default_logger(exp_config, args))

    trainer.restore(checkpoint_dir)

    policy = trainer.get_policy(DEFAULT_POLICY_ID)
    policy_model = policy.model

    torch.save(policy_model.state_dict(), export_path)


def train_model(mode = "train", checkpoint_dir = None, export_path = None, import_path = None):
    ray.init(local_mode = False)

    model.util.register_models()
    environment.env_loader.register_envs()

    exp_config = config.config_loader.ConfigLoader.load_config(args.config, args.hpo)

    model_type = model.model_wrapper.ModelType.TORCH
    if args.model_type == "tree":
        model_type = model.model_wrapper.ModelType.TREE

    if mode == "debug":
        trainer = teacher_student.util.get_trainer(exp_config["alg"])(config=exp_config, logger_creator=default_logger(exp_config, args))

        # run manual training loop and print results after each iteration
        for _ in range(5):
            result = trainer.train()
            print(pretty_print(result))

        policy = trainer.get_policy(DEFAULT_POLICY_ID)
        policy_model = policy.model
        # print(policy_model)

    elif mode == "train":
        stop = {
            "timesteps_total": args.max_steps,
        }

        results = tune.run(
            teacher_student.util.get_trainer(exp_config["alg"]),
            name=args.name,
            config=exp_config,
            checkpoint_freq=50,
            checkpoint_score_attr="episode_reward_mean",
            keep_checkpoints_num=5,
            stop=stop,
            num_samples=args.num_samples,
            max_concurrent_trials=args.max_concurrent if args.max_concurrent > 0 else None,
            checkpoint_at_end=True,
            callbacks=[CustomCheckpointCallback(args.model_path)],
            local_dir="./data/env3_baseline_compare4")

    elif mode == "experiments":
        stop = {
            "timesteps_total": args.max_steps,
        }

        results = tune.run(
            teacher_student.util.get_trainer(exp_config["alg"]),
            config=exp_config,
            checkpoint_freq=25,
            checkpoint_at_end=True,
            num_samples = 5,
            stop=stop,
            callbacks=[CustomCheckpointCallback(args.model_path)],
            local_dir="./data")

    elif mode == "rollout_rm":
        if import_path is None:
            raise("import_path must be specified for the 'evaluate' mode.")

        env = environment.env_loader.env_maker(exp_config["env_config"], exp_config["env"])

        loaded_model = model.model_wrapper.ModelWrapper(model_type)
        loaded_model.load(import_path, env.action_space, env.observation_space, exp_config["model"])
        configure = {
            "conv_filters": [
                [16, [2, 2], 1, (2,1)],
                ["pool", [2, 2], 2, (0,0)],
                [32, [2, 2], 1, (0,0)],
                [64, [2, 2], 1, (0,0)]
            ],
            "conv_activation": True,
            "fc_layer_sizes": [[256, 512], [512, 256], [256, 128], [128, 64], [64, 16], [16, 1]],
            "clip_at_last": "",
            "clip_scale": 1,
            } 
        teacher_model = RewardModel(config=configure)
        teacher_model.eval()
        teacher_model.load_state_dict(torch.load(os.getcwd() + '/LLM/RM/models/reward_model6k.pth', map_location=next(teacher_model.parameters()).device))

        accu_vf, steps = rollout_rm_episodes(loaded_model, env, num_episodes=args.eval_episodes, render_name="test", teacher_model=teacher_model)
        print("Evaluated {} episodes. Average vf accurate: {}. Average num steps: {}".format(args.eval_episodes, accu_vf, steps))


    elif mode == "evaluate":
        if import_path is None:
            raise("import_path must be specified for the 'evaluate' mode.")

        env = environment.env_loader.env_maker(exp_config["env_config"], exp_config["env"])

        loaded_model = model.model_wrapper.ModelWrapper(model_type)
        loaded_model.load(import_path, env.action_space, env.observation_space, exp_config["model"])

        reward, steps = rollout_episodes(loaded_model, env, num_episodes=args.eval_episodes, save_rollouts=args.save_eval_rollouts)

        print("Evaluated {} episodes. Average reward: {}. Average num steps: {}".format(args.eval_episodes, reward, steps))

    elif mode == "export":
        if checkpoint_dir is None:
            raise("checkpoint_dir must be specified for the 'export' mode.")
        if export_path is None:
            raise("export_path must be specified for the 'export' mode.")

        export_model(checkpoint_dir, export_path, exp_config)

    elif mode == "info":
        if import_path is None:
            raise("import_path must be specified for the 'evaluate' mode.")
        
        env = environment.env_loader.env_maker(exp_config["env_config"], exp_config["env"])

        loaded_model = model.model_wrapper.ModelWrapper(model_type)
        loaded_model.load(import_path, env.action_space, env.observation_space, exp_config["model"])

        num_params = model.util.count_parameters(loaded_model.model)

        print(loaded_model.model)
        print("Num params: " + str(num_params))


    ray.shutdown()


if __name__ == "__main__":
    args = parser.parse_args()
    print(f"Running with following CLI options: {args}")

    train_model(args.mode, args.checkpoint_dir, args.export_path, args.import_path)
