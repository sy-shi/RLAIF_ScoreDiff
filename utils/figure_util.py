import matplotlib.pyplot as plt
import os
import numpy as np
import pandas as pd
from scipy.interpolate import interp1d

class ResultReaderFromFile:
    def __init__(self, folder_name=None):
        if folder_name is None:
            results_dir = os.path.expanduser('~/Desktop/teacher-student_code/TS-LLM/data')
            latest_dir = max([os.path.join(results_dir, d) for d in os.listdir(results_dir)], key=os.path.getmtime)
        else:
            latest_dir = folder_name
        self.file_names = os.path.join(latest_dir, 'progress.csv')
        self.episode_reward = []
        self.episode_length = []
        self.episode_reward_mean = []
        self.time_steps = []
        self.time_steps_total = 0

    def read_results(self, smooth_weight:float):
        data = pd.read_csv(self.file_names)
        for index, row in data.iterrows():
            # for reward in eval(row['evaluation/hist_stats/episode_reward']):
            #     self.episode_reward.append(reward)
            # for length in eval(row['evaluation/hist_stats/episode_lengths']):
                # self.episode_length.append(length)
            if index == 0:
                self.time_steps_total = row['timesteps_total']
            # if 8000 + row['timesteps_total'] - self.time_steps_total <= 240000:  # remember to change with max steps and batch size
            # self.time_steps.append(8000 + row['timesteps_total'] - self.time_steps_total)
            self.time_steps.append(row['timesteps_total'])
            self.episode_reward_mean.append(row['episode_reward_mean'])
        self.smooth(smooth_weight)


    def _smooth_curve(self, scalars: np.ndarray, weight: float) -> list:  # Weight between 0 and 1
        scalars = np.nan_to_num(scalars)
        last = scalars[0]  # First value in the plot (first timestep)
        smoothed = list()
        for i in range(scalars.shape[0]):
            point = scalars[i]
            smoothed_val = last * weight + (1 - weight) * point  # Calculate smoothed value
            smoothed.append(smoothed_val)                        # Save it
            last = smoothed_val                                  # Anchor the last smoothed value
        return list(smoothed)

    def smooth(self, weight:float=0):
        self.episode_reward_mean = self._smooth_curve(self.episode_reward_mean, weight)



def curve_average(curves: list[ResultReaderFromFile]):
    timesteps = {}
    episode_reward_mean = {}
    action_advice_mean = {}

    for i in range(len(curves)):
        timesteps[i] = curves[i].time_steps
        episode_reward_mean[i] = curves[i].episode_reward_mean
    # Define a common set of timesteps for interpolation
    common_timesteps = np.linspace(0, 1000000, num=1000)  # For example, 1000 points from 0 to 1M
    interp_funcs = [interp1d(time_steps, episode_rewards, kind='linear', bounds_error=False, fill_value="extrapolate")
                    for time_steps, episode_rewards in [(timesteps[i], episode_reward_mean[i]) for i in range(len(curves))]]
    interpolated_data = [f(common_timesteps) for f in interp_funcs]
    # Calculate average and variance
    average_performance = np.mean(interpolated_data, axis=0)
    max_performance = np.max(interpolated_data, axis=0)
    min_performance = np.min(interpolated_data, axis=0)
    performance_variance = np.var(np.nan_to_num(interpolated_data), axis=0)
    # common_timesteps = curves[0].time_steps
    # max_performance = np.max(np.nan_to_num(np.array([curves[i].episode_reward_mean for i in range(len(curves))])), axis=0)
    # min_performance = np.min(np.nan_to_num(np.array([curves[i].episode_reward_mean for i in range(len(curves))])), axis=0)
    # average_performance = np.mean(np.nan_to_num(np.array([curves[i].episode_reward_mean for i in range(len(curves))])), axis=0)
    # print(curves[i].episode_reward_mean)
    # performance_variance = np.std(np.nan_to_num(np.array([curves[i].episode_reward_mean for i in range(len(curves))])), axis=0)
    return common_timesteps, average_performance, performance_variance, max_performance, min_performance


def curve_average2(curves: list[ResultReaderFromFile]):
    timesteps = {}
    episode_reward_mean = {}
    action_advice_mean = {}

    for i in range(len(curves)):
        timesteps[i] = curves[i].time_steps
        episode_reward_mean[i] = curves[i].episode_reward_mean
    # Define a common set of timesteps for interpolation
    common_timesteps = np.linspace(0, 1000000, num=1000)  # For example, 1000 points from 0 to 1M
    interp_funcs = [interp1d(time_steps, episode_rewards, kind='linear', bounds_error=False, fill_value="extrapolate")
                    for time_steps, episode_rewards in [(timesteps[i], episode_reward_mean[i]) for i in range(len(curves))]]
    interpolated_data = [f(common_timesteps) for f in interp_funcs]
    # Calculate average and variance
    average_performance = np.mean(interpolated_data, axis=0)
    max_performance = np.max(interpolated_data, axis=0)
    min_performance = np.min(interpolated_data, axis=0)
    performance_variance = np.var(interpolated_data, axis=0)
    return common_timesteps, average_performance, performance_variance, max_performance, min_performance