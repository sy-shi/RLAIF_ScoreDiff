import pickle
import os
from enum import Enum

class Ranking(Enum):
    GREATER = 0
    LESSER = 1
    EQUAL = 2


files = ["ranking_data_train_env3-llama8b.pkl"]

training_data = {"input": [], "target": [], "llm_target": []}

for i in range(len(files)):
    with open(files[i], 'rb') as file:
        data = pickle.load(file)
    inputs = data['input']
    targets = data['target']
    llm_targets = data['llm_target']
    for j in range(len(inputs)):
        training_data['input'].append(inputs[j])
        training_data['target'].append(targets[j])
        training_data['llm_target'].append(llm_targets[j])


pickle.dump(training_data, open('ranking_data_train_env3-llama8b.pkl', "wb"))

training_data = pickle.load(open("ranking_data_train_env3-llama8b.pkl", "rb"))


accu = 0

from environment.grid.doubledoor import DoubleDoorEnv
from environment.grid.wrappers import ActionMasking, FullyObsWrapper

env = DoubleDoorEnv({})
env = ActionMasking(FullyObsWrapper(env))

for i in range(len(training_data['input'])):
    if str(training_data['target'][i]) == str(training_data['llm_target'][i]):
        accu += 1
#     else:
#         print('----------------------ranking----------------------')
#         env.unwrapped.print_obs(training_data['input'][i][0]['image'])
#         env.unwrapped.print_obs(training_data['input'][i][1]['image'])
#         print("ranking:", training_data['llm_target'][i])
#         print("GT ranking:", training_data['target'][i])
print("llm ranking accu:", accu / len(training_data['input']))
print("{} rows of input and {} rows of output.".format(len(training_data["input"]), len(training_data["target"])))