from environment.grid.doubledoor import DoubleDoorEnv
from environment.grid.wrappers import ActionMasking, FullyObsWrapper
from environment.minigrid.minigrid import MiniGridEnv, OBJECT_TO_IDX
import numpy as np
import pickle
import os
from enum import Enum
from LLM.apis import read_pkl
from generate_ranking_data import llm_rank
import argparse

parser = argparse.ArgumentParser()
parser.add_argument(
    "--mode",
    type=str,
    help = "'cc' for remove, 'ccc' for correct, 'cccr' for remove"
)
parser.add_argument(
    "--threshold",
    type = float,
)

class Ranking(Enum):
    GREATER = 0
    LESSER = 1
    EQUAL = 2


def consistent_check_remove(env, training_data, args):
    mode = args.mode
    thres = args.threshold
    data_pair_dict = {}
    guilty_family = []
    cc_case = 0
    reverse_guilty_case = 0
    same_guilty_case = 0
    
    for i, [obs1, obs2, _] in enumerate(training_data['input']):
        sum_obs = obs1['image'] + obs2['image']
        obs_tuple = tuple(sum_obs.flatten())

        if obs_tuple in data_pair_dict:
            # if the obs_tuple has been met before
            # which means the new seen data pair is on reverse / same case of a previous data
            # store its idx into the list of this data pair dict, for future family removal
            cc_case += 1
            data_pair_dict[obs_tuple][3].append(i)
            if np.array_equal(obs1['image'], data_pair_dict[obs_tuple][0]) and \
                str(training_data['llm_target'][i]) == str(data_pair_dict[obs_tuple][1]):
                # obs1 == the obs2 of that previous data. reverse case
                # Error: producing same rank
                # push the data pair family into a guilty list
                reverse_guilty_case += 1
                guilty_family.append(data_pair_dict[obs_tuple][2])
            elif (not np.array_equal(obs1['image'], data_pair_dict[obs_tuple][0])) and \
            str(training_data['llm_target'][i]) != str(data_pair_dict[obs_tuple][1]):
                # obs1 == the obs1 of that previous data. same case
                # Error: producing a different rank
                # also push the data pair into a guilty list
                same_guilty_case += 1
                guilty_family.append(data_pair_dict[obs_tuple][2])
        else:
            # the data pair is first met
            # push the data pair into the dict
            # together with other information about second obs, llm target, index,
            # and a list to collect future data that have same obs1, obs2 with this data
            data_pair_dict[obs_tuple] = (obs2['image'], training_data['llm_target'][i], i, [])

    guilty_family = set(guilty_family)
    guilty_data = []
    # find all data index in the relative list of the guilty data pair
    # push them into a list for removal
    for guilt in guilty_family:
        # selectively remove the data
        # Assume guilt holds the correct answer.
        # If more than 70% agree or disagree with the answer, keep the family
        # Else throw away
        obs1_root = training_data['input'][guilt][0]
        obs2_root = training_data['input'][guilt][1]
        
        obs_tuple = tuple((obs1_root['image']+obs2_root['image']).flatten())
        ans = training_data['llm_target'][guilt]
        agree = 1
        agree_list = []
        agree_list.append(guilt)
        disagree = 0
        disagree_list = []
        for data_idx in data_pair_dict[obs_tuple][3]:
            obs1 = training_data['input'][data_idx][0]
            obs2 = training_data['input'][data_idx][1]
            if np.array_equal(obs1['image'], obs1_root['image']):
                # case same
                if str(training_data['llm_target'][data_idx]) == str(ans):
                    agree += 1
                    agree_list.append(data_idx)
                else:
                    disagree += 1
                    disagree_list.append(data_idx)
            else:
                # reverse case
                if str(training_data['llm_target'][data_idx]) == str(ans):
                    disagree += 1
                else:
                    agree += 1
        votes = len(data_pair_dict[obs_tuple][3]) + 1
        
        if mode == 'cc':
            if agree / votes > thres or disagree / votes > thres:
                pass
            else:
                # kill the whole family
                # pass
                guilty_data.append(guilt)
                for data_idx in data_pair_dict[obs_tuple][3]:
                    guilty_data.append(data_idx)
        elif mode == 'ccc':
            if agree / votes >= thres:
                for i in disagree_list:
                    training_data['llm_target'][i] = ans
            elif disagree / votes >= thres:
                if str(ans) == Ranking.GREATER:
                    ans_ = Ranking.LESSER
                else:
                    ans_ = Ranking.GREATER
                for i in agree_list:
                    training_data['llm_target'][i] = ans_
            # env.print_obs(obs1['image'])
            # env.print_obs(obs2['image'])
        else:
            if agree / votes >= thres:
                for i in disagree_list:
                    training_data['llm_target'][i] = ans
            elif disagree / votes >= thres:
                if str(ans) == Ranking.GREATER:
                    ans_ = Ranking.LESSER
                else:
                    ans_ = Ranking.GREATER
                for i in agree_list:
                    training_data['llm_target'][i] = ans_
            else:
                # kill the whole family
                # pass
                guilty_data.append(guilt)
                for data_idx in data_pair_dict[obs_tuple][3]:
                    guilty_data.append(data_idx)
    
    guilty_data = set(guilty_data)
    new_training_data = {'input':[], 'target':[], 'llm_target':[]}
    new_size = 0
    for i in range(len(training_data['input'])):
        if i not in guilty_data:
            new_training_data['input'].append(training_data['input'][i])
            new_training_data['target'].append(training_data['target'][i])
            new_training_data['llm_target'].append(training_data['llm_target'][i])
            new_size += 1

    print("consistency check case:", cc_case)
    print("reverse guilty case:", reverse_guilty_case)
    print("same guilty case:", same_guilty_case)
    print("new training data size:", new_size)

    return new_training_data


def generate_cc_dataset(training_data):
    data_pair_dict = {}
    cc_case = 0
    cc_success_family = []
    
    for i, [obs1, obs2, _] in enumerate(training_data['input']):
        sum_obs = obs1['image'] + obs2['image']
        obs_tuple = tuple(sum_obs.flatten())

        if obs_tuple in data_pair_dict:
            cc_case += 1
            data_pair_dict[obs_tuple][3].append(i)
            if np.array_equal(obs1['image'], data_pair_dict[obs_tuple][0]):
                # reverse case already seen!!
                # cc naturally completed. No further check needed
                cc_success_family.append(i)
                cc_success_family.append(data_pair_dict[obs_tuple][2])
        else:
            data_pair_dict[obs_tuple] = (obs2['image'], training_data['llm_target'][i], i, [])
    cc_success_family = set(cc_success_family)
    cc_success_data = []
    for success in cc_success_family:
        obs1 = training_data['input'][success][0]
        obs2 = training_data['input'][success][1]
        cc_success_data.append(success)
        obs_tuple = tuple((obs1['image']+obs2['image']).flatten())
        # print(len(data_pair_dict[obs_tuple][3]))
        # if len(data_pair_dict[obs_tuple][3]) == 11:
        #     print('--------------------------------')
        #     env.print_obs(obs1['image'])
        #     env.print_obs(obs2['image'])
        #     print("llm rank:", training_data['llm_target'][success])
        #     for i in range(len(data_pair_dict[obs_tuple][3])):
        #         idx = data_pair_dict[obs_tuple][3][i]
        #         env.print_obs(training_data['input'][idx][0]['image'])
        #         env.print_obs(training_data['input'][idx][1]['image'])
        #         print("llm rank:", training_data['llm_target'][idx])
        #     print('\n\n\n')
        for data_idx in data_pair_dict[obs_tuple][3]:
            cc_success_data.append(data_idx)
    for obs_tuple, (obs2, llm_target, idx, family) in data_pair_dict.items():
        if (not idx in cc_success_family) and len(family)>0:
            # The data pair doesn't have reverse pair, but have same pair
            # only use this single data pair for cc check
            for i in family:
                cc_success_data.append(i)
    cc_success_data = set(cc_success_data)
    new_training_data = {'input':[], 'target':[], 'llm_target':[]}
    for i in range(len(training_data['input'])):
        if i not in cc_success_data:
            new_training_data['input'].append(training_data['input'][i])
            new_training_data['target'].append(training_data['target'][i])
            new_training_data['llm_target'].append(training_data['llm_target'][i])
    
    return new_training_data


def generate_cc_llm_ranking(env, ccfile_path, llm_filepath):
    obs_list, new_obs_list, valid_action_list, lst_pos_list, new_pos_list, key_pos_list, goal_pos_list, gt_rank_list = read_pkl(ccfile_path)

    training_data = {"input": [], "target": [], "llm_target": []}

    with open(os.getcwd() + '/LLM/prompts/env3.txt', "r") as file:
        sys_prompt = file.read()
    n = len(obs_list)
    llm_accu = 0
    for step_idx in range(n):
        obs = obs_list[step_idx]
        new_obs = new_obs_list[step_idx]
        valid_action = valid_action_list[step_idx]
        lst_pos = lst_pos_list[step_idx]
        env.agent_pos = new_pos_list[step_idx]
        env.key_pos = key_pos_list[step_idx]
        env.goal_pos = goal_pos_list[step_idx]
        ranking = gt_rank_list[step_idx]
        print('---------------------- ranking ------------------', step_idx)
        if valid_action == MiniGridEnv.Actions.pickup or valid_action == MiniGridEnv.Actions.toggle or \
            new_obs['image'][env.goal_pos[0], env.goal_pos[1], 0] != OBJECT_TO_IDX['goal']:
            # cases where agent is picking up key, opening the door, or reaching the goal in this pair
            print("no reverse case, test again")
            env.unwrapped.print_obs(obs["image"])
            env.unwrapped.print_obs(new_obs["image"])
            llm_val = llm_rank(env, step_idx, new_obs, sys_prompt, lst_obs=obs, lst_pos=lst_pos)
            # print("\n\n\nbefore appending", training_data["llm_target"], llm_val)
            training_data["llm_target"].append(llm_val)
            training_data["input"].append([obs, new_obs, valid_action])
            training_data["target"].append(ranking)
            print(llm_val)
            print("GT rank:", ranking)
            if str(llm_val) == str(ranking):
                llm_accu += 1
        else:
            # there are reverse cases for this pair
            print("test reverse case")
            env.unwrapped.print_obs(new_obs["image"])
            env.unwrapped.print_obs(obs["image"])
            lst_pos = new_pos_list[step_idx]
            env.agent_pos = lst_pos_list[step_idx]
            llm_val = llm_rank(env, step_idx, obs, sys_prompt, lst_obs=new_obs, lst_pos=lst_pos)
            training_data["llm_target"].append(llm_val)
            training_data["input"].append([new_obs, obs, valid_action])
            if str(ranking) == str(Ranking.GREATER):
                reversed_ranking = Ranking.LESSER
            else:
                reversed_ranking = Ranking.GREATER
            training_data['target'].append(reversed_ranking)
            print(llm_val)
            print("GT rank:", reversed_ranking)
            if str(llm_val) == str(reversed_ranking):
                llm_accu += 1
        if step_idx % 25 == 0:
            pickle.dump(training_data, open(llm_filepath, "wb"))
            print('----------------------------')
            print('checkpoint! cur length of data file:')
            print(len(training_data["llm_target"]))
            print('llm accuracy', llm_accu / (step_idx+1))
    
    print('Finished! cur length of data file:')
    print(len(training_data["llm_target"]))
    print('llm accuracy', llm_accu / (step_idx+1))
    pickle.dump(training_data, open(llm_filepath, "wb"))
    return training_data

def consistent_check_single_case(env, training_data):
    data_pair_dict = {}
    guilty_family = []
    cc_case = 0
    reverse_guilty_case = 0
    same_guilty_case = 0
    
    for i, [obs1, obs2, _] in enumerate(training_data['input']):
        sum_obs = obs1['image'] + obs2['image']
        obs_tuple = tuple(sum_obs.flatten())

        if obs_tuple in data_pair_dict:
            # if the obs_tuple has been met before
            # which means the new seen data pair is on reverse / same case of a previous data
            # store its idx into the list of this data pair dict, for future family removal
            cc_case += 1
            data_pair_dict[obs_tuple][3].append(i)
        else:
            # the data pair is first met
            # push the data pair into the dict
            # together with other information about second obs, llm target, index,
            # and a list to collect future data that have same obs1, obs2 with this data
            data_pair_dict[obs_tuple] = (obs2['image'], training_data['llm_target'][i], i, [])

    single_case_list = []
    for obs_tuple, (obs2_image, llm_target, i, neighbor) in data_pair_dict.items():
        if len(neighbor) == 0:
            single_case_list.append(i)
    single_case_list = set(single_case_list)
    new_training_data = {'input':[], 'target':[], 'llm_target':[]}
    new_size = 0

    for i in range(len(training_data['input'])):
        if i not in single_case_list:
            new_training_data['input'].append(training_data['input'][i])
            new_training_data['target'].append(training_data['target'][i])
            new_training_data['llm_target'].append(training_data['llm_target'][i])
            new_size += 1
    print("new training data size:", new_size)

    return new_training_data


if __name__ == "__main__":
    args = parser.parse_args()
    old_data_file = 'ranking_data_train_env3-llama_family_cced.pkl'
    training_data_file = 'ranking_data_train_env3-llama_check.pkl'
    env = DoubleDoorEnv({})
    env = ActionMasking(FullyObsWrapper(env))
    old_data = pickle.load(open(old_data_file, "rb"))
    training_data = consistent_check_remove(env, old_data, args)
    pickle.dump(training_data, open(training_data_file, "wb"))
    # check data and accuracy
    training_data = pickle.load(open(training_data_file, "rb"))
    accu = 0
    obs_list, new_obs_list, valid_action_list, lst_pos_list, new_pos_list, key_pos_list, goal_pos_list, gt_rank_list = read_pkl(training_data_file)
    for i in range(len(training_data['input'])):
        agent_pos2 = new_pos_list[i]
        agent_pos1 = lst_pos_list[i]
        obs = obs_list[i]
        new_obs = new_obs_list[i]
        room2, is_room2 = env.unwrapped._find_room(agent_pos2)
        room1, is_room1 = env.unwrapped._find_room(agent_pos1)
        if str(training_data['target'][i]) == str(training_data['llm_target'][i]):
            accu += 1
        else:
            pass
    # print("accuracy for hallway {}".format(room3_hallway_acc / room3_hallway))
    print("llm ranking accu:", accu / len(training_data['input']))
    print("{} rows of input and {} rows of output.".format(len(training_data["input"]), len(training_data["target"])))

    # The following coded is appended when generating data to ask LLM again
    # new_training_data = generate_cc_dataset(training_data)
    # pickle.dump(new_training_data, open('ranking_data_train_env3-mixtral_cc.pkl', "wb"))

    # training_data = pickle.load(open('ranking_data_train_env3-mixtral_cc.pkl', "rb"))
    # accu = 0
    # for i in range(len(training_data['input'])):
    #     if str(training_data['target'][i]) == str(training_data['llm_target'][i]):
    #         accu += 1
    #     else:
    #         pass
    # print("llm ranking accu:", accu / len(training_data['input']))
    # print("{} rows of input and {} rows of output.".format(len(training_data["input"]), len(training_data["target"])))

    # cced_training_data = generate_cc_llm_ranking(env, 'ranking_data_train_env3-llama_single.pkl', 'ranking_data_train_env3-llama_single_cced.pkl')

    # training_data = pickle.load(open('ranking_data_train_env3-mixtral_cced.pkl', "rb"))
    # accu = 0
    # for i in range(len(training_data['input'])):
    #     if str(training_data['target'][i]) == str(training_data['llm_target'][i]):
    #         accu += 1
    #     else:
    #         pass
    # print("llm ranking accu:", accu / len(training_data['input']))
    # print("{} rows of input and {} rows of output.".format(len(training_data["input"]), len(training_data["target"])))
