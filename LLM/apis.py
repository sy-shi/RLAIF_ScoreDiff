from openai import OpenAI
import string, os
import numpy as np
from enum import Enum
from environment.minigrid.minigrid import OBJECT_TO_IDX
import pickle

class Ranking(Enum):
    GREATER = 0
    LESSER = 1
    EQUAL = 2

#from keys import API_KEY, ORG_KEY


def decode_obs(env, lst_pos, step_idx, lst_obs, obs):
    room, isroom = env.unwrapped._find_room(lst_pos)
    if room == 0:
        ag_room = 'Chamber1'
    elif room == 2:
        ag_room = 'Chamber2'
    elif room == 4 and isroom:
        ag_room = 'Chamber3'
    elif (room == 4 and not isroom) or room == 5:
        ag_room = 'Chamber4'
    elif room == 3:
        ag_room = 'Chamber5'
    else:
        ag_room = 'Chamber6'
    q = "Q:\nState[" + str(step_idx) + ']:\n'
    q += 'Agent: ' + ag_room + ' (' + str(lst_pos[0] - 1) + ',' + str(7 - lst_pos[1]) + ')\n'
    q += 'Passage to Chamber2: Chamber1 (3,5) right\n'

    q += 'Door at Chamber2 (7,5) to Chamber3: '
    if lst_obs["image"][env.rooms[2].doorPos[0], env.rooms[2].doorPos[1], 0] == OBJECT_TO_IDX["door"]:
        q += 'locked\n'
        if lst_obs["image"][env.key_pos[0][0], env.key_pos[0][1], 0] == OBJECT_TO_IDX["key"]:
            q += 'Key in Chamber2 (' + str(env.key_pos[0][0] - 1) + ',' + str(7 - env.key_pos[0][1]) + ')\n'
            description = 'The agent does not carry any key. It needs a key.\n'
            if obs["image"][env.key_pos[0][0], env.key_pos[0][1], 0] != OBJECT_TO_IDX["key"]:
                action = 'pick up'
        else:
            description = 'The agent carries the key in Chamber2. It can open the door when it is next to the door.\n'
        if obs["image"][env.rooms[2].doorPos[0], env.rooms[2].doorPos[1], 0] != OBJECT_TO_IDX["door"]:
            action = 'open the door'
    else:
        q += 'open\n'
        description = 'The agent does not carry any key. It needs a key.\n'

    q += 'Passage to Chamber4: Chamber4 (9,3) down\n'

    q += 'Door at Chamber4 (7,1) to Chamber5: '
    if lst_obs["image"][env.rooms[3].doorPos[0], env.rooms[3].doorPos[1], 0] == OBJECT_TO_IDX["door"]:
        q += 'locked\n'
        if lst_obs["image"][env.key_pos[1][0], env.key_pos[1][1], 0] == OBJECT_TO_IDX["key"]:
            q += 'Key in Chamber4 (' + str(env.key_pos[1][0] - 1) + ',' + str(7 - env.key_pos[1][1]) + ')\n'
            # if not description == 'The agent carries the key in Chamber2. It can open the door when it is next to the door.\n':
            # if not lst_obs["image"][env.rooms[2].doorPos[0], env.rooms[2].doorPos[1], 0] != OBJECT_TO_IDX["door"]:
            #     description = 'The agent does not carry any key. It needs a key.\n'
            if obs["image"][env.key_pos[1][0], env.key_pos[1][1], 0] != OBJECT_TO_IDX["key"]:
                action = 'pick up'
        else:
            description = 'The agent carries the key in Chamber4. It can open the door when it is next to the door.\n'
        if obs["image"][env.rooms[3].doorPos[0], env.rooms[3].doorPos[1], 0] != OBJECT_TO_IDX["door"]:
            action = 'open the door'
    else:
        q += 'open\n'
        description = None

    q += 'Passage to Chamber6: Chamber1 (3,1) left\n'
    q += 'Clinic: Chamber6 (' + str(env.goal_pos[0] - 1) + ',' + str(7 - env.goal_pos[1]) + ')\n\n'

    if description is not None:
        q += description
    q += 'Agent action: '
    if env.agent_pos[0] - lst_pos[0] > 0:
        q += 'move right'
    elif env.agent_pos[0] - lst_pos[0] < 0:
        q += 'move left'
    elif env.agent_pos[1] - lst_pos[1] < 0:
        q += 'move up'
    elif env.agent_pos[1] - lst_pos[1] > 0:
        q += 'move down'
    else:
        q += action

    q += "\n\nDoes the action taken by the Agent in State[" + str(step_idx) + "] help it progress towards the Clinic?\n\nA:\nLet's think step by step."

    return q

def decode_llm_msg(msg):
    #print(msg)
    lines = msg.split('\n')
    punctuation = string.punctuation
    lines = [line.strip(punctuation + string.whitespace) for line in lines]
    #lines[1] = lines[1].strip(punctuation + ' ')

    rank = False
    for line in lines:
        # if line[-3:].lower() == 'yes':
        #     rank = Ranking.GREATER
        # elif line[-2:].lower() == 'no':
        #     rank = Ranking.LESSER
        if "the answer is yes" in line.lower():
            rank = Ranking.GREATER
        elif "the answer is no" in line.lower():
            rank = Ranking.LESSER
    return rank


def value_est(q, type_check = False, prompts = None, anss = None):
    request = [
        {"role": "user", "content": q}
    ]

    client = OpenAI(
        base_url = 'http://localhost:11434/v1',
        api_key='ollama', # required, but unused
    )
    #client = OpenAI(api_key=API_KEY, organization=ORG_KEY, base_url="http://localhost:23002/v1")

    # create a chat completion
    print('prepared to ask')
    completion = client.chat.completions.create(
        model='llama3:70b',#'mixtral',#"Llama-2-13b-chat-hf", #'mixtral',#"Starling-LM-7B-alpha",#"gpt-4-1106-preview",
        messages=request,
        temperature=0.5,
        max_tokens=400
    )
    print(completion.choices[0].message.content)
    return completion.choices[0].message.content


def read_pkl(file_path):
    with open(file_path, 'rb') as file:
        data = pickle.load(file)
    input_pairs = data['input']  # List of [obs, new_obs]
    gt_rank_list = data['target']     # List of integers
    obs_list = []
    new_obs_list = []
    valid_action_list = []
    lst_pos_list = []
    new_pos_list = []
    key_pos_list = []
    goal_pos_list = []
    for obs, new_obs, valid_action in input_pairs:
        obs_list.append(obs)
        new_obs_list.append(new_obs)
        valid_action_list.append(valid_action)
        lst_pos = None
        new_pos = None
        lst_key_pos = [None, None]
        new_key_pos = [None, None]
        key_pos = [None, None]
        for i in range(obs['image'].shape[0]):
            for j in range(obs['image'].shape[1]):
                if obs['image'][i,j,0] == OBJECT_TO_IDX['agent']:
                    lst_pos = (i,j)
                    lst_pos_list.append(lst_pos)
                elif obs['image'][i,j,0] == OBJECT_TO_IDX['key']:
                    if i<8:
                        lst_key_pos[0] = (i,j)
                    else:
                        lst_key_pos[1] = (i,j)
                elif obs['image'][i,j,0] == OBJECT_TO_IDX['goal']:
                    goal_pos_list.append((i,j))

        for i in range(new_obs['image'].shape[0]):
            for j in range(new_obs['image'].shape[1]):
                if new_obs['image'][i,j,0] == OBJECT_TO_IDX['agent']:
                    new_pos = (i,j)
                    new_pos_list.append(new_pos)
                elif new_obs['image'][i,j,0] == OBJECT_TO_IDX['key']:
                    if i<8:
                        new_key_pos[0] = (i,j)
                    else:
                        new_key_pos[1] = (i,j)
        
        for i in range(2):
            if lst_key_pos[i] is not None or new_key_pos[i] is not None:
                key_pos[i] = lst_key_pos[i] if lst_key_pos[i] is not None else new_key_pos[i]
            else:
                key_pos[i] = (0,0)
        key_pos_list.append(key_pos)
    
    return obs_list, new_obs_list, valid_action_list, lst_pos_list, new_pos_list, key_pos_list, goal_pos_list, gt_rank_list 