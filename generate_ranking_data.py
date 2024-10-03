# from environment.multigrid.multigrid_cooperation import MultiGridCooperation, ActionsReduced
# import environment.multigrid.gym_wrapper
# from environment.multigrid.gym_multigrid.multigrid import World, DIR_TO_VEC
from environment.minigrid.minigrid import MiniGridEnv
from environment.grid.doubledoor import DoubleDoorEnv
from environment.grid.wrappers import ActionMasking, FullyObsWrapper
from environment.minigrid.minigrid import MiniGridEnv, OBJECT_TO_IDX
from enum import Enum
import numpy as np
import pickle, time
from PIL import Image
from LLM.apis import decode_obs, decode_llm_msg, value_est, read_pkl
from copy import deepcopy
import argparse
import os

class Ranking(Enum):
    GREATER = 0
    LESSER = 1
    EQUAL = 2

parser = argparse.ArgumentParser()
parser.add_argument(
    "--use_llm",
    type=bool,
    default=False,
    choices=[True, False],
    help="training data source"
)
parser.add_argument(
    "--sample_size",
    type=int,
    help="number of sampled state pairs"
)
parser.add_argument(
    "--filename",
    type=str,
    default='llm-ranking_data_train.pkl',
    help="name of the training database"
)


def render_env(env):
    rgb_frame = env.render(mode="rgb_array", highlight=False, tile_size=100)
    image = Image.fromarray(rgb_frame, "RGB")
    image.show()


def manhattan_distance(a, b):
    return np.abs(a - b).sum()

# Gets the current subgoal location as well as the action needed to achieve it
def get_subgoal(obs, env, agent_pos):
    first_door_room, i, second_door_room, j = env.unwrapped._first_door()
    first_key_pos = env.key_pos[i]
    first_door_pos = env.rooms[first_door_room].doorPos
    second_key_pos = env.key_pos[j]
    second_door_pos = env.rooms[second_door_room].doorPos

    if obs[first_key_pos[0], first_key_pos[1], 0] == OBJECT_TO_IDX["key"]:
        # The first key is not picked up yet
        goal = env.unwrapped.subgoal(agent_pos, first_key_pos)
        return goal, MiniGridEnv.Actions.pickup
    
    elif obs[first_door_pos[0], first_door_pos[1], 0] == OBJECT_TO_IDX["door"]:
        # The first door is not open
        goal = env.unwrapped.subgoal(agent_pos, first_door_pos)
        return goal, MiniGridEnv.Actions.forward
    
    elif obs[second_key_pos[0], second_key_pos[1], 0] == OBJECT_TO_IDX["key"]:
        # The second key is not picked up yet
        goal = env.unwrapped.subgoal(agent_pos, second_key_pos)
        return goal, MiniGridEnv.Actions.pickup
    
    elif obs[second_door_pos[0], second_door_pos[1], 0] == OBJECT_TO_IDX["door"]:
        # The second door is not open
        goal = env.unwrapped.subgoal(agent_pos, second_door_pos)
        return goal, MiniGridEnv.Actions.forward
    
    else:
        # Goal!
        goal = env.unwrapped.subgoal(agent_pos, env.goal_pos)
        return goal, MiniGridEnv.Actions.forward


# Returns true if obs2 is better (in a teamwork sense) than obs1.
# Should use only information in the observation to avoid leakage.
def rank_states(obs1, obs2, action, env):   
    if action == MiniGridEnv.Actions.pickup:
        # picking up a key is always good
        return Ranking.GREATER 
    if action == MiniGridEnv.Actions.toggle:#!
        # Agent is facing the door, thus toggle action is sampled
        # If the correct key is carried for the correct door
        first_door_room, i, second_door_room, j = env.unwrapped._first_door()
        if env.carrying != None:
            # print(env.carrying.color)
            agent_room, _ = env.unwrapped._find_room(env.agent_pos)
            if env.grid.get(*env.rooms[first_door_room].doorPos) is None \
            and env.carrying.color == env.rooms[first_door_room].color and agent_room == 2:
                return Ranking.GREATER
            if env.grid.get(*env.rooms[first_door_room].doorPos) is None and env.grid.get(*env.rooms[second_door_room].doorPos) is None \
            and env.carrying.color == env.rooms[second_door_room].color and agent_room == 5:
                return Ranking.GREATER
        return Ranking.EQUAL
    if obs2[env.goal_pos[0], env.goal_pos[1], 0] != OBJECT_TO_IDX["goal"]:
        # the goal is reached
        return Ranking.GREATER


    agent_pos_gt = env.agent_pos

    agent_location1 = np.where(obs1[:, :, 0] == OBJECT_TO_IDX["agent"])
    #print(agent_location1, end=' ')
    agent_pos1 = np.array((agent_location1[0][0], agent_location1[1][0]))
    # agent_ori1 = obs1[agent_pos1[0], agent_pos1[1], 2]
    #print(agent_ori1)

    agent_location2 = np.where(obs2[:, :, 0] == OBJECT_TO_IDX["agent"])
    #print(agent_location2, env.agents[agent_idx].dir)
    agent_pos2 = np.array((agent_location2[0][0], agent_location2[1][0]))
    # agent_ori2 = obs2[agent_pos2[0], agent_pos2[1], 2]

    # print("GT Pos: " + str(agent_pos_gt))
    # print("Pos 1: " + str(agent_pos1))
    # print("Pos 2: " + str(agent_pos2))

    assert(np.array_equal(agent_pos_gt, agent_pos2))

    # Find the current subgoal
    subgoal_pos1, subgoal_action1 = get_subgoal(obs1, env, agent_pos1)
    subgoal_pos2, subgoal_action2 = get_subgoal(obs2, env, agent_pos2)

    dist1 = manhattan_distance(agent_pos1, subgoal_pos1)
    dist2 = manhattan_distance(agent_pos2, subgoal_pos2)

    if not np.array_equal(subgoal_pos1, subgoal_pos2):
        # agent took the forward action and the subgoal changed
        if subgoal_pos1 in list([env.goal_pos, env.key_pos[0], env.key_pos[1]]):
            # subgoal1 is the key or goal, which agent can directly reach without passing through doors 
            # but subgoal2 changed (to hallway) after forward, bad case
            return Ranking.LESSER
        elif subgoal_pos2 in list([env.goal_pos, env.key_pos[0], env.key_pos[1]]):
            # after forward, subgoal changed from hallway to key / goal
            # the agent took a correct action
            return Ranking.GREATER
        else:
            # subgoal1 and subgoal 2 are both hallway
            # situation 1: agent should navigate to the key room hallway
            # situation 2: key is picked up, door is still locked, agent navigate door
            # situation 3: agent navigate to the goal room hallway
            # In any situation, the grand goal is never previous hallways
            agent_room, _ = env.unwrapped._find_room(agent_pos1)
            if agent_room % 2 == 0:
                # agent is in the upper rooms, it should go right or down anyway to reach the next door / hallway
                if subgoal_pos2[1] > subgoal_pos1[1]:
                    # the agent is right at hallway from room 4 to 5, or at hallway 2 to 4
                    return Ranking.GREATER
                elif subgoal_pos2[1] == subgoal_pos1[1]:
                    # the agent is going from room 0 to 2
                    if subgoal_pos2[0] > subgoal_pos1[0]:
                        return Ranking.GREATER
                    else:
                        return Ranking.LESSER
                else:
                    return Ranking.LESSER
            else:
                # agent is in the lower rooms, it should go left anyway
                if subgoal_pos2[0] > subgoal_pos1[0]:
                    return Ranking.LESSER
                else:
                    return Ranking.GREATER

    else:
        if action == MiniGridEnv.Actions.forward:
            # TODO
            # If the distance is less, then the new state is better
            if dist2 < dist1:
                return Ranking.GREATER
            
            # If the distance is greater, then the new state is worse
            if dist2 > dist1:
                return Ranking.LESSER
    
    return Ranking.EQUAL

def llm_rank(env, step_idx, obs, base_prompt, lst_obs=None, lst_pos=None):

    q = decode_obs(env, lst_pos, step_idx, lst_obs, obs)
    complete_q = base_prompt + q
    print(q)

    while True:
        #print(complete_q)

        try:
            llm_val = decode_llm_msg(value_est(complete_q))
        except:
            # pass
            continue
        if llm_val == False:
            # pass
            continue
        break
    # llm_val = False
    return llm_val


def collect_ranking_data_llm(env, filepath, llm_filepath, sample_size):
    obs_list, new_obs_list, valid_action_list, lst_pos_list, new_pos_list, key_pos_list, goal_pos_list, gt_rank_list = read_pkl(filepath)

    training_data = {"input": [], "target": [], "llm_target": []}

    with open(os.getcwd() + '/LLM/prompts/multi_lock.txt', "r") as file:
        sys_prompt = file.read()
    n = len(obs_list)
    llm_accu = 0
    pick_up_case = 0
    for step_idx in range(n):
        obs = obs_list[step_idx]
        new_obs = new_obs_list[step_idx]
        valid_action = valid_action_list[step_idx]
        lst_pos = lst_pos_list[step_idx]
        env.agent_pos = new_pos_list[step_idx]
        env.key_pos = key_pos_list[step_idx]
        env.goal_pos = goal_pos_list[step_idx]
        ranking = gt_rank_list[step_idx]
        print('GT', ranking)
        if ranking != Ranking.EQUAL:
            print('---------------------- ranking ------------------', step_idx)
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
        if step_idx % 25 == 0:
            pickle.dump(training_data, open(llm_filepath, "wb"))
            print('----------------------------')
            print('checkpoint! cur length of data file:')
            print(len(training_data["llm_target"]))
            print('llm accuracy', llm_accu / (step_idx+1))
        if step_idx >= sample_size:
            pickle.dump(training_data, open(llm_filepath, "wb"))
            print('----------------------------')
            print('checkpoint! cur length of data file:')
            print(len(training_data["llm_target"]))
            print('llm accuracy', llm_accu / (step_idx+1))
            return training_data
    #         if valid_action == MiniGridEnv.Actions.pickup:
    #             pick_up_case += 1
    # print("pick up case: ", pick_up_case)
    pickle.dump(training_data, open(llm_filepath, "wb"))
    return training_data


def generate_training_data(env, num_samples = 1, llm=False):
    lst_pos = [None, None]

    special_data_num = 0
    equal_num = 0


    training_data = {"input": [], "target": [], "llm_target": []}

    with open(os.getcwd() + '/LLM/prompts/stupid.txt', "r") as file:
        sys_prompt = file.read()

    step_idx = 0
    # for step_idx in range(num_samples):
    while True:
        picking_red_key = False
        opening_door = False
        passing_door = False

        obs = env.reset()
        obs = env.observation(env.randomize_state())
        # env.render(pause = 0.2)
        lst_pos = [env.agent_pos[0], env.agent_pos[1]]
        # print(obs)

        if obs["action_mask"][MiniGridEnv.Actions.pickup] != 0:
            valid_action = MiniGridEnv.Actions.pickup
            # print("key")
        elif obs["action_mask"][MiniGridEnv.Actions.toggle] != 0:
            valid_action = MiniGridEnv.Actions.toggle
            # print("door")
        else:
            valid_action = MiniGridEnv.Actions.forward

        new_obs, _, _, _ = env.step(valid_action)

        # if env.grid.get(*env.goal_pos) is None:
            # print("goal")
        #     special_data_num += 1

        # env.render(pause = 0.2)

            
        ranking = rank_states(obs["image"], new_obs["image"], valid_action, env)

        if llm and ranking != Ranking.EQUAL:
            print('---------------------- ranking ------------------')
            env.unwrapped.print_obs(obs["image"])
            env.unwrapped.print_obs(new_obs["image"])
            llm_val = llm_rank(env, step_idx, new_obs, sys_prompt, lst_obs=obs, lst_pos=lst_pos)
            # print("\n\n\nbefore appending", training_data["llm_target"], llm_val)
            training_data["llm_target"].append(llm_val)
            print(llm_val)
            # print("\n\n\nafter appending", training_data["llm_target"], llm_val)

        if ranking != Ranking.EQUAL:
            obs["image"][lst_pos[0], lst_pos[1], 2] = 0
            new_obs["image"][env.agent_pos[0], env.agent_pos[1], 2] = 0
            if valid_action == MiniGridEnv.Actions.toggle or env.grid.get(*env.goal_pos) is None or valid_action == MiniGridEnv.Actions.pickup:
                special_data_num += 1
            # time.sleep(3)
            training_data["input"].append([obs, new_obs, valid_action])
            training_data["target"].append(ranking)
            step_idx = len(training_data["input"])
        else:
            equal_num += 1

        
        # print(ranking)
        
        print(ranking)
        print('\n\n')

        if step_idx >= num_samples:
            break

    print(special_data_num)
    print(equal_num)
    return training_data


def create_training_data(args):
    env = DoubleDoorEnv({})
    env = ActionMasking(FullyObsWrapper(env))

    # for step_id in range(1000):
    #     print(step_id * 5)
    filename = args.filename
    if args.use_llm:
        orifile_name = "ranking_data6k.pkl"
        data = collect_ranking_data_llm(env, orifile_name, args.filename, args.sample_size)
    else:
        data = generate_training_data(env, num_samples=args.sample_size, llm=args.use_llm)
    pickle.dump(data, open(filename, "wb"))


def validate_training_data(filename="ranking_data_train_2.pkl"):
    training_data = pickle.load(open(filename, "rb"))

    print("{} rows of input and {} rows of output.".format(len(training_data["input"]), len(training_data["target"])))
        #print(training_data[agent_idx]["llm_target"])


if __name__ == "__main__":
    args = parser.parse_args()
    print(args.use_llm)
    create_training_data(args)
    validate_training_data(args.filename)

