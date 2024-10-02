from __future__ import annotations
import numpy as np
import time

from environment.minigrid.minigrid import IDX_TO_COLOR, OBJECT_TO_IDX, COLOR_TO_IDX, Door, Goal, Key, Wall, Grid, COLOR_NAMES, MiniGridEnv, DIR_TO_VEC
from gym_minigrid.minigrid import MissionSpace

class LockedRoom:
    def __init__(self, top, size, doorPos):
        self.top = top
        self.size = size
        self.doorPos = doorPos
        self.color = None
        self.locked = False

    def rand_pos(self, env):
        topX, topY = self.top
        sizeX, sizeY = self.size
        return env._rand_pos(topX + 1, topX + sizeX - 1, topY + 1, topY + sizeY - 1)


class DoubleDoorEnv(MiniGridEnv):
    # """
    # Empty grid environment, no obstacles, sparse reward
    # """

    # def __init__(
    #     self,
    #     config,
    #     size=9,
    #     max_steps = 200,
    #     agent_start_pos=(1,1),
    #     agent_start_dir=0,
    # ):
    #     self.agent_start_pos = agent_start_pos
    #     self.agent_start_dir = agent_start_dir

    #     super().__init__(
    #         grid_size=size,
    #         max_steps=max_steps,
    #         # Set this to True for maximum speed
    #         see_through_walls=True
    #     )

    # def _gen_grid(self, width, height):
    #     # Create an empty grid
    #     self.grid = Grid(width, height)

    #     # Generate the surrounding walls
    #     self.grid.wall_rect(0, 0, width, height)

    #     # Place a goal square in the bottom-right corner
    #     # self.grid.set(width - 2, height - 2, Goal())

    #     # Place the agent
    #     self.place_obj(Goal(), top=(0,0), size=(8,8))
    #     self.place_agent(top=(0,0), size=(8,8))

    #     self.mission = "get to the green goal square"

    # def render(self, pause = 0.01):
    #     r = super().render()
    #     time.sleep(pause)
    #     return r.getArray()
    
    
    """
    ## Description

    The environment has six rooms, one of which is locked. The agent receives
    a textual mission string as input, telling it which room to go to in order
    to get the key that opens the locked room. It then has to go into the locked
    room in order to reach the final goal. This environment is extremely
    difficult to solve with vanilla reinforcement learning alone.

    ## Mission Space

    "get the {lockedroom_color} key from the {keyroom_color} room, unlock the {door_color} door and go to the goal"

    {lockedroom_color}, {keyroom_color}, and {door_color} can be "red", "green",
    "blue", "purple", "yellow" or "grey".

    ## Action Space

    | Num | Name         | Action                    |
    |-----|--------------|---------------------------|
    | 0   | left         | Turn left                 |
    | 1   | right        | Turn right                |
    | 2   | forward      | Move forward              |
    | 3   | pickup       | Pick up an object         |
    | 4   | drop         | Unused                    |
    | 5   | toggle       | Toggle/activate an object |
    | 6   | done         | Unused                    |

    ## Observation Encoding

    - Each tile is encoded as a 3 dimensional tuple:
        `(OBJECT_IDX, COLOR_IDX, STATE)`
    - `OBJECT_TO_IDX` and `COLOR_TO_IDX` mapping can be found in
        [minigrid/core/constants.py](minigrid/core/constants.py)
    - `STATE` refers to the door state with 0=open, 1=closed and 2=locked

    ## Rewards

    A reward of '1 - 0.9 * (step_count / max_steps)' is given for success, and '0' for failure.

    ## Termination

    The episode ends if any one of the following conditions is met:

    1. The agent reaches the goal.
    2. Timeout (see `max_steps`).

    ## Registered Configurations

    - `MiniGrid-LockedRoom-v0`

    """

    def __init__(self, config, size=19, max_steps = 500, **kwargs):
        self.size = size

        mission_space = MissionSpace(
            mission_func=lambda color, type: f"Unused",
            ordered_placeholders=[COLOR_NAMES, ["box", "key"]],
        )
        if "random_init" in config:
            self.random_init = config.get("random_init", True)
        else:
            self.random_init = True
        
        super().__init__(
            width=13,
            height=9,
            max_steps=max_steps,
            # render_mode = "human",
            **kwargs,
        )
        self.partial_obs = False


    def _gen_grid(self, width, height):
        # Create the grid
        self.grid = Grid(width, height)

        # Generate the surrounding walls
        for i in range(0, width):
            self.grid.set(i, 0, Wall())
            self.grid.set(i, height - 1, Wall())
        for j in range(0, height):
            self.grid.set(0, j, Wall())
            self.grid.set(width - 1, j, Wall())

        # vertical room walls
        lWallIdx = width // 2 - 2
        rWallIdx = width // 2 + 2
        for j in range(0, height):
            self.grid.set(lWallIdx, j, Wall())
            self.grid.set(rWallIdx, j, Wall())

        self.rooms = []

        # horizontal room walls and generate 6 room layout
        #  ==  ==  ==
        #|| 0|| 2|| 4||
        #  ==  ==  ==
        #|| 1|| 3|| 5||
        #  ==  ==  == 
        # room 0, 1, 2, 3 with doorPos on its righthand side
        # room 4 with doorPos at the bottom of the room
        # room 5 has no door
        j = height // 2
        for i in range(1, width):
            self.grid.set(i, j, Wall())
        for i in range(0,3):
            if i < 2:
                self.rooms.append(LockedRoom((4*i, 0), (5,5), (4*(i+1), 2)))
                self.rooms.append(LockedRoom((4*i, 4), (5,5), (4*(i+1), 6)))
            else:
                self.rooms.append(LockedRoom((4*i, 0), (5,5), (4*i+2, 4)))
                self.rooms.append(LockedRoom((4*i, 4), (5,5), None))
            

        # Choose two random room to be locked
        # self.lockedrooms = np.random.choice([0,1,2,3,4], 2, replace=False)
        self.lockedrooms = [2,3]
        for i in range(0, 2):
            self.rooms[self.lockedrooms[i]].locked = True
        goalPos = self.rooms[1].rand_pos(self)
        if not self.random_init:
            goalPos = (1,5)
        self.goal_pos = goalPos
        
        # place goal at room 1
        self.grid.set(*goalPos, Goal())

        # Assign the door colors
        colors = set(COLOR_NAMES)
        # colors = ['red', 'green', '_grey', 'purple', 'blue', 'yellow']
        i = 0
        for room in self.rooms:
            color = self._rand_elem(sorted(colors))
            colors.remove(color)
            
            # print(room)
            if room.locked: 
                # print(colors)
                # color = colors[0]
                if i == 0:
                    color = 'red'
                    i += 1
                else:
                    color = 'green'
                room.color = color
                self.grid.set(*room.doorPos, Door(color, is_locked=True))
                # self.grid.set(*room.doorPos, None)
                # colors.pop(0)
            else:
                if room.doorPos:
                    self.grid.set(*room.doorPos, None)

        # put the keys
        self.key_pos = []
        for i in range(2):
            if self.lockedrooms[i] % 2 == 0:
                # if the locked door is in upper rooms
                # put the key into the room of the door
                keyRoom = self.rooms[self.lockedrooms[i]]
            else:
                # if the locked door is in lower rooms
                # put the key into room i + 2
                keyRoom = self.rooms[self.lockedrooms[i]+2]
            keyPos = keyRoom.rand_pos(self)
            if not self.random_init:
                keyPos = (5,1) if i == 0 else (10,7)
            self.key_pos.append(keyPos)
            self.grid.set(*keyPos, Key(self.rooms[self.lockedrooms[i]].color))

        # agent starts at room 0
        self.agent_pos = self.place_agent(
            top=(0, 0), size=(4, 4)
        )

        # Generate the mission string
        self.mission = (
            "double door"
        )

    
    def randomize_state(self):
        situation = np.random.rand()
        situation = 0
        # situation 0: both doors are open
        subgoal = None
        if situation < 0.33:
            subgoal = 0
            for i in range(2):

                # Generate doors as unlocked
                self.grid.set(*self.rooms[self.lockedrooms[i]].doorPos, None)
                # keys are all picked up
                self.grid.set(*self.key_pos[i], None)

            # remove the agent for random sample
            self.grid.set(*self.agent_pos, None)

            if not np.array_equal(self.goal_pos, (DIR_TO_VEC[2] + self.rooms[1].doorPos)):
                # goal is not behind the door
                upper_half = np.random.rand()
                if upper_half <= 0.3:
                    # agent in upper half of the environment
                    self.place_agent(top=(0,0), size=(12, 4))
                else:
                    # agent in lower half of the environment
                    self.place_agent(top=(0,4), size=(12,4))
            else:
                upper_half = np.random.rand()
                if upper_half <= 0.3:
                    # agent in upper half of the environment
                    self.place_agent(top=(0,0), size=(12, 4))
                else:
                    # agent in lower half of the environment
                    self.place_agent(top=(4,4), size=(8,4))
        

        # situation 1: the first door is open, while the second is locked
        elif situation >= 0.33 and situation < 0.66:
            # generate the first door as unlocked
            first_door_room, i, second_door_room, j = self._first_door()
            self.grid.set(*self.rooms[first_door_room].doorPos, None)
            # key of the first door is picked up
            self.grid.set(*self.key_pos[i], None)
            # remove the agent for random sample
            self.grid.set(*self.agent_pos, None)

            second_key_picked_up = np.random.rand()
            if second_key_picked_up < 0.45:
                subgoal = 1
                # if the second key is picked up
                self.carrying = self.grid.get(*self.key_pos[j])
                self.grid.set(*self.key_pos[j], None)
                if second_door_room % 2 == 0:
                    # the second door is in the upper half of the room
                    top = (0, 0)
                    size = ((second_door_room/2 + 1)* 4, 4)
                    self.place_agent(top = top, size = size)
                else:
                    # the second door is in the lower half of the room
                    if second_door_room == 1:
                        # The agent can be in rooms other than the goal room
                        upper_half = np.random.rand()
                        if upper_half <= 0.6:
                            # agent in upper half of the environment
                            self.place_agent(top=(0,0), size=(12, 4))
                        else:
                            # agent in lower half of the environment
                            self.place_agent(top=(4,4), size=(8,4))
                    else:
                        # The agent can be in upper rooms or the right most room
                        upper_half = np.random.rand()
                        if upper_half <= 0.62:
                            # agent in upper half of the environment
                            self.place_agent(top=(0,0), size=(12, 4))
                        else:
                            # agent in lower half of the environment
                            self.place_agent(top=(8,4), size=(4,4))
            else:
                subgoal = 2
                # the second key is not picked up
                if second_door_room % 2 == 0:
                    # the second door is in the upper half of the room
                    top = (0, 0)
                    if not np.array_equal(self.key_pos[j], (DIR_TO_VEC[0]+self.rooms[second_door_room-2].doorPos)):
                        # the key of the second door is not blocking the hallway from the previous room
                        size = ((second_door_room/2 + 1)* 4, 4)
                        self.place_agent(top = top, size = size)
                    else:
                        # the key of the second door is blocking the hallway from the previous room
                        size = (second_door_room*2+1, 4)
                        self.place_agent(top = top, size = size)
                else:
                    # the second door is in the lower half of the room
                    if second_door_room == 1:
                        # The agent can be in rooms other than the goal room
                        if not np.array_equal(self.key_pos[j], (DIR_TO_VEC[2]+self.rooms[3].doorPos)):
                            # The key is blocking the hallway from room 5 to room 3
                            upper_half = np.random.rand()
                            if upper_half <= 0.6:
                                # agent in upper half of the environment
                                self.place_agent(top=(0,0), size=(12, 4))
                            else:
                                # agent in lower half of the environment
                                self.place_agent(top=(4,4), size=(8,4))
                        else:
                            # The key is not blocking the hallway from room 5 to room 3
                            upper_half = np.random.rand()
                            if upper_half <= 0.62:
                                # agent in upper half of the environment
                                self.place_agent(top=(0,0), size=(12, 4))
                            else:
                                # agent in lower half of the environment
                                self.place_agent(top=(8,4), size=(4,4))
                    else:
                        # The agent can be in upper rooms or the right most room
                        if not np.array_equal(self.key_pos[j], (DIR_TO_VEC[1]+self.rooms[4].doorPos)):
                            # The key is not blocking the hallway from room 4 to room 5
                            upper_half = np.random.rand()
                            if upper_half <= 0.62:
                                # agent in upper half of the environment
                                self.place_agent(top=(0,0), size=(12, 4))
                            else:
                                # agent in lower half of the environment
                                self.place_agent(top=(8,4), size=(4,4))
                        else:
                            # The key is blocking the hallway from room 4 to room 5
                            self.place_agent(top=(0,0), size = (12,5))
            

        # situation 2: both doors are locked
        else:
            # find the first door
            first_door_room, i, second_door_room, j = self._first_door()
            # remove the agent for random sample
            self.grid.set(*self.agent_pos, None)

            first_key_picked_up = np.random.rand()
            if first_key_picked_up < 0.45:
                subgoal = 3
                # if the first key is pickedup
                self.carrying = self.grid.get(*self.key_pos[i])
                self.grid.set(*self.key_pos[i], None)
                if first_door_room % 2 == 0:
                    # the first door is in the upper half of the room
                    top = (0, 0)
                    size = ((first_door_room/2 + 1)* 4, 4)
                    self.place_agent(top = top, size = size)
                else:
                    # the first door is for room 3 in the lower half
                    upper_half = np.random.rand()
                    if upper_half <= 0.62:
                        # agent in upper half of the environment
                        self.place_agent(top=(0,0), size=(12, 4))
                    else:
                        # agent in lower half of the environment
                        self.place_agent(top=(8,4), size=(4,4))
                    
            else:
                subgoal = 4
                # the first key is not picked up
                if first_door_room % 2 == 0:
                    # the first door is in the upper half of the room
                    top = (0, 0)
                    if first_door_room == 0 or not np.array_equal(self.key_pos[i], (DIR_TO_VEC[0]+self.rooms[first_door_room-2].doorPos)):
                        # the key of the first door is not blocking the hallway from the previous room
                        size = ((first_door_room/2 + 1)* 4, 4)
                        self.place_agent(top = top, size = size)
                    else:
                        # the key of the first door is blocking the hallway from the previous room
                        size = (first_door_room*2+1, 4)
                        self.place_agent(top = top, size = size)
                else:
                    # the first door is for room 3 in the lower half
                    if not np.array_equal(self.key_pos[i], (DIR_TO_VEC[1]+self.rooms[4].doorPos)):
                        # The key is not blocking the hallway from room 4 to room 5
                        upper_half = np.random.rand()
                        if upper_half <= 0.62:
                            # agent in upper half of the environment
                            self.place_agent(top=(0,0), size=(12, 4))
                        else:
                            # agent in lower half of the environment
                            self.place_agent(top=(8,4), size=(4,4))
                    else:
                        # The key is blocking the hallway from room 4 to room 5
                        self.place_agent(top=(0,0), size = (12,5))

        room, isroom = self._find_room(self.agent_pos)
        replace_agent = np.random.rand()
        if replace_agent <= 0.37:
            self._place_agent_near_hallway(room, self.rooms[room].doorPos)

        # if self.carrying:
        #     print(self.carrying.color)

        if self.partial_obs:
            obs = self.gen_obs()
        else:
            full_grid = self.grid.encode()
            full_grid[self.agent_pos[0]][self.agent_pos[1]] = np.array(
            [OBJECT_TO_IDX["agent"], COLOR_TO_IDX["red"], self.agent_dir]
        )
        obs={"image": full_grid}

        return obs, subgoal

    def _place_agent_near_hallway(self, room, doorPos):
        if room == 5:
            # room 5 has no hallway or door
            return
        if self.grid.get(*self.rooms[room].doorPos) is None:
            # The door is not locked
            # place agent in the hallway, or two sides of the hallway
            if room != 4:
                # for room 0, 1, 2, 3, just place at doorPos[0]-1, doorPos[0], doorPos[0]+1
                self.place_agent(top=(doorPos[0]-1, doorPos[1]), size=(3,1))
            else:
                self.place_agent(top=(doorPos[0], doorPos[1]-1), size=(1,3))
        else:
            # The door is locked
            # Place agent next to the door in the previous room
            # It is possible that there is a key
            first_door_room, i, second_door_room, j = self._first_door()
            key_idx = 0
            if room == first_door_room:
                key_idx = i
            else:
                key_idx = j
            if self.grid.get(*self.key_pos[key_idx]) is not None:
                # the key is not picked up
                if room == 0 or room == 2:
                    self.place_agent(top=(doorPos[0]-2, doorPos[1]), size=(2,1))
                elif room == 1 or room == 3:
                    self.place_agent(top=(doorPos[0]+1, doorPos[1]), size=(2,1))
                else:
                    self.place_agent(top=(doorPos[0], doorPos[1]-2), size=(1,2))
            else:
                # the key of the room is picked up
                if room == 0 or room == 2:
                    self.place_agent(top=(doorPos[0]-1, doorPos[1]), size=(2,1))
                elif room == 1 or room == 3:
                    self.place_agent(top=(doorPos[0], doorPos[1]), size=(2,1))
                else:
                    self.place_agent(top=(doorPos[0], doorPos[1]-1), size=(1,2))
            

    def _first_door(self):
        if self.lockedrooms[0] % 2 == 0 and self.lockedrooms[1] % 2 == 0:
            if self.lockedrooms[0] < self.lockedrooms[1]:
                return self.lockedrooms[0], 0, self.lockedrooms[1], 1
            else:
                return self.lockedrooms[1], 1, self.lockedrooms[0], 0
        elif self.lockedrooms[0] % 2 != 0 and self.lockedrooms[1] % 2 != 0:
            if self.lockedrooms[0] < self.lockedrooms[1]:
                return self.lockedrooms[1], 1, self.lockedrooms[0], 0
            else:
                return self.lockedrooms[0], 0, self.lockedrooms[1], 1
        else:
            if self.lockedrooms[0] % 2 == 0:
                return self.lockedrooms[0], 0, self.lockedrooms[1], 1
            else:
                return self.lockedrooms[1], 1, self.lockedrooms[0], 0
    
    def _find_room(self, pos):
        """
        calculate the room the pos is in
        return:
        room / hallway number, 
        True for room, False for hallway
        """
        if pos[1] < 4:
            if pos[0] == 4:
                return 0, False
            elif pos[0] == 8:
                return 2, False
            else:
                return 2 * (pos[0] // 4), True 
        elif pos[1] > 4:
            if pos[0] == 4:
                return 1, False
            elif pos[0] == 8:
                return 3, False
            else:
                return 2 * (pos[0] // 4) + 1, True
        else:
            return 4, False

    def subgoal(self, pos1, pos2):
        # calculate the first subgoal agent should go
        # if it is going from pos1 to pos2
        room1, isroom1 = self._find_room(pos1)
        room2, isroom2 = self._find_room(pos2)
        if room1 == room2:
            # if agent and goal is in the same room, with hallway
            return pos2
        if (room2 % 2) == (room1 % 2):
            # agent and goal both in lower / upper rooms
            if room1 < room2 and isroom1:
                # agent is on the left, and not in the hallway between rooms
                return self.rooms[room1].doorPos
            elif room1 == room2-2 and (not isroom1):
                # agent is in the hallway to room2
                return pos2
            elif room1 < room2-2 and (not isroom1):
                # agent in hallway from (0)1 to (2)3, goal in room (4)5
                return self.rooms[room1+2].doorPos
            else:
                # agent 1'room is on the right
                if room1 > room2+2:
                    # not in the adjacent room
                    return self.rooms[room1-2].doorPos
                else:
                    # in the adjacent room
                    return self.rooms[room2].doorPos
        else:
            if room1 % 2 == 0:
                # agent is in upper side, while goal is in lower side
                if room1 == 4 and (not isroom1):
                    # the edge case that agent 1 is in the hallway to room5
                    if room2 == 5:
                        # the goal happens to be in room 5
                        return pos2
                    else:
                        # The goal is not in room 5, the agent should first go hallway to room 3
                        return self.rooms[3].doorPos
                elif room1 == 4:
                    # The agent is in room4, first go to its hallway to room 5
                    return self.rooms[room1].doorPos
                else:
                    # The agent is in room 0 or 2
                    if room1 == 2 and (not isroom1):
                        # agent is in the hallway from room 2 to room 4, directly go to room4's hallway
                        return self.rooms[4].doorPos
                    elif isroom1:
                        # in room 0 or 2
                        return self.rooms[room1].doorPos
                    else:
                        # in the hallway of room 0
                        return self.rooms[2].doorPos
            else:
                # agent is in lower side, while goal is in upper side
                if room1 == 5:
                    # Go to the hallway from room 4 to room 5 anyway
                    return self.rooms[4].doorPos
                else:
                    if room1 == 3 and (not isroom1):
                        # agent is in the hallway from room 3 to room 5, should go to hallway of room 4 to room 5
                        return self.rooms[4].doorPos
                    elif isroom1:
                        # in room 1 or 3, go to the hallway of room 1 first
                        return self.rooms[room1].doorPos
                    else:
                        # in the hallway of room 1
                        return self.rooms[3].doorPos


    def render(self, pause = 0.01):
        r = super().render()
        time.sleep(pause)
        return r.getArray()
        
    def print_obs(self, obs_r):
        #print(' -'*9)
        for j in range(obs_r.shape[1]):
            #print('=', end='')
            for i in range(obs_r.shape[0]):
                print('|', end='')
                obj_type = obs_r[i,j,0]
                #print(obj_type)
                if obj_type == 2:
                    print('==', end='')
                elif obj_type == 10:
                    if obs_r[i,j,2] == 0:
                        print('a>', end='')
                    elif obs_r[i,j,2] == 1:
                        print('av', end='')
                    elif obs_r[i,j,2] == 2:
                        print('a<', end='')
                    elif obs_r[i,j,2] == 3:
                        print('a^', end='')
                elif obj_type == 14:
                    print('T ', end='')
                elif obj_type == 8:
                    print('H ', end='')
                elif obj_type == 5:
                    print(IDX_TO_COLOR[obs_r[i,j,1]][0]+' ', end='')
                    # if obs_r[i,j,1] == 0:
                    #     print('r ', end='')
                    # elif obs_r[i,j,1] == 5:
                    #     print('g ', end='')
                elif obj_type == 13:
                    print('V ', end='')
                elif obj_type == 4:
                    print(IDX_TO_COLOR[obs_r[i,j,1]][0]+' ', end='')
                    # if obs_r[i,j,2] == 1:
                    #     print('D ', end='')
                    # elif obs_r[i,j,2] >= 2:
                    #     print(obs_r[i,j,2], end=' ')
                    # else:
                    #     print('  ', end='')
                else:
                    print('  ', end='')
            print('|')
        print('\n\n')