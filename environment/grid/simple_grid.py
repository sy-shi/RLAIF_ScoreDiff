import enum
import gym
import gym_minigrid
import numpy as np



# left, top, right, bottom. wwww is a closed room.
# wwwo,owww is two connected rooms with an opening between them.
# wwwo,owwd/wwwo,odww is 4 connected rooms. top 2 have an opening. right 2 have a door connecting them.
# note that adjacent rooms must match up with the wall type.
# Give this as a list
# [["wwwo", "owwd"], ["wwwo", "odww"]].
class MultiRoomGrid(gym_minigrid.minigrid.MiniGridEnv):
    def __init__(self, config, start_rooms, goal_rooms, room_size=3, max_steps=100):
        self.num_rows = len(config)
        self.num_cols = len(config[0])
        self.room_size = room_size
        self.start_rooms = start_rooms
        self.goal_rooms = goal_rooms
        self.config = config
        self.max_tries = 100

        self.width = (self.num_cols * room_size) + (self.num_cols + 1) # Sum of room sizes + 1 space extra for walls.
        self.height = (self.num_rows * room_size) + (self.num_rows + 1) # Sum of room sizes + 1 space extra for walls.

        # Placeholder mission space does nothing for now, since we don't want to use it.
        mission_space = gym_minigrid.minigrid.MissionSpace(
            mission_func=lambda color, type: f"Unused",
            ordered_placeholders=[gym_minigrid.minigrid.COLOR_NAMES, ["box", "key"]],
        )

        super().__init__(
            mission_space=mission_space,
            max_steps=max_steps,
            width=self.width,
            height=self.height,
        )


    def _sample_room(self, ul):
        try_idx = 0
        while try_idx < self.max_tries:
            loc = (np.random.randint(low=ul[0]+1, high=ul[0]+(self.room_size + 1)), np.random.randint(low=ul[1]+1, high=ul[1]+(self.room_size + 1)))

            if self.grid.get(*loc) == None and (self.agent_pos is None or not np.allclose(loc, self.agent_pos)):
                return loc

            try_idx += 1

        raise("Failed to sample point in room.")


    def _construct_room(self, room_config, ul):
        # Build default walls on all 4 sides
        self.grid.wall_rect(*ul, self.room_size + 2, self.room_size + 2)

        # Examine each wall in the room config
        for dir, wall in zip(("l", "t", "r", "b"), room_config):
            # Carve out an opening or door
            if wall == "o" or wall == "d":
                if dir == "l":
                    opening_idx = (ul[0], ul[1] + (self.room_size + 2) // 2)
                elif dir == "r":
                    opening_idx = (ul[0] + self.room_size + 1, ul[1] + (self.room_size + 2) // 2)
                elif dir == "t":
                    opening_idx = (ul[0] + (self.room_size + 2) // 2, ul[1])
                elif dir == "b":
                    opening_idx = (ul[0] + (self.room_size + 2) // 2, ul[1] + self.room_size + 1)

                if wall == "o":
                    obj_type = gym_minigrid.minigrid.Floor()
                else:
                    obj_type = gym_minigrid.minigrid.Door("red", is_open=False, is_locked=True)

                self.grid.set(*opening_idx, obj_type)


    def _gen_grid(self, width, height):
        # Create an empty grid
        self.grid = gym_minigrid.minigrid.Grid(width, height)

        self.mission = ""
        ul = [0, 0]
        key_required = False

        for row in self.config:
            for col in row:
                if "d" in col:
                    key_required = True

                self._construct_room(col, ul)
                ul[0] += self.room_size + 1

            ul[0] = 0
            ul[1] += self.room_size + 1

        # Sample agent start location
        room_idx = np.random.choice(len(self.start_rooms))
        room_ul = (self.room_size + 1) * np.array(self.start_rooms[room_idx][::-1])
        self.agent_pos = self._sample_room(room_ul)
        self.agent_dir = np.random.randint(low=0, high=4)

        # Place goal
        room_idx = np.array(self.goal_rooms[np.random.choice(len(self.goal_rooms))][::-1])
        room_ul = ((self.room_size + 1) * room_idx[0], (self.room_size + 1) * room_idx[1])
        self._place_object(room_ul, gym_minigrid.minigrid.Goal())

        if key_required:
            # Place key. Can be in any of the start rooms.
            room_idx = np.random.choice(len(self.start_rooms))
            room_ul = (self.room_size + 1) * np.array(self.start_rooms[room_idx][::-1])
            self._place_object(room_ul, gym_minigrid.minigrid.Key("red"))

    
    def _place_object(self, ul, obj):
        loc = self._sample_room(ul)
        self.put_obj(obj, *loc)

        return loc

