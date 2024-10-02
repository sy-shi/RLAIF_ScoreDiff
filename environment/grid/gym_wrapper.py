from environment.grid.simple_grid import MultiRoomGrid
from environment.grid.doubledoor import DoubleDoorEnv


class GymMultiGrid(MultiRoomGrid):
    def __init__(self, config):
        super().__init__(
            config=config["config"],
            start_rooms=config["start_rooms"],
            goal_rooms=config["goal_rooms"],
            room_size=config["room_size"],
            max_steps=config["max_steps"]
        )


class GymDoubleDoor(DoubleDoorEnv):
    def __init__(self, config):
        super().__init__(
            config=config["config"],
            max_steps=config["max_steps"]
        )