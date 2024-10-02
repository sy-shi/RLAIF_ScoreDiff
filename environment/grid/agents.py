import gym_minigrid
import numpy as np

from environment.grid.bfs import BreadthFirstSearchPlanner


# This agent deterministically computes a Dijkstra's algorithm to each goal and then selects the shortest path and follows those actions.
# If two objects are equally close, one of them is selected at random.
# This should produce ambiguous paths in the case of the OneAgentTwoObjects scenario.
class RuleBasedGreedy():
    def __init__(self):
        self.path = None
        self.plan_index = None

    # Take in an observation and predict the next action
    def predict(self, observation, agent_pos=None):
        # observation = observation["image"]
        
        if self.path is None:
            planner = BreadthFirstSearchPlanner()

            self.path = planner.plan(observation, agent_pos=agent_pos)
            self.plan_index = 0

            if self.path is None:
                raise("Unable to reach goal")

        next_location = self.path[self.plan_index]

        if agent_pos is None:
            # Get vector offset of current direction to next
            agent_location = np.where(observation[:, :, 0] == gym_minigrid.minigrid.OBJECT_TO_IDX["agent"])
            agent_pos = np.array((agent_location[0][0], agent_location[1][0]))

        location_vector = (next_location - agent_pos)
        direction_vector = gym_minigrid.minigrid.DIR_TO_VEC[observation[agent_pos[0], agent_pos[1], 2]]

        # If we're already facing the next direction, move forward
        if np.all(location_vector == direction_vector):
            # If there is a key in the way, pick it up
            if observation[next_location[0], next_location[1], 0] == gym_minigrid.minigrid.OBJECT_TO_IDX["key"]:
                return gym_minigrid.minigrid.MiniGridEnv.Actions.pickup
            # If there is a door in the way and it is closed, toggle it (will open if key is being carried and it is locked)
            elif observation[next_location[0], next_location[1], 0] == gym_minigrid.minigrid.OBJECT_TO_IDX["door"] and observation[next_location[0], next_location[1], 2] != 0:
                return gym_minigrid.minigrid.MiniGridEnv.Actions.toggle
            # Otherwise just move forward and increment plan
            else:
                self.plan_index += 1
                return gym_minigrid.minigrid.MiniGridEnv.Actions.forward

        # Otherwise turn, and leave plan index where it is because we haven't taken a step yet and need to do so for a future action
        if ((np.allclose(location_vector, [0, -1]) or np.allclose(location_vector, [-1, 0])) and (np.allclose(direction_vector, [-1, 0]) or np.allclose(direction_vector, [0, 1]))) or \
           (np.allclose(location_vector, [0, 1]) and np.allclose(direction_vector, [1, 0])) or (np.allclose(location_vector, [1, 0]) and np.allclose(direction_vector, [0, -1])):
            return gym_minigrid.minigrid.MiniGridEnv.Actions.right
        
        return gym_minigrid.minigrid.MiniGridEnv.Actions.left