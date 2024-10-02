import gym_minigrid
import numpy as np
import queue


class BreadthFirstSearchPlanner():
    def __init__(self):
        self._comp_fn = self._get_closest_path

    def plan(self, observation, agent_pos = None, goals = None, subgoals = None):
        if agent_pos is None:
            # Add agent start node
            agent_location = np.where(observation[:, :, 0] == gym_minigrid.minigrid.OBJECT_TO_IDX["agent"])
            agent_pos = (agent_location[0][0], agent_location[1][0])

        # If key is present, plot a path to key.
        # And then plot a path from key to goal.
        if subgoals is None:
            key_locations = np.where(observation[:, :, 0] == gym_minigrid.minigrid.OBJECT_TO_IDX["key"])
            subgoals = []

            for x, y in zip(key_locations[0], key_locations[1]):
                subgoals.append((x, y))

            assert(len(subgoals) <= 1)

        if goals is None:
            goal_locations = np.where(observation[:, :, 0] == gym_minigrid.minigrid.OBJECT_TO_IDX["goal"])
            goals = []
            for x, y in zip(goal_locations[0], goal_locations[1]):
                goals.append((x, y))

        if len(subgoals) > 0:
            start_pos = subgoals[0]

            # Find path from agent pos to subgoal if exists
            subgoal_path = self._breath_first_search(agent_pos, subgoals[0], observation)

            if not subgoal_path:
                print("Failed to find subgoal path")
                return None
        else:
            start_pos = agent_pos
            subgoal_path = []
        
        paths = []
        for goal in goals:
            try:
                path = self._breath_first_search(start_pos, goal, observation)
            except:
                continue

            # If there is no existing shortest path or the new path is shorter, set the new path
            paths.append(path)

        if not paths:
            print("Failed to find regular paths")
            return None

        best_path = self._comp_fn(paths)

        # Add the subgoal path (may be empty if there is no subgoal)
        best_path = subgoal_path + best_path

        return best_path

    def _get_closest_path(self, paths):
        path_lengths = [len(path) for path in paths]

        best_index = np.argmin(path_lengths)

        return paths[best_index]

    def _get_furthest_path(self, paths):
        path_lengths = [len(path) for path in paths]

        best_index = np.argmax(path_lengths)

        return paths[best_index]

    def _get_random_path(self, paths):
        best_index = np.random.choice(range(len(paths)))

        return paths[best_index]

    def _breath_first_search(self, start_node, end_node, observation):
        node_queue = queue.Queue()
        visited = set()
        parent = dict()

        node_queue.put(start_node)
        visited.add(start_node)
        parent[start_node] = None

        # Loop until queue is empty
        while not node_queue.empty():
            # Pop current node from front of queue
            current_node = node_queue.get()

            if current_node == end_node:
                break

            for neighbor_node in self._get_neighbors(current_node, observation):
                # print("Looking at neighbor: " + str(neighbor_node))
                if neighbor_node not in visited:
                    node_queue.put(neighbor_node)
                    parent[neighbor_node] = current_node
                    visited.add(neighbor_node)

        assert(current_node == end_node)

        return self._construct_path(parent, end_node)

    def _construct_path(self, parent, end_node):
        path = [end_node]

        while parent[end_node] is not None:
            path.append(parent[end_node])
            end_node = parent[end_node]

        # Return the reversed path, except for the first node (as that contains the start location)
        return path[-2::-1]

    def _get_neighbors(self, node, observation):
        actions = [
            [-1, 0],
            [1, 0],
            [0, -1],
            [0, 1]
        ]

        neighbors = []

        for action in actions:
            neighbor_node = np.array(node) + np.array(action)
            
            # Note that this ignores doors but that's ok, it's what we want.
            if np.all(neighbor_node > 0) and np.all(neighbor_node < observation.shape[:2]) and (observation[neighbor_node[0], neighbor_node[1], 0] != gym_minigrid.minigrid.OBJECT_TO_IDX["wall"]):
                neighbors.append(tuple(neighbor_node))

        return neighbors
