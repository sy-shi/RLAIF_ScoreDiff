import gym
import gym_minigrid
import numpy as np

from gym import spaces
from gym.core import ObservationWrapper

from environment.grid.bfs import BreadthFirstSearchPlanner


# A backwards compatibility wrapper so that RLlib can continue using the old deprecated Gym API
class GymCompatWrapper(gym.Wrapper):
    def reset(self, **kwargs):
        obs, _ = self.env.reset(**kwargs)

        return obs

    def step(self, action):
        observation, reward, terminated, truncated, info = self.env.step(action)

        # Since RLlib doesn't support the truncated variable (yet), incorporate it into terminated
        terminated = terminated or truncated

        return observation, reward, terminated, info


class FullyObsWrapper(ObservationWrapper):
    """
    Fully observable gridworld using a compact grid encoding
    """

    def __init__(self, env):
        super().__init__(env)

        self.observation_space = gym.spaces.Dict({
            "image": spaces.Box(
                low=0,
                high=255,
                shape=(self.env.width, self.env.height, 3),
                dtype="uint8"),
        })

    def observation(self, obs):
        env = self.unwrapped
        full_grid = env.grid.encode()
        full_grid[env.agent_pos[0]][env.agent_pos[1]] = np.array(
            [gym_minigrid.minigrid.OBJECT_TO_IDX["agent"], gym_minigrid.minigrid.COLOR_TO_IDX["red"], env.agent_dir]
        )

        return {"image": full_grid}


class ActionMasking(ObservationWrapper):
    def __init__(self, env):
        super().__init__(env)

        # The action mask sets a value for each action of either 0 (invalid) or 1 (valid).
        self.observation_space = spaces.Dict({
            **self.observation_space.spaces,
            "action_mask": gym.spaces.Box(0.0, 1.0, shape=(self.action_space.n,))
        })

    def observation(self, obs):
        action_mask = np.ones(self.action_space.n)

        # Look at the position directly in front of the agent
        front_pos = self.unwrapped.front_pos
        front_pos_type = obs["image"][front_pos[0]][front_pos[1]][0]

        if front_pos_type == gym_minigrid.minigrid.OBJECT_TO_IDX["wall"]:
            # print("forward: ", self.env.Actions.forward.value)
            action_mask[self.env.actions.forward.value] = 0.0

        if front_pos_type != gym_minigrid.minigrid.OBJECT_TO_IDX["key"]:
            action_mask[self.env.actions.pickup.value] = 0.0

        if front_pos_type != gym_minigrid.minigrid.OBJECT_TO_IDX["door"]:
            action_mask[self.env.actions.toggle.value] = 0.0

        # Now disable actions that we intend to never use
        # action_mask[self.env.Actions.drop.value] = 0.0
        # action_mask[self.env.Actions.done.value] = 0.0

        return {**obs, "action_mask": action_mask}


class DoorUnlockBonus(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)

    def step(self, action):
        obs = self.unwrapped.grid.encode()

        # If we just unlocked a door, add a reward shaping bonus.
        front_pos = self.unwrapped.front_pos
        front_pos_type = obs[front_pos[0]][front_pos[1]][0]
        front_pos_state = obs[front_pos[0]][front_pos[1]][2]

        
        if front_pos_type == gym_minigrid.minigrid.OBJECT_TO_IDX["door"] and front_pos_state == 2:
            is_locked_door = True
        else:
            is_locked_door = False

        obs, reward, done, info = self.env.step(action)
        
        
        bonus = 0.0
        if is_locked_door and action == self.env.Actions.toggle:
            front_pos_state = obs["image"][front_pos[0]][front_pos[1]][2]
            if front_pos_state == 0:
                bonus = 0.5

        reward += bonus

        return obs, reward, done, info


class ExplorationBonus(gym.Wrapper):
    """
    Adds an exploration bonus based the distance to the goal along a path.
    """

    def __init__(self, env):
        super().__init__(env)
        self.path = None
        self.path_idx = None

    def step(self, action):
        obs, reward, done, info = self.env.step(action)

        agent_pos = np.array(self.unwrapped.agent_pos)
        dist_to_current = np.linalg.norm(self.path[self.path_idx] - agent_pos)

        if self.path_idx < len(self.path) - 1:
            dist_to_next = np.linalg.norm(self.path[self.path_idx + 1] - agent_pos)

            if dist_to_next < dist_to_current:
                self.path_idx += 1
                
        if self.path_idx > 0:
            dist_to_prev = np.linalg.norm(self.path[self.path_idx - 1] - agent_pos)
            if dist_to_prev < dist_to_current:
                self.path_idx -= 1

            # print("Dist to prev: " + str(dist_to_prev))

        # The penalty is the remaining path length
        penalty = float(len(self.path) - self.path_idx)

        # Add penalty for distance from path
        penalty += float(np.linalg.norm(self.path[self.path_idx] - agent_pos))

        # Scale the penalty by the path length to fall between [0, 1]
        penalty /= len(self.path)
        # print(penalty)
        penalty /= self.max_steps

        reward -= penalty

        return obs, reward, done, info

    def _get_grid_obs(self):
        env = self.unwrapped
        full_grid = env.grid.encode()
        full_grid[env.agent_pos[0]][env.agent_pos[1]] = np.array([
            gym_minigrid.minigrid.OBJECT_TO_IDX['agent'],
            gym_minigrid.minigrid.COLOR_TO_IDX['red'],
            env.agent_dir
        ])

        return full_grid

    def reset(self, **kwargs):
        obs = self.env.reset(**kwargs)

        planner = BreadthFirstSearchPlanner()
        self.path = planner.plan(self._get_grid_obs())

        # Push the agent's starting position into the path so we can get an accurate counting of path length
        agent_pos = self.unwrapped.agent_pos
        self.path.insert(0, agent_pos)

        self.path_idx = 0

        return obs


class ActionBonus(gym.Wrapper):
    """
    Wrapper which adds an exploration bonus.
    This is a reward to encourage exploration of less
    visited (state,action) pairs.
    Example:
        >>> import miniworld
        >>> import gymnasium as gym
        >>> from minigrid.wrappers import ActionBonus
        >>> env = gym.make("MiniGrid-Empty-5x5-v0")
        >>> _, _ = env.reset(seed=0)
        >>> _, reward, _, _, _ = env.step(1)
        >>> print(reward)
        0
        >>> _, reward, _, _, _ = env.step(1)
        >>> print(reward)
        0
        >>> env_bonus = ActionBonus(env)
        >>> _, _ = env_bonus.reset(seed=0)
        >>> _, reward, _, _, _ = env_bonus.step(1)
        >>> print(reward)
        1.0
        >>> _, reward, _, _, _ = env_bonus.step(1)
        >>> print(reward)
        1.0
    """

    def __init__(self, env):
        """A wrapper that adds an exploration bonus to less visited (state,action) pairs.
        Args:
            env: The environment to apply the wrapper
        """
        super().__init__(env)
        self.counts = {}

    def step(self, action):
        """Steps through the environment with `action`."""
        obs, reward, terminated, info = self.env.step(action)

        env = self.unwrapped
        tup = (tuple(env.agent_pos), env.agent_dir, action)

        # Get the count for this (s,a) pair
        pre_count = 0
        if tup in self.counts:
            pre_count = self.counts[tup]

        # Update the count for this (s,a) pair
        new_count = pre_count + 1
        self.counts[tup] = new_count

        bonus = 1 / np.sqrt(new_count)
        reward += bonus

        return obs, reward, terminated, info

    def reset(self, **kwargs):
        """Resets the environment with `kwargs`."""
        return self.env.reset(**kwargs)


class StateBonus(gym.Wrapper):
    """
    Adds an exploration bonus based on which positions
    are visited on the grid.
    Example:
        >>> import miniworld
        >>> import gymnasium as gym
        >>> from minigrid.wrappers import StateBonus
        >>> env = gym.make("MiniGrid-Empty-5x5-v0")
        >>> _, _ = env.reset(seed=0)
        >>> _, reward, _, _, _ = env.step(1)
        >>> print(reward)
        0
        >>> _, reward, _, _, _ = env.step(1)
        >>> print(reward)
        0
        >>> env_bonus = StateBonus(env)
        >>> obs, _ = env_bonus.reset(seed=0)
        >>> obs, reward, terminated, truncated, info = env_bonus.step(1)
        >>> print(reward)
        1.0
        >>> obs, reward, terminated, truncated, info = env_bonus.step(1)
        >>> print(reward)
        0.7071067811865475
    """

    def __init__(self, env):
        """A wrapper that adds an exploration bonus to less visited positions.
        Args:
            env: The environment to apply the wrapper
        """
        super().__init__(env)
        self.counts = {}

    def step(self, action):
        """Steps through the environment with `action`."""
        obs, reward, terminated, info = self.env.step(action)

        # Tuple based on which we index the counts
        # We use the position after an update
        env = self.unwrapped
        tup = tuple(env.agent_pos)

        # Get the count for this key
        pre_count = 0
        if tup in self.counts:
            pre_count = self.counts[tup]

        # Update the count for this key
        new_count = pre_count + 1
        self.counts[tup] = new_count

        bonus = 1 / np.sqrt(new_count)
        reward += bonus

        return obs, reward, terminated, info

    def reset(self, **kwargs):
        """Resets the environment with `kwargs`."""
        return self.env.reset(**kwargs)