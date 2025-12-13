import gymnasium as gym
from gymnasium import Wrapper


class StaleScoreWrapper(Wrapper):
    """
    Terminates the episode if the game score does not change for a specified number of steps.
    """
    def __init__(self, env: gym.Env, max_stale_steps: int = 10):
        super().__init__(env)
        self.max_stale_steps = max_stale_steps
        self.last_score = 0
        self.stale_step_count = 0

    def reset(self, **kwargs):
        self.last_score = 0
        self.stale_step_count = 0
        return self.env.reset(**kwargs)

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)

        current_score = info['game_score']

        if current_score == self.last_score:
            self.stale_step_count += 1
        else:
            self.stale_step_count = 0
            self.last_score = current_score

        if self.stale_step_count >= self.max_stale_steps:
            terminated = True

        return obs, reward, terminated, truncated, info

