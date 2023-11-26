import matplotlib.pyplot as plt
import numpy as np

from config import MAX_GOLD, EPSILON, GAMMA
from env import Env
import plotly.express as px


class Agent:
    def __init__(self, env: Env):
        self.value_function = np.zeros(MAX_GOLD + 1)
        self.env = env

    def apply_value_iteration(self):
        loop = True
        while loop:
            delta = 0
            for state in range(MAX_GOLD + 1):
                current_state_value = self.value_function[state]
                action_values = self._get_action_values(state)
                self.value_function[state] = max(action_values)
                delta = max(
                    abs(current_state_value - self.value_function[state]), delta
                )
            print(f"delta: {delta}")
            if delta < EPSILON:
                loop = False

    def display_highest_policy(self) -> None:
        fig = px.line(self.policy())
        fig.show()

    def policy(self):
        policy = np.zeros(MAX_GOLD + 1)
        for state in range(MAX_GOLD + 1):
            action_values = self._get_action_values(state)
            policy[state] = np.argmax(action_values)
        return policy

    def display_all_highest_policies(self):
        plt.imshow(self._get_all_highest_actions().transpose())

    def q(self):
        q = np.zeros((MAX_GOLD + 1, MAX_GOLD + 1))
        for state in range(MAX_GOLD + 1):
            action_values = self._get_action_values(state)
            q[state] = action_values.copy()
        return q

    def _get_action_values(self, state: int) -> np.array:
        action_values = np.zeros(MAX_GOLD + 1)
        for action in self.env.possible_actions(state):
            action_value = 0
            probabilities, rewards = self.env.transitions(state, action)
            for next_state in range(MAX_GOLD + 1):
                action_value += probabilities[next_state] * (
                    rewards[next_state] + GAMMA * self.value_function[next_state]
                )
            action_values[action] = action_value
        return action_values

    def _get_all_highest_actions(self):
        actions_max = self.q().copy()
        max_actions = np.amax(actions_max, axis=1)
        for i in range(MAX_GOLD + 1):
            actions_max[i, :] -= max_actions[i]
        actions_max = np.where(actions_max > -EPSILON, 1, 0)
        return actions_max
