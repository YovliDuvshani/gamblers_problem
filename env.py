import numpy as np

from config import MAX_GOLD, PH


class Env:
    @staticmethod
    def transitions(state: int, action: int):
        probabilities = np.zeros(MAX_GOLD + 1)
        rewards = np.zeros(MAX_GOLD + 1)

        if state not in [0, MAX_GOLD]:
            probabilities[state - action] = 1 - PH
            probabilities[state + action] = PH
            rewards[MAX_GOLD] = 1

        return probabilities, rewards

    @staticmethod
    def possible_actions(state: int):
        return list(range(0, max(min(MAX_GOLD - state, state), 0) + 1))
