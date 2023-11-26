from agent import Agent
from env import Env
import matplotlib.pyplot as plt

if __name__ == "__main__":
    env = Env()
    agent = Agent(env)

    agent.apply_value_iteration()

    agent.display_highest_policy()
    agent.display_all_highest_policies()
