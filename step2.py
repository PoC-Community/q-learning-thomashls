import numpy as np
import gym
import matplotlib.pyplot as plt
import io
import sys

env = gym.make('FrozenLake-v1', is_slippery=False)

def random_action(env):
    return env.action_space.sample()

observation, info = env.reset()

action = random_action(env)
observation, reward, done, _, info = env.step(action)

old_stdout = sys.stdout
sys.stdout = buffer = io.StringIO()
env.render()
sys.stdout = old_stdout
lake_render = buffer.getvalue()

print(lake_render)

print(f"actions: {env.action_space.n}\nstates: {env.observation_space.n}")
print(f"Current state: {observation}")

env.close()
