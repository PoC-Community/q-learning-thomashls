import numpy as np
import gym
import matplotlib.pyplot as plt
import io
import sys

LEARNING_RATE = 0.05
DISCOUNT_RATE = 0.99

def init_q_table(x: int, y: int) -> np.ndarray:
    return np.zeros((x, y))

def q_function(q_table: np.ndarray, state: int, action: int, reward: int, newState: int) -> float:
    old_value = q_table[state, action]
    next_max = np.max(q_table[newState])
    new_value = old_value + LEARNING_RATE * (reward + DISCOUNT_RATE * next_max - old_value)
    q_table[state, action] = new_value
    return new_value

def game_loop(env: gym.Env, q_table: np.ndarray, state: int, action: int) -> tuple:
    new_state, reward, done, _, info = env.step(action)
    q_function(q_table, state, action, reward, new_state)
    return q_table, new_state, done, reward

def random_action(env):
    return env.action_space.sample()

def best_action(q_table: np.ndarray, state: int) -> int:
    return np.argmax(q_table[state])

env = gym.make("FrozenLake-v1", map_name="4x4", is_slippery=False, render_mode="human")
q_table = init_q_table(env.observation_space.n, env.action_space.n)

state, info = env.reset()
while (True):
    env.render()
    action = best_action(q_table, state)
    q_table, state, done, reward = game_loop(env, q_table, state, action)
    if done:
        break

env.close()