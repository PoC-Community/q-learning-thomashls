# import random
# import gymnasium as gym
import numpy as np
# import matplotlib.pyplot as plt

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

q_table = init_q_table(5, 4)

q_table[0, 1] = q_function(q_table, state=0, action=1, reward=-1, newState=3)

print("Q-Table after action:\n" + str(q_table))

assert(q_table[0, 1] == -LEARNING_RATE), f"The Q function is incorrect: the value of qTable[0, 1] should be -{LEARNING_RATE}"
