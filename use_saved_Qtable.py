'''
Use the previously saved Q-table
'''

import numpy as np
import gym

env = gym.make("MountainCar-v0")
#q_table = np.load("-87.0-Best_Q-table.npy")
#q_table = np.load("24999-Best_Q-table.npy")
q_table = np.load("Ep22838_R-89.0_Q-table.npy")

DISCETE_OBS_SPACES = [20] * len(env.observation_space.high) #[20, 20]
# Each space size or resolution
discrete_obs_res = (env.observation_space.high - env.observation_space.low) / DISCETE_OBS_SPACES
DISCRETE_GOAL_POSITION = np.round((env.goal_position - env.observation_space.low[0]) / discrete_obs_res[0])

def make_obs_discrete(obs):
	discrete_obs = np.round((obs - env.observation_space.low) / discrete_obs_res)
	return tuple(discrete_obs.astype(np.int))

done = False
state = make_obs_discrete(env.reset())
while not done:
	action = np.argmax(q_table[state])
	new_state, reward, done, _ = env.step(action)
	state = make_obs_discrete(new_state)
	env.render()
env.close()