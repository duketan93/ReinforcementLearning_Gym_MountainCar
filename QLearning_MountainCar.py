'''
Classic Control Example from Gym
=== MountainCar-v0 ===
'''

import numpy as np
import gym
import matplotlib.pyplot as plt

# Set environment with Gym
env = gym.make("MountainCar-v0")
# Reset the environment to default
env.reset()

# Check the number of observations (states) from this environment
# The box space is n-dimensional box which store n types of observations
print("The number of observations types: ", env.observation_space)
# Check the box bounds
print("The high values of each observation types: ", env.observation_space.high)
print("The low values of each observation types: ", env.observation_space.low)

# Check the number of actions
print("The number of actions: ", env.action_space.n)

'''
The observation are continuous values which make it impossible to define a discrete q-table. 
So we have to choose the number of discrete spaces for the observations. 
In this case car position and car velocity. 
Let's choose 20 discrete spaces for both of them. 
'''
DISCETE_OBS_SPACES = [20] * len(env.observation_space.high) #[20, 20]
# Each space size or resolution
discrete_obs_res = (env.observation_space.high - env.observation_space.low) / DISCETE_OBS_SPACES
print("Resolution of each observation, [pos, vel]: ", discrete_obs_res)
# Get the discrete value of the goal position
DISCRETE_GOAL_POSITION = np.round((env.goal_position - env.observation_space.low[0]) / discrete_obs_res[0])

# Initialized Q-Table with random rewards (or 0 reward)
q_table = np.random.uniform(-1, 0, size=(DISCETE_OBS_SPACES + [env.action_space.n])) #[20, 20, 3] table

# Set hyperparameters
LEARNING_RATE = 0.1
DISCOUNT_RATE = 0.95 #5% Inflation after each step
# This is to highlight the importance of future rewards as compared to current or past reward

EPISODES = 25000 #
SAMPLE_SIZE = 1000

# Introduce randomness when selecting the actions to let agent to explore the environment
epsilon = 0.95 #Randomness to promote exploration

# Use for stats analysis and ploting graph
rewards = []
stats = {'ep':[], 'mean':[], 'max':[], 'min':[]}

'''
After each step, the returned new states or new observations are still continuous values, we need to convert them to discrete values as per defined earlier 
'''
def make_obs_discrete(obs):
	discrete_obs = np.round((obs - env.observation_space.low) / discrete_obs_res)
	return tuple(discrete_obs.astype(np.int))

# Start of main
if __name__ == "__main__":
	best_reward = -200
	for episode in range(EPISODES):
		episode_reward = 0
		# Print number of episode at every SAMPLE SIZE
		if episode % SAMPLE_SIZE == 0:
			print("Running at episode: ", episode)

		# Beginning of each episode
		done = False
		state = make_obs_discrete(env.reset()) #Get the initial discrete state
		while not done:
			if np.random.random() < epsilon:
				action = np.random.randint(0, env.action_space.n) #Choose random action between 0, 1, 2
			else:
				action = np.argmax(q_table[state])
			new_state, reward, done, _ = env.step(action)
			episode_reward += reward
			new_state = make_obs_discrete(new_state)
			# Render at every SAMPLE SIZE
			if episode % SAMPLE_SIZE == 0:
				env.render()
			if not done:
				current_q = q_table[state + (action,)]
				max_future_q = np.max(q_table[new_state])
				# Q-func = current_Q + lr*[reward + discount*(max_future_reward) - current_Q]
				new_q = (1 - LEARNING_RATE) * current_q + LEARNING_RATE * (reward + DISCOUNT_RATE * max_future_q)
				# Update Q-Table with new Q values
				q_table[state + (action,)] = new_q
			# done and reaching the goal
			elif new_state[0] >= DISCRETE_GOAL_POSITION: #NOTE: This is discrete space
				q_table[state + (action,)] = 0 #reward for reaching the goal
				print(f"We have achive the goal at episode {episode}")
			state = new_state

		# Update stats
		rewards.append(episode_reward)
		# Eery SAMPLE SIZE, take mean, max, and min
		if not episode % SAMPLE_SIZE:
			stats['ep'].append(episode)
			mean = sum(rewards[-SAMPLE_SIZE:])/len(rewards[-SAMPLE_SIZE:])
			stats['mean'].append(mean)
			stats['max'].append(max(rewards[-SAMPLE_SIZE:]))
			stats['min'].append(min(rewards[-SAMPLE_SIZE:]))

		# Print some stats at every SAMPLE SIZE
		if not episode % SAMPLE_SIZE:
			print(f"Episode: {episode}, Episode Reward: {episode_reward}, Mean Rewards: {mean}, Epsilon Value: {epsilon}")

		# epsilon decay policy
		if epsilon >= 0.5:
			epsilon = epsilon - (epsilon / 5000)
		elif epsilon >= 0.01 and episode <= (EPISODES - EPISODES//10):
			epsilon = epsilon - (epsilon / 10000)
		else:
			epsilon = 0.01

		# Save the max reward q-table, the best q-table
		if episode_reward > best_reward:
			best_reward = episode_reward
			to_save_q_table = q_table
			save_num = episode

	env.close()

	# Save the best q table
	np.save(f"Ep{save_num}_R{best_reward}_Q-table.npy", to_save_q_table)
	# Sanity check, they should be the same value, the best score
	print(f"The best reward is {best_reward}, the max reward is {max(rewards)}.")

	# Plot the stats in graph
	plt.figure()
	plt.plot(rewards, label='rewards')
	plt.plot(stats['ep'], stats['mean'], label='Mean')
	plt.plot(stats['ep'], stats['max'], label='Max')
	plt.plot(stats['ep'], stats['min'], label='Min')
	plt.legend(loc=2)
	plt.grid(True)
	plt.show()