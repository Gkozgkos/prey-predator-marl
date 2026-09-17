import numpy as np
from prey_predator.env import marl_environment
from prey_predator.predator import QAgent
import pickle

env = marl_environment(seed = 42)

agents = [QAgent(seed=i) for i  in range(env.num_predators)]

episodes = 2000
epsilon = 1.0
epsilon_min = 0.01
epsilon_decay = 0.995

catches_per_episode = []
rewards_per_episode = [[] for _ in range(env.num_predators)]

for episode in range(episodes):
    env.reset()
    done = False

    episode_rewards = [0.0 for _ in range(env.num_predators)]

    while not done:
        states = []
        for i in range(env.num_predators):
            states.append(env.observation(env.predators[i]))

        actions = []
        for i in range(env.num_predators):
            actions.append(agents[i].choose_action(states[i], epsilon))

        next_states, rewards, done = env.step(actions)

        for i in range(env.num_predators):
            agents[i].update( states[i], actions[i] ,rewards[i], next_states[i])

        for i in range(env.num_predators):
            episode_rewards[i] += rewards[i] 

    epsilon = max(epsilon_min, epsilon * epsilon_decay)

    catches_per_episode.append(env.caught)
    for i in range(env.num_predators):
        rewards_per_episode[i].append(episode_rewards[i])

    if episode % 50  == 0:
        print(f"episode: {episode}, epsilon :{epsilon:.3f} , steps:{env.step_count}, caught: { env.caught}, escaped: {env.escaped}, states :{len(agents[0].q_table)}")

for i, agent in enumerate(agents):
    agent.save(f"q_table_{i}.pkl")

with open("metrics.pkl", "wb") as f:
    pickle.dump({"catches": catches_per_episode, "rewards": rewards_per_episode}, f)