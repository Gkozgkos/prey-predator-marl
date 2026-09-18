import numpy as np
from prey_predator.env import marl_environment
from prey_predator.predator import QAgent

def evaluate(agents, env, episodes = 500):
    catches = []
    total_rewards = []

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
                actions.append(agents[i].choose_action(states[i], 0.0))

            rewards, done = env.step(actions)

            for i in range(env.num_predators):
                episode_rewards[i] += rewards[i] 

        catches.append(env.caught)
        total_rewards.append(sum(episode_rewards))

        if episode % 50  == 0:
            print(f"episode: {episode}, steps:{env.step_count}, caught: { env.caught}, escaped: {env.escaped}, states :{len(agents[0].q_table)}")

    return np.mean(catches), np.mean(total_rewards)
