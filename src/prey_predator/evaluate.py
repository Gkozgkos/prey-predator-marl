import numpy as np
from prey_predator.env import marl_environment
from prey_predator.predator import QAgent

def evaluate(agents, env, episodes = 500, render =False, epsilon =0.0):
    catches = []
    total_rewards = []

    for episode in range(episodes):
        env.reset()
        done = False

        episode_rewards = [0.0 for _ in range(env.num_predators)]

        while not done:

            if render:
                env.render()
            states = []
            for i in range(env.num_predators):
                states.append(env.observation(env.predators[i]))

            actions = []
            for i in range(env.num_predators):
                actions.append(agents[i].choose_action(states[i], epsilon))

            _ , rewards, done = env.step(actions)

            for i in range(env.num_predators):
                episode_rewards[i] += rewards[i] 

        catches.append(env.caught)
        total_rewards.append(sum(episode_rewards))

        if episode % 50  == 0:
            print(f"episode: {episode}, steps:{env.step_count}, caught: { env.caught}, escaped: {env.escaped}, states :{len(agents[0].q_table)}")

    return np.mean(catches), np.mean(total_rewards)

if __name__ == "__main__":
    for eps in [0.0, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1.0]:
        env = marl_environment(seed=999)
        agents = [QAgent(num_actions=8) for _ in range(env.num_predators)]
        for i, a in enumerate(agents):
            a.load(f"runs/run_001/q_table_{i}.pkl")

        # measure
        mean_catches, mean_reward = evaluate(agents, env, episodes=200, epsilon=eps)
        print(f"eps={eps}: catches/episode {mean_catches:.3f}   reward/episode: {mean_reward:.2f}")