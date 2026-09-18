import numpy as np
from prey_predator.env import marl_environment
from prey_predator.predator import QAgent
import pickle
import os

RUNS_DIR = "runs"
os.makedirs(RUNS_DIR, exist_ok=True)

n = 1
while os.path.exists(os.path.join(RUNS_DIR, f"run_{n:03d}")):
    n += 1
run_dir = os.path.join(RUNS_DIR, f"run_{n:03d}")
os.makedirs(run_dir)
print(f"saving to {run_dir}")

env = marl_environment(seed = 42)

agents = [QAgent(seed=i, num_actions=8) for i  in range(env.num_predators)]

episodes = 10000
epsilon = 1.0
epsilon_min = 0.01
epsilon_decay = 0.9995

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
            agents[i].update( states[i], actions[i] ,rewards[i], next_states[i], done)

        for i in range(env.num_predators):
            episode_rewards[i] += rewards[i] 

    epsilon = max(epsilon_min, epsilon * epsilon_decay)

    catches_per_episode.append(env.caught)
    for i in range(env.num_predators):
        rewards_per_episode[i].append(episode_rewards[i])

    if episode % 50  == 0:
        print(f"episode: {episode}, epsilon :{epsilon:.3f} , steps:{env.step_count}, caught: { env.caught}, escaped: {env.escaped}, states :{len(agents[0].q_table)}")

for i, agent in enumerate(agents):
    agent.save(os.path.join(run_dir, f"q_table_{i}.pkl"))

with open(os.path.join(run_dir, "metrics.pkl"), "wb") as f:
    pickle.dump({"catches": catches_per_episode, "rewards": rewards_per_episode}, f)

config = {
    "episodes": episodes,
    "epsilon_decay": epsilon_decay,
    "grid_size": env.grid_size,
    "num_predators": env.num_predators,
    "num_preys": env.num_preys,
    "num_lairs": env.num_lairs,
    "predator_vision": env.predator_vision_range,
    "prey_vision": env.prey_vision_range,
    "max_steps": env.max_steps,
    "block_density": env.block_density,
    "discount": agents[0].discount,
    "learning_rate": agents[0].learning_rate,
    "num_actions": agents[0].num_actions,
}
with open(os.path.join(run_dir, "config.pkl"), "wb") as f:
    pickle.dump(config, f)