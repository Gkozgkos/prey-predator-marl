import numpy as np
from prey_predator.env import marl_environment
from prey_predator.predator import QAgent

env = marl_environment(seed = 42)

agents = [QAgent(seed=i) for i  in range(env.num_predators)]

episodes = 90000
epsilon = 1.0
epsilon_min = 0.01
epsilon_decay = 0.999943

for episode in range(episodes):
    env.reset()
    done = False

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

    epsilon = max(epsilon_min, epsilon * epsilon_decay)

    if episode % 50  == 0:
        print(f"episode: {episode}, epsilon :{epsilon:.3f} , steps:{env.step_count}, caught: { env.caught}, escaped: {env.escaped}, states :{len(agents[0].q_table)}")