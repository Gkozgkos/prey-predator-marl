import numpy as np
from prey_predator.env import marl_environment
from prey_predator.predator import QAgent

env = marl_environment(seed=999)
agents = [QAgent(num_actions=8) for _ in range(env.num_predators)]
for i, a in enumerate(agents):
    a.load(f"runs/run_001/q_table_{i}.pkl")

env.reset()
done = False
env.render()

while not done:
    states = [env.observation(env.predators[i]) for i in range(env.num_predators)]
    actions = [agents[i].choose_action(states[i], 0.0) for i in range(env.num_predators)]
    _, _, done = env.step(actions)
    env.render()

print(f"steps: {env.step_count}, caught: {env.caught}, escaped: {env.escaped}")
env.close()