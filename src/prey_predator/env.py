import numpy as np
class marl_environment():
    def __init__(self, grid_size = 16, num_predators = 2, num_preys = 3, num_lairs = 3, vision_range = 5, max_steps = 5000,  seed= None):
        
        self.grid_size = grid_size
        self.num_predators = num_predators
        self.num_preys = num_preys
        self.num_lairs = num_lairs
        self.vision_range = vision_range
        self.max_steps = max_steps
        self.rng = np.random.default_rng(seed)

    def reset(self):
        self.predators = []
        self.preys = []
        self.lair = []
        for _ in range(self.num_predators):
            self.predators.append(self.rng.integers(0, self.grid_size, size = 2))
        for _ in range(self.num_preys):
            self.preys.append(self.rng.integers(0, self.grid_size, size = 2))    
        for _ in range(self.num_lairs):
            self.lair.append(self.rng.integers(0, self.grid_size, size = 2))

        self.step_count = 0