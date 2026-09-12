import numpy as np
class marl_environment():
        
    MOVES = np.array([[0,-1], [0,1],[-1,0],[1,0]])
    '''[0, -1]   up      x same, y down
       [0,  1]   down    x same, y up
       [-1, 0]   left    x down, y same
       [1,  0]   right   x up,   y same'''

    #initializing an environment
    def __init__(self, grid_size = 16, num_predators = 2, num_preys = 3, num_lairs = 3, vision_range = 5, max_steps = 5000,  seed= None):
        
        self.grid_size = grid_size
        self.num_predators = num_predators
        self.num_preys = num_preys
        self.num_lairs = num_lairs
        self.vision_range = vision_range
        self.max_steps = max_steps
        self.rng = np.random.default_rng(seed)

        
    #reseting the enviroment by setting steps at 0, giving random positions to each entity (predators, preys, lairs). 
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

        #entities movement
    def step(self, actions):
        self.step_count += 1

        #predators movement

        for i, action in enumerate(actions):
            self.predators[i] = np.clip( self.predators[i] + self.MOVES[action], 0,self.grid_size -1 ) #np.clip prevents movement of the grid
