import numpy as np


class marl_environment():
        
    MOVES = np.array([[0,-1], [0,1],[-1,0],[1,0]])
    '''[0, -1]   up      x same, y down
       [0,  1]   down    x same, y up
       [-1, 0]   left    x down, y same
       [1,  0]   right   x up,   y same'''

    #initializing an environment
    def __init__(self, grid_size = 16, num_predators = 2, num_preys = 3, num_lairs = 2, predator_vision_range = 5,prey_vision_range = 2, max_steps = 2000,  seed= None):
        
        self.grid_size = grid_size
        self.num_predators = num_predators
        self.num_preys = num_preys
        self.num_lairs = num_lairs
        self.predator_vision_range = predator_vision_range
        self.prey_vision_range = prey_vision_range
        self.max_steps = max_steps
        self.rng = np.random.default_rng(seed)

    def can_see(self, a, b, vision_range):
        return self.distance(a, b) <= vision_range

    def distance(self, a, b):
        return np.abs(a-b).sum()
                
    #reseting the enviroment by setting steps at 0, giving random positions to each entity (predators, preys, lairs). 
    def reset(self):

        self.predators = []
        self.preys = []
        self.lairs = []

        for _ in range(self.num_predators):
            self.predators.append(self.rng.integers(0, self.grid_size, size = 2))
        for _ in range(self.num_preys):
            self.preys.append(self.rng.integers(0, self.grid_size, size = 2))    
        for _ in range(self.num_lairs):
            self.lairs.append(self.rng.integers(0, self.grid_size, size = 2))

        self.step_count = 0
        self.caught = 0
        self.escaped = 0
    
    def step(self, actions):

        self.step_count += 1
        self.move_predators(actions)
        self.move_preys()
        rewards = self.reward_system()
        self.check_status_prey()
        observations = [self.observation(pred) for pred in self.predators] 

        return observations, rewards, self.terminated()

    #predator movement
    def move_predators(self, actions):

        
        for i, action in enumerate(actions):
            self.predators[i] = np.clip( self.predators[i] + self.MOVES[action], 0, self.grid_size -1) #np.clip prevents movement off the grid (value, minimum, maximum)

    #prey movement
    def move_preys(self):

        for i in range(len(self.preys)):
            prey = self.preys[i]

            visible = []
            visible_lair = []
            for pred in self.predators:
                if np.abs(prey - pred).sum() <= self.prey_vision_range:
                  visible.append(pred)

            if not visible:
                continue

            for lair in self.lairs:
                if np.abs(prey - lair).sum() <= self.prey_vision_range:
                    visible_lair.append(lair)
                    
            if visible_lair:
                nearest = min(visible_lair, key=lambda l: np.abs(prey - l).sum())
                direction = np.clip(nearest - prey, -1, 1)
                self.preys[i] = np.clip(prey + direction, 0, self.grid_size - 1)
                continue

            distances = []

            for pred in visible:
                distances.append(np.abs(prey- pred).sum())

            best_score = min(distances)
            best_pos = prey

            for move in self.MOVES:
                option = np.clip(prey + move, 0, self.grid_size -1)

                distance_option = []

                for pred in visible:
                    distance_option.append(np.abs(option - pred).sum())
                score = min(distance_option)        

                if score > best_score:
                    best_score = score
                    best_pos = option

            self.preys[i] = best_pos    

    #per step it checks if each of the preys are still on the grid, by compare their coords to the predators and lairs.
    def check_status_prey(self):
        alive = []

        for prey in self.preys:

            on_predator = any(np.array_equal(prey, pred)for pred in self.predators)
            on_lair = any(np.array_equal(prey, lair)for lair in self.lairs)

            if on_lair:
                self.escaped += 1

            elif on_predator:
                self.caught += 1

            else:
                alive.append(prey)

        self.preys = alive        

    #terminating the episode if the parameters are met.
    def terminated(self):  
        return len(self.preys)== 0 or self.step_count >= self.max_steps

    #set a penalty per step, reward per catch and approximate to a prey
    def reward_system(self):
        rewards = [-0.005 for _ in range(self.num_predators)] 

        for i, pred in enumerate(self.predators):

            if any(np.array_equal(pred, prey) for prey in self.preys):
                rewards[i] += 2

            for prey in self.preys:
                if self.can_see(pred, prey, self.predator_vision_range):
                    distance = self.distance(prey, pred)
                    rewards[i] += 0.05 * (self.predator_vision_range - distance)/ self.predator_vision_range

        return rewards

    def observation(self, predator):
        preys_in_range = []
        lairs_in_range = []
        
        for prey in self.preys:
            if self.can_see(prey, predator, self.predator_vision_range):
                preys_in_range.append(prey)

        if preys_in_range:
            closest_prey = min(preys_in_range, key=lambda p: self.distance(predator, p))  
            rel_prey = tuple(closest_prey - predator)

        else:
            rel_prey = None
                    
        for lair in self.lairs:
            if self.can_see(lair, predator, self.predator_vision_range):
                lairs_in_range.append(lair)

        if lairs_in_range:
            closest_lair = min(lairs_in_range, key=lambda p: self.distance(predator, p))
            rel_lair = tuple(closest_lair - predator)
        else:
            rel_lair = None

        return (rel_prey, rel_lair)    