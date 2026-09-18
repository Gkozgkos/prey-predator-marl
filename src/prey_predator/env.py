import numpy as np
import pygame

class marl_environment():
        
    MOVES = np.array([[0,-1], [0,1], [-1,0], [1,0],[-1,-1], [-1,1], [1,-1], [1,1]])
    '''[0, -1]   up      x same, y down
       [0,  1]   down    x same, y up
       [-1, 0]   left    x down, y same
       [1,  0]   right   x up,   y same'''

    #initializing an environment
    def __init__(self, grid_size = 16, num_predators = 2, num_preys = 3, num_lairs = 1, predator_vision_range = 5,prey_vision_range = 5, max_steps = 2000,  seed= None, block_density = 0.35, render_mode = None):
        
        self.grid_size = grid_size
        self.num_predators = num_predators
        self.num_preys = num_preys
        self.num_lairs = num_lairs
        self.predator_vision_range = predator_vision_range
        self.prey_vision_range = prey_vision_range
        self.max_steps = max_steps
        self.rng = np.random.default_rng(seed)
        self.block_density = block_density
        self.render_mode = render_mode
        self.screen = None

    def can_see(self, a, b, vision_range):
        return self.distance(a, b) <= vision_range
    
    #chebyshev distance
    def distance(self, a, b):
        return np.abs(a-b).max()
    
    def is_blocked(self, pos):
        return self.blocked[pos[0], pos[1]]

    def free_positions(self):
        while True:
            pos = self.rng.integers(0, self.grid_size, size =2)
            if not self.is_blocked(pos):
                return pos

    def nearest_prey_distance(self,predator):
        visible = [prey for prey in self.preys if self.can_see(predator, prey, self.predator_vision_range )]
        if not visible:
            return None
        return min(self.distance(predator, prey) for prey in visible)
          
    #reseting the enviroment by setting steps at 0, giving random positions to each entity (predators, preys, lairs). 
    def reset(self):

        self.blocked = np.zeros((self.grid_size, self.grid_size), dtype=bool)
        num_blocked = int(self.grid_size * self.grid_size * self.block_density)

        self.predators = []
        self.preys = []
        self.lairs = []
        placed = 0

        while placed < num_blocked:

            x = self.rng.integers(0, self.grid_size)
            y = self.rng.integers(0, self.grid_size)

            if not self.blocked[x,y]:

                self.blocked[x,y] = True
                placed += 1

        for _ in range(self.num_predators):
            self.predators.append(self.free_positions())
        for _ in range(self.num_preys):
            self.preys.append(self.free_positions())    
        for _ in range(self.num_lairs):
            self.lairs.append(self.free_positions())

        self.step_count = 0
        self.caught = 0
        self.escaped = 0
    
    def step(self, actions):

        self.step_count += 1
        prev_distances = [self.nearest_prey_distance(p) for p in self.predators]
        self.move_predators(actions)
        self.move_preys()
        rewards = self.reward_system(prev_distances)
        self.check_status_prey()
        observations = [self.observation(pred) for pred in self.predators] 

        return observations, rewards, self.terminated()

    #predator movement
    def move_predators(self, actions):
        
        for i, action in enumerate(actions):
                temp_pos = np.clip(self.predators[i] + self.MOVES[action], 0, self.grid_size -1)
                if not self.is_blocked(temp_pos):
                    self.predators[i] = temp_pos
    #prey movement
    def move_preys(self):

        for i in range(len(self.preys)):
            prey = self.preys[i]

            visible = []
            visible_lair = []
            for pred in self.predators:
                if self.can_see(prey, pred, self.prey_vision_range):
                  visible.append(pred)

            if not visible:
                continue

            for lair in self.lairs:
                if self.can_see(prey, lair, self.prey_vision_range):
                    visible_lair.append(lair)
                    
            if visible_lair:
                nearest = min(visible_lair, key=lambda l: self.distance(prey, l))
                direction = np.clip(nearest - prey, -1, 1)
                temp_pos = np.clip(prey + direction, 0, self.grid_size - 1)
                if not self.is_blocked(temp_pos):
                    self.preys[i] = temp_pos
                continue

            distances = []

            for pred in visible:
                distances.append(self.distance(prey, pred))

            best_score = min(distances)
            best_pos = prey

            for move in self.MOVES:
                option = np.clip(prey + move, 0, self.grid_size -1)
                if self.is_blocked(option):
                    continue

                distance_option = []

                for pred in visible:
                    distance_option.append(self.distance(option, pred))
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
    def reward_system(self, prev_distances):
        rewards = [-0.005 for _ in range(self.num_predators)] 

        for i, pred in enumerate(self.predators):

            if any(np.array_equal(pred, prey) for prey in self.preys):
                rewards[i] += 2

            new_distance = self.nearest_prey_distance(pred)
            old_distance = prev_distances[i]

            if old_distance is not None and new_distance is not None:
                rewards[i] += 0.05 * (old_distance - new_distance)

            

            """for prey in self.preys:
                if self.can_see(pred, prey, self.predator_vision_range):
                    distance = self.distance(prey, pred)
                    rewards[i] += 0.05 * (self.predator_vision_range - distance)/ self.predator_vision_range"""

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

    def _init_pygame(self):
        pygame.init()
        self.cell_size = 30
        self.window_size = self.grid_size * self.cell_size
        self.screen = pygame.display.set_mode((self.window_size, self.window_size))
        self.clock = pygame.time.Clock()

    def render(self):

        if self.screen is None:
            self._init_pygame()

        self.screen.fill((255, 255, 255))

        #obstacles and walls
        for x in range(self.grid_size):
            for y in range(self.grid_size):
                if self.blocked[x,y]:
                    rect = pygame.Rect(x * self.cell_size, y * self.cell_size, self.cell_size, self.cell_size)
                    pygame.draw.rect(self.screen, (0,0,0), rect)

        # lairs
        for lair in self.lairs:
            rect = pygame.Rect(lair[0] * self.cell_size, lair[1] * self.cell_size, self.cell_size, self.cell_size)
            pygame.draw.rect(self.screen, (128, 128, 128), rect)

        # vision range
        for pred in self.predators:
            r = self.predator_vision_range
            size = (2 * r + 1)* self.cell_size
            rect = pygame.Rect((pred[0] - r) * self.cell_size , (pred[1] -r )* self.cell_size, size, size)
            pygame.draw.rect(self.screen, (173, 216, 230), rect,  1)

        for prey in self.preys:
            r = self.prey_vision_range
            size = (2 * r + 1)* self.cell_size
            rect = pygame.Rect((prey[0] - r) * self.cell_size , (prey[1] - r) * self.cell_size, size, size)
            pygame.draw.rect(self.screen, (173, 216, 230), rect,  1)

        # preys (yellow)
        for prey in self.preys:
            rect = pygame.Rect(prey[0] * self.cell_size, prey[1] * self.cell_size, self.cell_size, self.cell_size)
            pygame.draw.rect(self.screen, (255, 255, 0), rect, border_radius=100)

        # predators
        for pred in self.predators:
            agent_color = (255, 0, 0) if any(self.can_see(pred, prey, self.predator_vision_range) for prey in self.preys) else (0, 0, 0)
            rect = pygame.Rect(pred[0] * self.cell_size, pred[1] * self.cell_size, self.cell_size, self.cell_size)
            pygame.draw.rect(self.screen, agent_color, rect, width=3, border_radius=100)

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.close()

        pygame.display.flip()
        self.clock.tick(10)

    def close(self):
        if self.screen is not None:
            pygame.quit()
            self.screen = None