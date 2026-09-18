import numpy as np
import pickle

class QAgent():

    def __init__(self, learning_rate = 0.3, discount = 0.9, num_actions = 4, seed = None):
        self.learning_rate = learning_rate
        self.discount = discount
        self.num_actions = num_actions
        self.rng = np.random.default_rng(seed)
        self.q_table = {}

    def choose_action(self, state, epsilon):

        if state not in self.q_table:
            self.q_table[state] = np.zeros(self.num_actions)

        if self.rng.random() < epsilon:
            return self.rng.integers(0, self.num_actions)
        
        else:
            return np.argmax(self.q_table[state])
        
    def update(self, state, action, reward, next_state, done= False):

        if state not in self.q_table:
            self.q_table[state] = np.zeros(self.num_actions)
        
        if next_state not in self.q_table:
            self.q_table[next_state] = np.zeros(self.num_actions)

        if done:
            target = reward
        else:
            target = reward +self.discount * np.max(self.q_table[next_state])
        old_value = self.q_table[state][action]
        self.q_table[state][action] = (1- self.learning_rate) * old_value + self.learning_rate * target

    def save(self, path):
        with open(path, "wb") as f:
            pickle.dump(self.q_table, f) 

    def load(self, path):
        with open(path,"rb") as f:
            self.q_table = pickle.load(f)