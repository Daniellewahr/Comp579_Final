from collections import namedtuple
import numpy as np

class PrioritizedReplayMemory:
    def __init__(self, capacity, alpha=0.6, beta=0.4, beta_annealing_steps=1000):
        self.capacity = capacity
        self.memory = []
        self.alpha = alpha
        self.beta = beta
        self.beta_annealing_steps = beta_annealing_steps
        self.transitions = namedtuple('Transition', ('state', 'action', 'reward', 'next_state'))
        self.priorities = np.zeros((capacity,), dtype=np.float32)
        self.position = 0
        self.priorities_sum_alpha = 0.0

    def push(self, state, action, reward, next_state):
        max_priority = np.max(self.priorities) if self.memory else 1.0
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        self.memory[self.position] = self.transitions(state, action, reward, next_state)
        self.priorities[self.position] = max_priority
        self.priorities_sum_alpha += max_priority ** self.alpha
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        if len(self.memory) == self.capacity:
            priorities = self.priorities
        else:
            priorities = self.priorities[:self.position]

        probabilities = priorities ** self.alpha / self.priorities_sum_alpha
        probabilities /= probabilities.sum()

        # Anneal beta over time
        beta = min(1.0, self.beta + (1.0 - self.beta) * (self.position / self.beta_annealing_steps))

        indices = np.random.choice(len(self.memory), batch_size, p=probabilities)
        experiences = [self.memory[idx] for idx in indices]

        # Compute importance sampling weights
        weights = (len(self.memory) * probabilities[indices]) ** (-beta)
        weights /= weights.max()

        return indices, experiences, weights

    def update_priorities(self, indices, td_errors):
        for idx, td_error in zip(indices, td_errors):
            self.priorities[idx] = abs(td_error) + 1e-5
        self.priorities_sum_alpha = np.sum(self.priorities ** self.alpha)

    def __len__(self):
        return len(self.memory)
