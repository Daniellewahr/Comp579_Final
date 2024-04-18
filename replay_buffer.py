import numpy as np

# class for storing and sampling experiences
class ReplayBuffer:
    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
        self.priorities = np.zeros(capacity, dtype=np.float32)
        self.alpha = 0.6  # Hyperparameter for prioritized experience replay
        self.beta = 0.4  # Importance-sampling weight parameter
        self.beta_increment_per_sampling = 0.001  # Increment value for beta
        
    def add(self, experience):
        max_priority = self.priorities.max() if self.memory else 1.0
        self.memory.append(experience)
        self.priorities[len(self.memory) - 1] = max_priority
    
    def sample(self, batch_size):
        if len(self.memory) == self.capacity:
            priorities = self.priorities
        else:
            priorities = self.priorities[:len(self.memory)]
        probabilities = priorities ** self.alpha / np.sum(priorities ** self.alpha)
        # normalize the probabilities
        probabilities /= probabilities.sum()
        indices = np.random.choice(len(self.memory), batch_size, p=probabilities)
        samples = [self.memory[idx] for idx in indices]
        total = len(self.memory)
        weights = (total * probabilities[indices]) ** (-self.beta)
        weights /= weights.max()
        return samples, indices, torch.tensor(weights, dtype=torch.float32).unsqueeze(1)
    
    def update_priorities(self, indices, priorities):
        for idx, priority in zip(indices, priorities):
            self.priorities[idx] = priority