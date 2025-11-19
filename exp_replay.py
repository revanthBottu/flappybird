from collections import deque
import random

#
#
#
#

class ReplayMem():
    def __init__(self, maxlen, seed=None):
        self.memory = deque(maxlen=maxlen)
        
        # Set random seed for reproducibility
        if seed is not None:
            random.seed(seed)

    # Add a transition to the replay memory
    def push(self, transition):
        self.memory.append(transition)

    #Get a sample batch of transitions from the replay memory
    def sample(self, sample_size):
        return random.sample(self.memory, sample_size)

    #Length of replay memory
    def __len__(self):
        return len(self.memory)