from collections import deque
import random
import numpy as np

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


class PrioritizedReplayMem():
    """
    Prioritized Experience Replay Memory
    Samples experiences based on their TD-error, allowing the agent to learn more from important transitions.
    """
    def __init__(self, maxlen, alpha=0.6, beta_start=0.4, beta_frames=100000, seed=None):
        """
        Args:
            maxlen: Maximum capacity of the replay buffer
            alpha: How much prioritization to use (0 = uniform sampling, 1 = full prioritization)
            beta_start: Initial importance sampling weight (corrects bias from prioritized sampling)
            beta_frames: Number of frames over which beta is annealed to 1.0
            seed: Random seed for reproducibility
        """
        self.maxlen = maxlen
        self.alpha = alpha
        self.beta_start = beta_start
        self.beta_frames = beta_frames
        self.frame = 1
        
        self.memory = []
        self.priorities = []
        self.pos = 0
        
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)
    
    def beta_by_frame(self, frame_idx):
        """
        Linearly anneal beta from beta_start to 1.0 over beta_frames
        Beta corrects the bias introduced by prioritized sampling
        """
        return min(1.0, self.beta_start + frame_idx * (1.0 - self.beta_start) / self.beta_frames)
    
    def push(self, transition, td_error=None):
        """
        Add a transition to the replay memory with priority based on TD-error
        
        Args:
            transition: (state, action, reward, next_state, done) tuple
            td_error: TD-error for this transition (if None, uses max priority)
        """
        # Use max priority for new experiences to ensure they're sampled at least once
        max_priority = max(self.priorities) if self.priorities else 1.0
        
        if td_error is not None:
            priority = (abs(td_error) + 1e-6) ** self.alpha
        else:
            priority = max_priority
        
        if len(self.memory) < self.maxlen:
            self.memory.append(transition)
            self.priorities.append(priority)
        else:
            # Circular buffer: overwrite oldest experience
            self.memory[self.pos] = transition
            self.priorities[self.pos] = priority
            self.pos = (self.pos + 1) % self.maxlen
    
    def sample(self, batch_size):
        """
        Sample a batch of experiences based on their priorities
        
        Returns:
            batch: List of sampled transitions
            indices: Indices of sampled experiences (for updating priorities)
            weights: Importance sampling weights to correct bias
        """
        N = len(self.memory)
        if N == 0:
            return [], [], []
        
        # Calculate sampling probabilities
        priorities = np.array(self.priorities)
        probs = priorities / priorities.sum()
        
        # Sample indices based on priorities
        indices = np.random.choice(N, batch_size, p=probs, replace=False)
        
        # Calculate importance sampling weights
        beta = self.beta_by_frame(self.frame)
        self.frame += 1
        
        # weights = (N * P(i))^(-beta)
        weights = (N * probs[indices]) ** (-beta)
        # Normalize weights by max weight for stability
        weights = weights / weights.max()
        
        # Get the actual transitions
        batch = [self.memory[idx] for idx in indices]
        
        return batch, indices, weights
    
    def update_priorities(self, indices, td_errors):
        """
        Update priorities for sampled experiences based on new TD-errors
        
        Args:
            indices: Indices of experiences to update
            td_errors: New TD-errors for these experiences
        """
        for idx, td_error in zip(indices, td_errors):
            priority = (abs(td_error) + 1e-6) ** self.alpha
            self.priorities[idx] = priority
    
    def __len__(self):
        return len(self.memory)