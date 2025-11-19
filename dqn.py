import torch
from torch import nn
import torch.nn.functional as F

#Neural Network for Deep Q-Learning, used to approx. Q function. 
# 
#The Q function attempts to predict the future reward for an action in the state the agent is in.

class DQN(nn.Module):
    # This is a two-layer neural network with everything fully connected.
    def __init__(self, state_dim, action_dim, hidden_dim=256):
        super(DQN, self).__init__()
        #State is the input that the agent observes in order to make its decision. state_dim is the input layer of the neural network. The input layer
        # feeds the observation into the hidden layer for processing.
        # Action is the output of the neural network. 
        # After calculations predicting the future rewards of each action, the nn outputs the best action into action_dim.
        self.fc1 = nn.Linear(state_dim, hidden_dim)  # fc1 converts state_dim to hidden_dim of 256 nodes/neurons in this case.
        self.fc2 = nn.Linear(hidden_dim, action_dim) # fc2 converts hidden_dim calculations to action_dim outputs
        
    def forward(self, x):
        x = F.relu(self.fc1(x)) #ReLU(f(x) = max(0,x)), applied after first layer to make non-linear function
        x = self.fc2(x) #Last layer no activation function, as we want raw output values for Q-values.
        return x
    
#TEST CODE
if __name__ == "__main__":
    # Test the DQN network
    state_dim = 12
    action_dim = 2
    # Create model
    model = DQN(state_dim, action_dim)
        
    # Create sample input
    sample_state = torch.randn(10, state_dim)
        
    # Forward pass
    output = model(sample_state)
        
    print(f"Input shape: {sample_state.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Output: {output}")