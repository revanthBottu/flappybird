import itertools
import random
import flappy_bird_gymnasium
import gymnasium
import torch
from torch import nn
from dqn import DQN
from exp_replay import ReplayMem
import yaml

device = "cuda" if torch.cuda.is_available() else "cpu"

class Agent: #configuration of hyperparameters from yaml file, uses Epsilon-Greedy policy for action selection, which means epsilon values
    def __init__(self, hyperparameters):
        # Load hyperparameters from YAML file
        with open("hyperparams.yml", 'r') as file:
            all_hyperparams = yaml.safe_load(file)
            hyperparameters = all_hyperparams[hyperparameters]
        #Hyperparameters: configurable values that influence the learning of the agent. These values are tuned to optimize performance, learning time, and stability.
        self.network_sync_rate = hyperparameters['network_sync_rate']
        self.discount_factor_g = hyperparameters['discount_factor_g']
        self.replay_memory_size = hyperparameters['replay_memory_size']
        self.batch_size = hyperparameters['batch_size']
        self.epsilon_start = hyperparameters['epsilon_start']
        self.epsilon_decay = hyperparameters['epsilon_decay']
        self.learning_rate_a = hyperparameters['learning_rate_a']
        self.epsilon_min = hyperparameters['epsilon_min']
        
        self.loss_fn = torch.nn.MSELoss() #Predicts how far the DQN's predictions are from the target Q-values. MSE is used because it punishes large errors, making learning faster and more stable.
        self.optimizer = None    

    def run(self,is_train=True, render=False):
        #env = gymnasium.make("FlappyBird-v0", render_mode="human", use_lidar=False) # Initialize the environment
        env = gymnasium.make("CartPole-v1", render_mode="human" if render else None)
        
        num_states = env.observation_space.shape[0] #dimension of observation space
        num_actions = env.action_space.n      #number of possible actions

        reward_history = [] # To keep track of rewards per episode
        epsilon_history = [] # To keep track of epsilon values over time

        policy_dqn = DQN(num_states, num_actions).to(device) # Create DQN model
        
        if is_train:
            replay_memory = ReplayMem(self.replay_memory_size, seed=42) # Initialize replay memory with seed 42(Hitchhiker's Guide to the Galaxy hehe)
            
            epsilon = self.epsilon_start
            
            target_dqn = DQN(num_states, num_actions).to(device) # Create DQN model target
            target_dqn.load_state_dict(policy_dqn.state_dict()) # Initialize target DQN with policy DQN weights
            
            #track steps taken
            step_ctr = 0
            
            self.optimizer = torch.optim.Adam(policy_dqn.parameters(), lr=self.learning_rate_a) # Adam optimizer for training the DQN

        for episode in itertools.count():
            state, _ = env.reset() # Reset environment to initial state
            state = torch.tensor(state, dtype=torch.float, device=device) 
            
            terminated = False
            ep_reward = 0.0
            
            
            while not terminated:
                
                if is_train and random.random() < epsilon: # Probability check for taking random action
                    # exploring random action
                    action = env.action_space.sample() #taking the action and saving as tensor
                    action = torch.tensor(action, dtype=torch.float, device=device)
                else:
                    with torch.no_grad():
                        action = policy_dqn(state.unsqueeze(dim = 0)).squeeze().argmax() # Finding best action from DQN
                        #unsqueeze to add batch dimension at the beginning, squeeze so that output is 1D tensor
    

                #processing phase
                next_state, reward, terminated, _, info = env.step(action.item()) #environment observation after action
                
                
                # Acc Reward
                ep_reward += reward
                
                next_state = torch.tensor(next_state, dtype=torch.float, device=device)
                reward = torch.tensor(reward, dtype=torch.float, device=device)
                
                if is_train:
                    replay_memory.push((state, action, reward, next_state, terminated)) # Store transition in replay memory
                    
                    step_ctr += 1
                    
                state = next_state # Move to the next state
                
            reward_history.append(ep_reward) # Log total reward for the episode
            
            epsilon = max(epsilon * self.epsilon_decay, self.epsilon_min) # Decay epsilon
            epsilon_history.append(epsilon)
            
            if len(replay_memory) > self.batch_size:
                
                #get a sample from replay memory
                minibatch = replay_memory.sample(self.batch_size)
                self.optimize(minibatch, policy_dqn, target_dqn)
                
                if step_ctr > self.network_sync_rate:
                    target_dqn.load_state_dict(policy_dqn.state_dict())
                    step_ctr = 0
    
    def optimize(self, minibatch, policy_dqn, target_dqn):
        
        for state, action, new_state, reward, terminated in minibatch:
            
            if terminated:
                target = reward
            else:
                with torch.no_grad():
                    target_q = reward + self.discount_factor_g * target_dqn(new_state).max()
            
            current_q = policy_dqn(state)
            
            loss = self.loss_fn(current_q, target_q)
            
            #Model Optimization
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()                
            
if __name__ == "__main__":
    agent = Agent('cartpole1')
    agent.run(is_train=True, render=True)