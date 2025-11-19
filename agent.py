import itertools
import random
import flappy_bird_gymnasium
import gymnasium
import torch
from dqn import DQN
from exp_replay import ReplayMem
import yaml

device = "cuda" if torch.cuda.is_available() else "cpu"

class Agent: #configuration of hyperparameters from yaml file, uses Epsilon-Greedy policy for action selection, which means epsilon values
    def __init__(self, hyperparameters):
        # Load hyperparameters from YAML file
        with open("hyperparameters.yml", 'r') as file:
            all_hyperparams = yaml.safe_load(file)
            hyperparameters = all_hyperparams[hyperparameters]
        
        self.replay_memory_size = hyperparameters['replay_memory_size']
        self.batch_size = hyperparameters['batch_size']
        self.epsilon_start = hyperparameters['epsilon_start']
        self.epsilon_decay = hyperparameters['epsilon_decay']
        self.epsilon_min = hyperparameters['epsilon_min']
        
            

    def run(self,is_train=True, render=False):
        #env = gymnasium.make("FlappyBird-v0", render_mode="human", use_lidar=False) # Initialize the environment
        env = gymnasium.make("CartPole-v1", render_mode="human" if render else None, use_lidar=False)
        
        num_states = env.observation_space.shape[0] #dimension of observation space
        num_actions = env.action_space.n      #number of possible actions

        reward_history = [] # To keep track of rewards per episode
        epsilon_history = [] # To keep track of epsilon values over time

        policy_dqn = DQN(num_states, num_actions) # Create DQN model
        
        if is_train:
            replay_memory = ReplayMem(self.replay_memory_size, seed=42) # Initialize replay memory with seed 42(Hitchhiker's Guide to the Galaxy hehe)
            
            epsilon = self.epsilon_start

        for episode in itertools.count():
            state, _ = env.reset() # Reset environment to initial state
            terminated = False
            ep_reward = 0.0
            
            
            while not terminated:
                
                if is_train and random.random() < epsilon:
                    # exploring random action
                    action = env.action_space.sample()
                else:
                    action = policy_dqn(state).argmax().item() # Finding best action from DQN
                
                # next action
                action = env.action_space.sample() #returns 0(do nothing) or 1(flap)

                #processing phase
                next_state, reward, terminated, _, info = env.step(action) #environment observation after action
                
                # Acc Reward
                ep_reward += reward
                
                if is_train:
                    replay_memory.push((state, action, reward, next_state, terminated)) # Store transition in replay memory
                    
                state = next_state # Move to the next state
                
            reward_history.append(ep_reward) # Log total reward for the episode
            
            epsilon = max(epsilon * self.epsilon_decay, self.epsilon_min) # Decay epsilon
            epsilon_history.append(epsilon)