import itertools
import random
import flappy_bird_gymnasium
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import gymnasium
import torch
from torch import nn
from dqn import DQN
from exp_replay import ReplayMem, PrioritizedReplayMem
import yaml

from datetime import datetime,timedelta
import argparse

DATE_FORMAT = "%Y%m%d_%H%M%S"

RUNS_DIR = "runs"
os.makedirs(RUNS_DIR, exist_ok=True)

matplotlib.use('Agg')  # generate plots as images and save instead of displaying them.

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

class Agent: #configuration of hyperparameters from yaml file, uses Epsilon-Greedy policy for action selection, which means epsilon values
    def __init__(self, hyperparameters):
        # Load hyperparameters from YAML file
        with open("hyperparams.yml", 'r') as file:
            all_hyperparams = yaml.safe_load(file)
        hyperparameter_set = all_hyperparams[hyperparameters]
        self.hyperparameters_name = hyperparameters
        #Hyperparameters: configurable values that influence the learning of the agent. These values are tuned to optimize performance, learning time, and stability.
        self.network_sync_rate = hyperparameter_set['network_sync_rate']
        self.discount_factor_g = hyperparameter_set['discount_factor_g']
        self.replay_memory_size = hyperparameter_set['replay_memory_size']
        self.batch_size = hyperparameter_set['batch_size']
        self.epsilon_start = hyperparameter_set['epsilon_start']
        self.epsilon_decay = hyperparameter_set['epsilon_decay']
        self.learning_rate_a = hyperparameter_set['learning_rate_a']
        self.epsilon_min = hyperparameter_set['epsilon_min']
        self.stop_on_reward = hyperparameter_set['stop_on_reward']
        self.fc1_nodes = hyperparameter_set['fc1_nodes']
        self.enable_double_dqn = hyperparameter_set['enable_double_dqn']
        self.enable_dueling_dqn = hyperparameter_set['enable_dueling_dqn']
        self.use_prioritized_replay = hyperparameter_set.get('use_prioritized_replay', False)
        self.train_frequency = hyperparameter_set.get('train_frequency', 4)
        self.gradient_steps = hyperparameter_set.get('gradient_steps', 1)
        self.env_make_params = hyperparameter_set.get('env_make_params', {})
        
        #NN Error function and optimizer
        self.loss_fn = torch.nn.MSELoss() #Predicts how far the DQN's predictions are from the target Q-values. MSE is used because it punishes large errors, making learning faster and more stable.
        self.optimizer = None
        
        #Path to Run Info
        self.log_file = os.path.join(RUNS_DIR, f'{self.hyperparameters_name}.log')
        self.model_file = os.path.join(RUNS_DIR, f'{self.hyperparameters_name}.pt')
        self.graph_file = os.path.join(RUNS_DIR, f'{self.hyperparameters_name}.png')
        

    def run(self,is_train=True, render=False):
        # Only render during evaluation, never during training
        env = gymnasium.make("FlappyBird-v0", render_mode="human" if (render and not is_train) else None, use_lidar=False)
        #env = gymnasium.make("CartPole-v1", render_mode="human" if render else None)
        
        num_states = env.observation_space.shape[0] #dimension of observation space
        num_actions = env.action_space.n      #number of possible actions

        reward_history = [] # To keep track of rewards per episode
        

        policy_dqn = DQN(num_states, num_actions, self.fc1_nodes).to(device) # Create DQN model
        
        if is_train:
            # Initialize replay memory (prioritized or standard)
            if self.use_prioritized_replay:
                replay_memory = PrioritizedReplayMem(self.replay_memory_size, alpha=0.6, beta_start=0.4, seed=42)
                print("Using Prioritized Experience Replay")
            else:
                replay_memory = ReplayMem(self.replay_memory_size, seed=42)
                print("Using Standard Experience Replay")
            
            epsilon = self.epsilon_start
            
            target_dqn = DQN(num_states, num_actions, self.fc1_nodes).to(device) # Create DQN model target
            target_dqn.load_state_dict(policy_dqn.state_dict()) # Initialize target DQN with policy DQN weights
            
            #track steps taken
            step_ctr = 0
            epsilon_history = [] # To keep track of epsilon values over time
            self.optimizer = torch.optim.Adam(policy_dqn.parameters(), lr=self.learning_rate_a) # Adam optimizer for training the DQN
            
            best_reward = -float('inf')  # Start with negative infinity so any reward is better
            last_saved_time = datetime.now()
            training_start_time = datetime.now()
            best_reward_time = datetime.now()  # Track timestamp of last best reward
            last_demo_milestone = 0  # Track last reward milestone for demonstration
            demo_interval = self.stop_on_reward / 5  # Demonstrate every 1/5th of target reward
        else:
            policy_dqn.load_state_dict(torch.load(self.model_file))
            policy_dqn.eval()

        for episode in itertools.count():
            state, _ = env.reset() # Reset environment to initial state
            state = torch.tensor(state, dtype=torch.float, device=device) 
            
            terminated = False
            ep_reward = 0.0
            
            
            while not terminated:
                
                if is_train and random.random() < epsilon: # Probability check for taking random action
                    # exploring random action
                    action = env.action_space.sample() #taking the action and saving as tensor
                    action = torch.tensor(action, dtype=torch.long, device=device)
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
                    replay_memory.push((state, action, reward, next_state, terminated)) # Store transition in replay memory(converted to tuple and stored)
                    
                    step_ctr += 1
                    
                    # Train multiple times per step when enough experience is collected
                    if len(replay_memory) > self.batch_size and step_ctr % self.train_frequency == 0:
                        for _ in range(self.gradient_steps):
                            # Sample from replay memory (prioritized or standard)
                            if self.use_prioritized_replay:
                                minibatch, indices, weights = replay_memory.sample(self.batch_size)
                                td_errors = self.optimize(minibatch, policy_dqn, target_dqn, weights)
                                # Update priorities based on TD errors
                                replay_memory.update_priorities(indices, td_errors.detach().cpu().numpy())
                            else:
                                minibatch = replay_memory.sample(self.batch_size)
                                self.optimize(minibatch, policy_dqn, target_dqn)
                        
                        # Sync target network
                        if step_ctr > self.network_sync_rate:
                            target_dqn.load_state_dict(policy_dqn.state_dict())
                            step_ctr = 0
                    
                state = next_state # Move to the next state
                
            reward_history.append(ep_reward) # Log total reward for the episode
            
            if is_train:
                # Visual demonstration at reward milestones (every 1/5th of stop_on_reward)
                if best_reward > 0:  # Only check milestones for positive rewards
                    current_milestone = int(best_reward // demo_interval) * demo_interval
                    if current_milestone > last_demo_milestone and current_milestone >= demo_interval:
                        print(f"\n=== Reward Milestone {current_milestone}: Running visual demonstration ===")
                        self.demonstrate_agent(policy_dqn, num_states, num_actions)
                        print(f"=== Resuming training ===\n")
                        last_demo_milestone = current_milestone
                
                if ep_reward > best_reward:
                    # Calculate time elapsed since last best reward
                    current_time = datetime.now()
                    time_elapsed = current_time - best_reward_time
                    total_elapsed = current_time - training_start_time
                    
                    hours, remainder = divmod(time_elapsed.total_seconds(), 3600)
                    minutes, seconds = divmod(remainder, 60)
                    time_str = f"{int(hours)}h {int(minutes)}m {int(seconds)}s"
                    
                    total_hours, total_remainder = divmod(total_elapsed.total_seconds(), 3600)
                    total_minutes, total_seconds = divmod(total_remainder, 60)
                    total_time_str = f"{int(total_hours)}h {int(total_minutes)}m {int(total_seconds)}s"
                    
                    # Calculate absolute improvement
                    improvement = ep_reward - best_reward
                    
                    # Calculate percentage improvement (handle negative best_reward properly)
                    if best_reward != -float('inf') and best_reward != 0:
                        if best_reward > 0:
                            percent_improvement = (improvement / best_reward) * 100
                        else:
                            # For negative rewards, show absolute improvement
                            percent_improvement = None
                    else:
                        percent_improvement = None
                    
                    if percent_improvement is not None:
                        log_msg = f"Episode {episode}: New best reward {ep_reward:.2f} (previous {best_reward:.2f}, +{improvement:.2f}, +{percent_improvement:.1f}%, time since last: {time_str}, total time: {total_time_str}). Saving model."
                    else:
                        log_msg = f"Episode {episode}: New best reward {ep_reward:.2f} (previous {best_reward:.2f}, +{improvement:.2f}, time since last: {time_str}, total time: {total_time_str}). Saving model."
                    
                    print(log_msg)
                    with open(self.log_file, 'a') as log_f:
                        log_f.write(log_msg + '\n')
                    
                    torch.save(policy_dqn.state_dict(), self.model_file)
                    best_reward = ep_reward
                    best_reward_time = current_time  # Update timestamp
                
                current_time = datetime.now()
                if current_time - last_saved_time > timedelta(seconds=10):
                    self.save_graph(reward_history,epsilon_history)
                    last_saved_time = current_time
                    
                epsilon = max(epsilon * self.epsilon_decay, self.epsilon_min) # Decay epsilon
                epsilon_history.append(epsilon)
                
                # Check if we've reached the stopping reward
                if ep_reward >= self.stop_on_reward:
                    total_training_duration = datetime.now() - training_start_time
                    hours, remainder = divmod(total_training_duration.total_seconds(), 3600)
                    minutes, seconds = divmod(remainder, 60)
                    
                    success_msg = f"\n{'='*60}\n🎉 SUCCESS! Reached target reward of {self.stop_on_reward}!\n"
                    success_msg += f"Episode: {episode}\n"
                    success_msg += f"Best Reward: {best_reward:.2f}\n"
                    success_msg += f"Total Training Time: {int(hours)}h {int(minutes)}m {int(seconds)}s\n"
                    success_msg += f"{'='*60}\n"
                    
                    print(success_msg)
                    with open(self.log_file, 'a') as log_f:
                        log_f.write(success_msg)
                    
                    # Final demonstration
                    print("Running final demonstration...")
                    self.demonstrate_agent(policy_dqn, num_states, num_actions)
                    break
    def demonstrate_agent(self, policy_dqn, num_states, num_actions):
        """Run a visual demonstration episode to show current learning progress"""
        demo_env = gymnasium.make("FlappyBird-v0", render_mode="human", use_lidar=False)
        demo_state, _ = demo_env.reset()
        demo_state = torch.tensor(demo_state, dtype=torch.float, device=device)
        demo_terminated = False
        demo_reward = 0.0
        
        policy_dqn.eval()  # Set to evaluation mode
        
        while not demo_terminated:
            with torch.no_grad():
                action = policy_dqn(demo_state.unsqueeze(dim=0)).squeeze().argmax()
            
            next_state, reward, demo_terminated, _, _ = demo_env.step(action.item())
            demo_reward += reward
            demo_state = torch.tensor(next_state, dtype=torch.float, device=device)
        
        demo_env.close()
        policy_dqn.train()  # Set back to training mode
        print(f"Demonstration reward: {demo_reward:.2f}")
    
    def save_graph(self, reward_history, epsilon_history):
        fig = plt.figure(figsize=(12, 5))
        
        mean_rewards = np.zeros(len(reward_history))
        for i in range(len(mean_rewards)):
            mean_rewards[i] = np.mean(reward_history[max(0, i-99):(i+1)])
        plt.subplot(121)
        plt.ylabel('Mean Reward')
        plt.plot(mean_rewards, label='Mean Reward (100 episodes)')
        
        plt.subplot(122)
        plt.ylabel('Epsilon')
        plt.plot(epsilon_history, label='Epsilon')
        
        plt.subplots_adjust(wspace=1.0,hspace = 1.0)
        #save
        plt.savefig(self.graph_file)
        plt.close(fig)
        
    def optimize(self, minibatch, policy_dqn, target_dqn, weights=None): #stacking creates batch tensors, which are then used to compute current and target Q-values
        states, actions, rewards, next_states, dones = zip(*minibatch)
        
        states = torch.stack(states)
        actions = torch.stack(actions)
        rewards = torch.stack(rewards)
        next_states = torch.stack(next_states)
        dones = torch.tensor(dones).float().to(device)
        
        # Next Q values from target network
        with torch.no_grad():
            #Double DQN logic
            if self.enable_double_dqn: #if enabled, use policy DQN to select best actions, then evaluate with target DQN
                best_actions = policy_dqn(next_states).argmax(1)
                target_q_values = rewards + (1-dones) * self.discount_factor_g * target_dqn(next_states).gather(1, best_actions.unsqueeze(1)).squeeze() #Implementing the Bellman equation for target Q-values, but using Double DQN from best actions
            else: #otherwise, use standard DQN target calculation
                target_q_values = rewards + (1-dones) * self.discount_factor_g * target_dqn(next_states).max(1)[0] #Implementing the Bellman equation for target Q-values
                # The Bellman equation is used to compute the target Q-values, which represent the expected future rewards.
        
        # Current Q values
        current_q_values = policy_dqn(states).gather(1, index = actions.unsqueeze(1)).squeeze() #Gets the current Q-Values for the actions taken in the sampled transitions
        
        # Compute TD errors (for prioritized replay)
        td_errors = current_q_values - target_q_values
        
        # Apply importance sampling weights if using prioritized replay
        if weights is not None:
            weights = torch.tensor(weights, dtype=torch.float, device=device)
            # Weighted MSE loss
            loss = (weights * (td_errors ** 2)).mean()
        else:
            # Standard MSE loss
            loss = self.loss_fn(current_q_values, target_q_values)
        
        # Optimize the model
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(policy_dqn.parameters(), max_norm=10)  # Gradient clipping
        self.optimizer.step()
        
        # Return TD errors for priority updates
        return td_errors
        


#Testing            
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train or test the DQN agent on FlappyBird-v0 environment.")
    parser.add_argument('hyperparameters', help = '')
    parser.add_argument('--train', action='store_true', help='Training Mode')
    args = parser.parse_args()
    
    dq1 = Agent(hyperparameters=args.hyperparameters)
    
    if args.train:
        dq1.run(is_train=True)
    else:
        dq1.run(is_train=False, render=True)