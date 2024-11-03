import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from collections import deque
import os
import random
import pandas as pd 
import sys
sys.path.append('.')

from utilities.game import Agent, Actions
from pacman import Directions
from utilities import util

class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)
    
    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))
    
    def sample(self, batch_size):
        return random.sample(self.buffer, batch_size)
    
    def __len__(self):
        return len(self.buffer)
class DQN(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(DQN, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 32),
            nn.ReLU(),
            nn.Linear(32, output_dim)
        )
    
    def forward(self, x):
        return self.network(x)

class DeepQLearningAgent(Agent):
    def __init__(self, state_size=12, epsilon=1.0, epsilon_min=0.01,
                 epsilon_decay=0.995, gamma=0.99, learning_rate=0.1,
                 memory_size=50000, batch_size=64, target_update=500,
                 numTraining=100):
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")

        self.state_size = state_size
        self.memory = ReplayBuffer(memory_size)
        self.batch_size = batch_size
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.target_update = target_update
        self.numTraining = numTraining
        self.episodesSoFar = 0
        
        # Actions
        self.actions = [Directions.NORTH, Directions.SOUTH, Directions.EAST, Directions.WEST, Directions.STOP]
        self.action_size = len(self.actions)
        
        # Networks
        self.q_network = DQN(state_size, self.action_size).to(self.device)
        self.target_network = DQN(state_size, self.action_size).to(self.device)
        self.target_network.load_state_dict(self.q_network.state_dict())
        
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=learning_rate)
        
        # Tracking
        self.episodeRewards = 0
        self.accumTrainRewards = 0
        self.rewards = []
        self.losses = []
        self.steps = 0
        self.lastState = None
        self.lastAction = None
        self.last_score = 0

    def getFeatures(self, state):
        features = np.zeros(self.state_size)
        pacman_pos = state.getPacmanPosition()
        food = state.getFood()
        ghosts = state.getGhostStates()
        capsules = state.getCapsules()
        
        # Score and basic info
        features[0] = state.getScore() / 500.0  # Normalize score
        features[1] = len(food.asList()) / (food.width * food.height)  # Food ratio
        
        # Pacman position
        features[2:4] = [pos/food.width for pos in pacman_pos]  # Normalized position
        
        # Closest food distance
        food_list = food.asList()
        if food_list:
            min_food_dist = min([util.manhattanDistance(pacman_pos, food) for food in food_list])
            features[4] = min_food_dist / (food.width + food.height)
        
        # Ghost features
        for i, ghost in enumerate(ghosts[:2]):
            ghost_pos = ghost.getPosition()
            dist = util.manhattanDistance(pacman_pos, ghost_pos)
            features[5+i*2] = dist / (food.width + food.height)
            # Is ghost scared?
            features[6+i*2] = float(ghost.scaredTimer > 0)
        
        # Capsule features
        if capsules:
            min_capsule_dist = min([util.manhattanDistance(pacman_pos, caps) for caps in capsules])
            features[9] = min_capsule_dist / (food.width + food.height)
            features[10] = len(capsules) / (food.width * food.height)
        
        # Legal actions
        features[11] = len(state.getLegalPacmanActions()) / 5.0  # Normalize by max actions
        
        return torch.FloatTensor(features).to(self.device)

    def getAction(self, state):
        legal = state.getLegalPacmanActions()
        
        # Remove STOP if possible
        if len(legal) > 1 and Directions.STOP in legal:
            legal.remove(Directions.STOP)
            
        # Epsilon-greedy
        if random.random() < self.epsilon:
            action = random.choice(legal)
        else:
            state_tensor = self.getFeatures(state).unsqueeze(0)
            with torch.no_grad():
                q_values = self.q_network(state_tensor)
                
            # Filter legal actions
            legal_q_values = {}
            for action in legal:
                if action in self.actions:
                    idx = self.actions.index(action)
                    legal_q_values[action] = q_values[0][idx].item()
            
            action = max(legal_q_values.items(), key=lambda x: x[1])[0] if legal_q_values else random.choice(legal)
        
        # Store for learning
        self.lastState = state
        self.lastAction = action
        self.last_score = state.getScore()
        
        return action

    def update(self, state, action, nextState, reward):
        # Store transition
        self.memory.push(
            self.getFeatures(state),
            self.actions.index(action),
            reward,
            self.getFeatures(nextState),
            nextState.isWin() or nextState.isLose()  # Done flag
        )
        
        # Training step
        if len(self.memory) >= self.batch_size:
            batch = self.memory.sample(self.batch_size)
            state_batch = torch.stack([s for s, _, _, _, _ in batch])
            action_batch = torch.tensor([a for _, a, _, _, _ in batch], device=self.device)
            reward_batch = torch.tensor([r for _, _, r, _, _ in batch], device=self.device)
            next_state_batch = torch.stack([ns for _, _, _, ns, _ in batch])
            done_batch = torch.tensor([d for _, _, _, _, d in batch], device=self.device)
            
            # Current Q values
            current_q_values = self.q_network(state_batch).gather(1, action_batch.unsqueeze(1))
            
            # Next Q values
            with torch.no_grad():
                next_q_values = self.target_network(next_state_batch).max(1)[0]
                next_q_values[done_batch] = 0.0  # Terminal states
                target_q_values = reward_batch + self.gamma * next_q_values
            
            # Update
            loss = nn.MSELoss()(current_q_values.squeeze(), target_q_values)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            
            self.losses.append(loss.item())
            print(f"Loss: {loss.item():.2f}")
            
            # Update target network
            if self.steps % self.target_update == 0:
                self.target_network.load_state_dict(self.q_network.state_dict())
            
            self.steps += 1

    def final(self, state):
        # Calculate final reward with additional win/lose bonus
        if self.lastState is not None:
            final_reward = (state.getScore() - self.last_score)  # Score difference
            if state.isWin():
                final_reward += 500
            elif state.isLose():
                final_reward -= 500
                
            self.update(self.lastState, self.lastAction, state, final_reward)
        
        self.episodesSoFar += 1
        
        # Update epsilon during training
        if self.episodesSoFar < self.numTraining:
            self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
            
        if self.episodesSoFar % 10 == 0:
            print(f"Episode {self.episodesSoFar}")
            print(f"Epsilon: {self.epsilon:.4f}")
            if self.rewards:
                print(f"Average Score (last 10): {np.mean(self.rewards[-10:]):.2f}")
        
        # Save rewards
        self.rewards.append(state.getScore())

        # save rewards when training is done
        if self.episodesSoFar == self.numTraining:
            train_data = pd.DataFrame()
            train_data['Rewards'] = self.rewards
            train_data.to_excel('data_train_pacman_DeepQLearningAgent.xlsx')
        
        # save rewards when testing is done
        if self.episodesSoFar > self.numTraining:
            test_data = pd.DataFrame()
            test_data['Rewards'] = self.rewards[self.numTraining:]
            test_data.to_excel('data_test_pacman_DeepQLearningAgent.xlsx')

        self.lastState = None
        self.lastAction = None
