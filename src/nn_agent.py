import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from game_state import SnakeGame


class SnakeNet(nn.Module):
    """Neural network for snake game."""
    def __init__(self):
        super(SnakeNet, self).__init__()
        # Input: snake position (2), food position (2), 
        # obstacles (4), walls (4), snake length (1)
        self.fc1 = nn.Linear(13, 32)
        self.fc2 = nn.Linear(32, 32)
        self.fc3 = nn.Linear(32, 4)  # 4 possible moves

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)


class NeuralNetAI:
    """AI agent using neural network for snake game."""
    def __init__(self):
        self.model = SnakeNet()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=0.001)
        self.actions = ['up', 'down', 'left', 'right']
        self.epsilon = 1.0  # Exploration rate
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995

    def get_state(self, game):
        """Get game state as neural network input."""
        head_x, head_y = game.snake[-1]
        food_x, food_y = game.food
        
        # Normalize positions
        head_x = head_x / game.w
        head_y = head_y / game.h
        food_x = food_x / game.w
        food_y = food_y / game.h
        
        # Check immediate surroundings for obstacles
        obstacles = []
        for dx, dy in [(0, 1), (0, -1), (-1, 0), (1, 0)]:
            next_x = (game.snake[-1][0] + dx) % game.w
            next_y = (game.snake[-1][1] + dy) % game.h
            obstacles.append(1.0 if (next_x, next_y) in game.snake[:-1] else 0.0)
        
        # Check distance to walls
        walls = [
            head_y,  # Distance to top wall
            1 - head_y,  # Distance to bottom wall
            head_x,  # Distance to left wall
            1 - head_x  # Distance to right wall
        ]
        
        # Normalize snake length
        length = len(game.snake) / (game.w * game.h)
        
        state = [
            head_x, head_y,  # Snake head position
            food_x, food_y,  # Food position
            *obstacles,  # Obstacle positions
            *walls,  # Wall distances
            length  # Snake length
        ]
        return torch.FloatTensor(state)

    def next_move(self, game):
        """Choose next move using epsilon-greedy strategy."""
        if np.random.random() < self.epsilon:
            return np.random.choice(self.actions)
        
        state = self.get_state(game)
        with torch.no_grad():
            output = self.model(state)
            # Mask out dangerous moves
            for i, action in enumerate(self.actions):
                dx, dy = {
                    'up': (0, 1),
                    'down': (0, -1),
                    'left': (-1, 0),
                    'right': (1, 0)
                }[action]
                next_x = (game.snake[-1][0] + dx) % game.w
                next_y = (game.snake[-1][1] + dy) % game.h
                if (next_x, next_y) in game.snake[:-1]:
                    output[i] = float('-inf')
            return self.actions[torch.argmax(output).item()]

    def train(self, episodes=1000):
        """Train the neural network through gameplay."""
        for episode in range(episodes):
            game = SnakeGame(20, 20)
            episode_reward = 0
            steps = 0
            
            while True:
                state = self.get_state(game)
                action = self.next_move(game)
                
                # Take action
                status = getattr(game, f"move_{action}")()
                
                # Calculate reward
                reward = 0
                if status == 'game_over':
                    reward = -10
                elif status == 'food':
                    reward = 10
                else:
                    # Reward for staying alive
                    reward = 0.1
                    
                    # Reward for moving towards food
                    head_x, head_y = game.snake[-1]
                    food_x, food_y = game.food
                    current_dist = abs(head_x - food_x) + abs(head_y - food_y)
                    if current_dist < steps:
                        reward += 1
                
                episode_reward += reward
                
                # Update network
                self.optimizer.zero_grad()
                output = self.model(state)
                action_idx = self.actions.index(action)
                loss = -torch.log(output[action_idx]) * reward
                loss.backward()
                self.optimizer.step()
                
                # Decay exploration rate
                self.epsilon = max(
                    self.epsilon_min,
                    self.epsilon * self.epsilon_decay
                )
                
                steps += 1
                if status == 'game_over' or steps > 1000:
                    print(
                        f"Episode {episode}: Score {len(game.snake)}, "
                        f"Steps {steps}, Epsilon {self.epsilon:.2f}"
                    )
                    break

