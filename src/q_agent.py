import numpy as np
from game_state import SnakeGame


class QLearningAI:
    def __init__(self, learning_rate=0.1, discount_factor=0.95, epsilon=1.0):
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.epsilon = epsilon
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.q_table = {}
        self.actions = ['up', 'down', 'left', 'right']
        self.steps_since_food = 0

    def get_state_key(self, game):
        """Convert game state to a tuple for use as a dictionary key."""
        head_x, head_y = game.snake[-1]
        food_x, food_y = game.food
        
        # Get relative food position
        food_dx = food_x - head_x
        food_dy = food_y - head_y
        
        # Check immediate surroundings for obstacles
        obstacles = []
        for dx, dy in [(0, 1), (0, -1), (-1, 0), (1, 0)]:
            next_x = (head_x + dx) % game.w
            next_y = (head_y + dy) % game.h
            is_danger = (next_x, next_y) in game.snake[:-1]
            obstacles.append(1 if is_danger else 0)
        
        # Get direction to food (normalized)
        food_dir_x = 1 if food_dx > 0 else (-1 if food_dx < 0 else 0)
        food_dir_y = 1 if food_dy > 0 else (-1 if food_dy < 0 else 0)
        
        # Create state tuple
        return (
            food_dir_x,  # Direction to food (x)
            food_dir_y,  # Direction to food (y)
            *obstacles,  # Obstacle positions
            len(game.snake)  # Snake length
        )

    def get_q_value(self, state, action):
        """Get Q-value for state-action pair."""
        if state not in self.q_table:
            self.q_table[state] = {a: 0.0 for a in self.actions}
        return self.q_table[state][action]

    def next_move(self, game):
        """Choose next move using epsilon-greedy strategy."""
        state = self.get_state_key(game)
        
        if np.random.random() < self.epsilon:
            # Only choose from safe moves during exploration
            safe_moves = []
            for action in self.actions:
                dx, dy = {
                    'up': (0, 1),
                    'down': (0, -1),
                    'left': (-1, 0),
                    'right': (1, 0)
                }[action]
                next_x = (game.snake[-1][0] + dx) % game.w
                next_y = (game.snake[-1][1] + dy) % game.h
                if (next_x, next_y) not in game.snake[:-1]:
                    safe_moves.append(action)
            
            return (np.random.choice(safe_moves) if safe_moves 
                   else np.random.choice(self.actions))
        
        # Get Q-values for all actions
        q_values = [self.get_q_value(state, action) for action in self.actions]
        
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
                q_values[i] = float('-inf')
        
        return self.actions[np.argmax(q_values)]

    def update(self, state, action, reward, next_state):
        """Update Q-value for state-action pair."""
        old_value = self.get_q_value(state, action)
        next_max = max([
            self.get_q_value(next_state, a) for a in self.actions
        ])
        new_value = (1 - self.learning_rate) * old_value + self.learning_rate * (
            reward + self.discount_factor * next_max
        )
        self.q_table[state][action] = new_value

    def train(self, episodes=1000):
        """Train the agent through gameplay."""
        for episode in range(episodes):
            game = SnakeGame(20, 20)
            episode_reward = 0
            self.steps_since_food = 0
            last_distance = float('inf')
            
            while True:
                state = self.get_state_key(game)
                action = self.next_move(game)
                
                # Take action
                status = getattr(game, f"move_{action}")()
                
                # Calculate reward
                reward = 0
                if status == 'game_over':
                    reward = -20
                elif status == 'food':
                    reward = 20
                    self.steps_since_food = 0
                else:
                    # Reward for moving towards food
                    head_x, head_y = game.snake[-1]
                    food_x, food_y = game.food
                    current_distance = abs(head_x - food_x) + abs(head_y - food_y)
                    
                    if current_distance < last_distance:
                        reward = 2
                    else:
                        reward = -1
                    
                    # Penalty for taking too long
                    self.steps_since_food += 1
                    if self.steps_since_food > 50:
                        reward -= 0.5
                    
                    last_distance = current_distance
                
                # Get next state
                next_state = self.get_state_key(game)
                
                # Update Q-value
                self.update(state, action, reward, next_state)
                
                # Decay epsilon
                self.epsilon = max(
                    self.epsilon_min,
                    self.epsilon * self.epsilon_decay
                )
                
                episode_reward += reward
                
                if status == 'game_over' or self.steps_since_food > 100:
                    print(
                        f"Episode {episode}: Score {len(game.snake)}, "
                        f"Steps {self.steps_since_food}, "
                        f"Epsilon {self.epsilon:.2f}"
                    )
                    break
