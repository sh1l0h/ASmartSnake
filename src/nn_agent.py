import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import random
from collections import deque, namedtuple
from dataclasses import dataclass
from typing import List, Tuple, Dict, Optional
import matplotlib.pyplot as plt
import matplotlib

matplotlib.use('Agg')
from game_state import SnakeGame


@dataclass
class GameState:
    head_pos: Tuple[int, int]
    food_pos: Tuple[int, int]
    snake_body: List[Tuple[int, int]]
    board_width: int
    board_height: int
    score: int
    steps_since_food: int

    def get_danger_directions(self) -> List[bool]:
        hx, hy = self.head_pos
        dangers = []
        for dx, dy in [(0, 1), (0, -1), (-1, 0), (1, 0)]:
            nx = (hx + dx) % self.board_width
            ny = (hy + dy) % self.board_height
            dangers.append((nx, ny) in self.snake_body[:-1])
        return dangers

    def get_food_direction(self) -> Tuple[float, float]:
        fx, fy = self.food_pos
        hx, hy = self.head_pos
        dx = fx - hx
        dy = fy - hy
        return (
            1.0 if dx > 0 else (-1.0 if dx < 0 else 0.0),
            1.0 if dy > 0 else (-1.0 if dy < 0 else 0.0),
        )

    def get_wall_distances(self) -> List[float]:
        hx, hy = self.head_pos
        return [
            hy / self.board_height,
            (self.board_height - 1 - hy) / self.board_height,
            hx / self.board_width,
            (self.board_width - 1 - hx) / self.board_width,
        ]

    def to_neural_input(self) -> np.ndarray:
        hx, hy = self.head_pos
        fx, fy = self.food_pos

        norm_hx = hx / self.board_width
        norm_hy = hy / self.board_height
        norm_fx = fx / self.board_width
        norm_fy = fy / self.board_height

        dangers = self.get_danger_directions()
        food_dir = self.get_food_direction()
        walls = self.get_wall_distances()
        length = len(self.snake_body) / (self.board_width * self.board_height)

        return np.array(
            [
                norm_hx,
                norm_hy,
                norm_fx,
                norm_fy,
                *[1.0 if d else 0.0 for d in dangers],
                *food_dir,
                *walls,
                length,
            ],
            dtype=np.float32,
        )


@dataclass
class TrainingMetrics:
    episode: int
    score: int
    steps: int
    epsilon: float
    loss: float
    avg_score_last_100: float
    max_score: int


@dataclass
class TrainingConfig:
    """
    Config for training a NN agent.

    learning_rate: Size of weight update steps. Too high = unstable. Too low = slow.
    gamma: Discount factor. 0.95 → 10-steps-ahead reward worth ~60%.
    epsilon_start: Initial exploration rate (100% random actions).
    epsilon_end: Final exploration rate (1% random).
    epsilon_decay: Rate at which exploration decreases per episode.
    memory_size: How many past experiences to store.
    batch_size: How many memories to learn from per step.
    target_update_freq: Steps between syncing main → target network.
    max_episodes: Total training episodes.
    max_steps_per_episode: Max steps per episode before force-ending.
    early_stop_threshold: Average score threshold for early stopping.
    """

    learning_rate: float = 0.001
    epsilon_start: float = 1.0
    epsilon_end: float = 0.01
    epsilon_decay: float = 0.98
    batch_size: int = 32
    memory_size: int = 10000
    target_update_freq: int = 100
    max_episodes: int = 2000
    max_steps_per_episode: int = 1000
    gamma: float = 0.95
    early_stop_threshold: float = 15.0


Experience = namedtuple(
    'Experience',
    ['state', 'action', 'reward', 'next_state', 'done'],
)


class ExperienceBuffer:

    def __init__(self, capacity: int):
        self.buffer = deque(maxlen=capacity)

    def push(self, state: GameState, action: int, reward: float,
             next_state: Optional[GameState], done: bool):
        self.buffer.append(Experience(state, action, reward, next_state, done))

    def sample(self, batch_size: int) -> List[Experience]:
        return random.sample(self.buffer, batch_size)

    def __len__(self) -> int:
        return len(self.buffer)


class DQNNetwork(nn.Module):

    def __init__(self, input_size: int, hidden_size: int, output_size: int):
        super(DQNNetwork, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, output_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)


class TrainingVisualizer:

    def __init__(self):
        self.episodes = []
        self.scores = []
        self.avg_scores = []
        self.epsilons = []
        self.losses = []

    def update_metrics(self, metrics: TrainingMetrics):
        self.episodes.append(metrics.episode)
        self.scores.append(metrics.score)
        self.avg_scores.append(metrics.avg_score_last_100)
        self.epsilons.append(metrics.epsilon)
        if metrics.loss > 0:
            self.losses.append(metrics.loss)

    def plot_progress(self):
        plt.figure(figsize=(12, 8))

        plt.subplot(2, 2, 1)
        plt.plot(self.episodes, self.scores, alpha=0.5)
        plt.plot(self.episodes, self.avg_scores, 'r-', linewidth=2)
        plt.xlabel('Episode')
        plt.ylabel('Score')
        plt.title('Training Progress')
        plt.legend(['Score', 'Avg Score (100)'])

        plt.subplot(2, 2, 2)
        plt.plot(self.episodes, self.epsilons)
        plt.xlabel('Episode')
        plt.ylabel('Epsilon')
        plt.title('Exploration Rate')

        if self.losses:
            plt.subplot(2, 2, 3)
            plt.plot(self.losses)
            plt.xlabel('Training Step')
            plt.ylabel('Loss')
            plt.title('Training Loss')

        plt.tight_layout()
        plt.savefig('storage/nn/training_progress.png')
        plt.close()

    def save_plots(self, filepath: str):
        self.plot_progress()


class SnakeTrainer:

    def __init__(self, config: TrainingConfig, board_width: int,
                 board_height: int):
        self.config = config
        self.board_width = board_width
        self.board_height = board_height

        input_size = 15
        hidden_size = 64
        output_size = 4

        self.main_network = DQNNetwork(input_size, hidden_size, output_size)
        self.target_network = DQNNetwork(input_size, hidden_size, output_size)
        self.target_network.load_state_dict(self.main_network.state_dict())

        self.optimizer = optim.Adam(
            self.main_network.parameters(),
            lr=config.learning_rate,
        )
        self.memory = ExperienceBuffer(config.memory_size)
        self.visualizer = TrainingVisualizer()
        self.metrics_history: List[TrainingMetrics] = []
        self.epsilon = config.epsilon_start
        self.training_step = 0
        self.best_score = 0

    def train(self) -> None:
        for episode in range(self.config.max_episodes):
            metrics = self._run_episode()
            self.metrics_history.append(metrics)
            self.visualizer.update_metrics(metrics)

            if episode % 10 == 0:
                print(
                    f"Episode {episode}: Score={metrics.score}, Avg100={metrics.avg_score_last_100:.2f}, Epsilon={metrics.epsilon:.3f}"
                )

            if episode % 50 == 0:
                self.visualizer.plot_progress()

            if self._should_stop_training():
                print(f"Training stopped early at episode {episode}")
                break

        self.save_weights(
            f'storage/nn/snake_weights_{self.board_width}x{self.board_height}.pth'
        )
        self.visualizer.save_plots('storage/nn/final_training_progress.png')

    def _run_episode(self) -> TrainingMetrics:
        game = SnakeGame(self.board_width, self.board_height)
        state = self._game_to_state(game)
        total_reward = 0
        steps = 0
        episode_loss = 0
        loss_count = 0

        for step in range(self.config.max_steps_per_episode):
            action = self._select_action(state)
            action_name = ['up', 'down', 'left', 'right'][action]

            old_score = len(game.snake)
            old_distance = abs(game.snake[-1][0] -
                               game.food[0]) + abs(game.snake[-1][1] -
                                                   game.food[1])

            status = getattr(game, f"move_{action_name}")()

            reward = 0
            done = False

            if status == 'game_over':
                reward = -10
                done = True
            elif status == 'food':
                reward = 10
            else:
                new_distance = abs(game.snake[-1][0] -
                                   game.food[0]) + abs(game.snake[-1][1] -
                                                       game.food[1])
                if new_distance < old_distance:
                    reward = 1
                else:
                    reward = -0.1

            total_reward += reward

            if done:
                next_state = None
            else:
                next_state = self._game_to_state(game)

            self.memory.push(state, action, reward, next_state, done)

            if len(self.memory) >= self.config.batch_size:
                loss = self._optimize_model()
                if loss > 0:
                    episode_loss += loss
                    loss_count += 1

            if done:
                break

            state = next_state
            steps += 1

        self.epsilon = max(self.config.epsilon_end,
                           self.epsilon * self.config.epsilon_decay)

        if self.training_step % self.config.target_update_freq == 0:
            self._update_target_network()

        self.training_step += 1

        score = len(game.snake) - 1
        if score > self.best_score:
            self.best_score = score

        recent_scores = [m.score for m in self.metrics_history[-99:]] + [score]
        avg_score = sum(recent_scores) / len(recent_scores)

        avg_loss = episode_loss / loss_count if loss_count > 0 else 0

        return TrainingMetrics(
            episode=len(self.metrics_history),
            score=score,
            steps=steps,
            epsilon=self.epsilon,
            loss=avg_loss,
            avg_score_last_100=avg_score,
            max_score=self.best_score,
        )

    def _select_action(self, state: GameState) -> int:
        if random.random() < self.epsilon:
            return random.randint(0, 3)

        with torch.no_grad():
            state_tensor = torch.FloatTensor(
                state.to_neural_input()).unsqueeze(0)
            q_values = self.main_network(state_tensor)
            return q_values.argmax().item()

    def _optimize_model(self) -> float:
        batch = self.memory.sample(self.config.batch_size)

        states = torch.FloatTensor([e.state.to_neural_input() for e in batch])
        actions = torch.LongTensor([e.action for e in batch])
        rewards = torch.FloatTensor([e.reward for e in batch])

        non_final_mask = torch.tensor(
            [e.next_state is not None for e in batch],
            dtype=torch.bool,
        )
        non_final_next_states = torch.FloatTensor([
            e.next_state.to_neural_input()
            for e in batch if e.next_state is not None
        ], )

        current_q_values = self.main_network(states).gather(
            1,
            actions.unsqueeze(1),
        )

        next_q_values = torch.zeros(self.config.batch_size)
        if len(non_final_next_states) > 0:
            next_q_values[non_final_mask] = self.target_network(
                non_final_next_states).max(1)[0].detach()

        target_q_values = rewards + (self.config.gamma * next_q_values)

        loss = F.mse_loss(current_q_values.squeeze(), target_q_values)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return loss.item()

    def _update_target_network(self) -> None:
        self.target_network.load_state_dict(self.main_network.state_dict())

    def _should_stop_training(self) -> bool:
        if len(self.metrics_history) < 100:
            return False

        recent_avg = sum(m.score for m in self.metrics_history[-100:]) / 100
        if recent_avg > self.config.early_stop_threshold:
            return True

        return False

    def save_weights(self, filepath: str) -> None:
        torch.save(
            {
                'model_state_dict': self.main_network.state_dict(),
                'board_width': self.board_width,
                'board_height': self.board_height,
                'best_score': self.best_score,
            },
            filepath,
        )

    def _game_to_state(self, game: SnakeGame) -> GameState:
        return GameState(
            head_pos=game.snake[-1],
            food_pos=game.food,
            snake_body=game.snake,
            board_width=game.w,
            board_height=game.h,
            score=game.score,
            steps_since_food=0,
        )


class WeightLoader:

    @staticmethod
    def get_weight_filename(board_width: int, board_height: int) -> str:
        return f'storage/nn/snake_weights_{board_width}x{board_height}.pth'

    @staticmethod
    def load_weights(filepath: str) -> Dict:
        return torch.load(filepath, map_location=torch.device('cpu'))

    @staticmethod
    def weights_exist(board_width: int, board_height: int) -> bool:
        import os
        filename = WeightLoader.get_weight_filename(board_width, board_height)
        return os.path.exists(filename)


class NeuralNetAI:

    def __init__(self):
        self.board_width = 20
        self.board_height = 20
        self.network = DQNNetwork(15, 64, 4)
        self._load_trained_weights()
        self.actions = ['up', 'down', 'left', 'right']

    def _load_trained_weights(self) -> None:
        if WeightLoader.weights_exist(self.board_width, self.board_height):
            filename = WeightLoader.get_weight_filename(
                self.board_width,
                self.board_height,
            )
            checkpoint = WeightLoader.load_weights(filename)
            self.network.load_state_dict(checkpoint['model_state_dict'])
            self.network.eval()

    def next_move(self, game: SnakeGame) -> str:
        state = self._game_to_state(game)

        with torch.no_grad():
            state_tensor = torch.FloatTensor(
                state.to_neural_input()).unsqueeze(0)
            q_values = self.network(state_tensor)

            for i, action in enumerate(self.actions):
                dx, dy = {
                    'up': (0, 1),
                    'down': (0, -1),
                    'left': (-1, 0),
                    'right': (1, 0),
                }[action]
                next_x = (game.snake[-1][0] + dx) % game.w
                next_y = (game.snake[-1][1] + dy) % game.h
                if (next_x, next_y) in game.snake[:-1]:
                    q_values[0][i] = float('-inf')

            return self.actions[q_values.argmax().item()]

    def _game_to_state(self, game: SnakeGame) -> GameState:
        return GameState(
            head_pos=game.snake[-1],
            food_pos=game.food,
            snake_body=game.snake,
            board_width=game.w,
            board_height=game.h,
            score=game.score,
            steps_since_food=0,
        )


class NNRunnerAgent(NeuralNetAI):

    def __init__(self, board_width: int, board_height: int):
        self.board_width = board_width
        self.board_height = board_height
        self.network = DQNNetwork(15, 64, 4)
        self._load_trained_weights()
        self.actions = ['up', 'down', 'left', 'right']


if __name__ == '__main__':
    config = TrainingConfig()
    trainer = SnakeTrainer(config, 20, 20)
    trainer.train()
