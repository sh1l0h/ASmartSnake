import random
import pickle
from collections import defaultdict
from dataclasses import dataclass
from typing import List, Tuple, Dict, Optional
import matplotlib.pyplot as plt
import matplotlib
import os

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

    def get_food_direction(self) -> Tuple[int, int]:
        fx, fy = self.food_pos
        hx, hy = self.head_pos
        dx = fx - hx
        dy = fy - hy
        return (
            1 if dx > 0 else (-1 if dx < 0 else 0),
            1 if dy > 0 else (-1 if dy < 0 else 0),
        )

    def to_state_key(self) -> Tuple:
        """Convert state to tuple key for Q-table lookup."""
        food_dir = self.get_food_direction()
        dangers = self.get_danger_directions()

        return (
            food_dir[0],  # Food direction X
            food_dir[1],  # Food direction Y
            *[1 if d else 0 for d in dangers],  # Danger in each direction
            min(len(self.snake_body), 10),  # Snake length (capped)
        )


@dataclass
class TrainingMetrics:
    episode: int
    score: int
    steps: int
    epsilon: float
    avg_score_last_100: float
    max_score: int
    q_table_size: int


@dataclass
class TrainingConfig:
    """
    Config for training a Q-learning agent.

    learning_rate: Alpha parameter for Q-update. 0.1 = blend 10% new, 90% old.
    gamma: Discount factor. 0.95 → future rewards decay exponentially.
    epsilon_start: Initial exploration rate (100% random actions).
    epsilon_end: Final exploration rate (1% random).
    epsilon_decay: Rate at which exploration decreases per episode.
    max_episodes: Total training episodes.
    max_steps_per_episode: Max steps per episode before force-ending.
    early_stop_threshold: Average score threshold for early stopping.
    """

    learning_rate: float = 0.1
    epsilon_start: float = 1.0
    epsilon_end: float = 0.01
    epsilon_decay: float = 0.995
    max_episodes: int = 1000
    max_steps_per_episode: int = 1000
    gamma: float = 0.95
    early_stop_threshold: float = 50.0


class QTable:
    """Q-table implementation with defaultdict for automatic initialization."""

    def __init__(self):
        self.table = defaultdict(lambda: defaultdict(float))

    def get_q_value(self, state_key: Tuple, action: int) -> float:
        return self.table[state_key][action]

    def set_q_value(self, state_key: Tuple, action: int, value: float):
        self.table[state_key][action] = value

    def get_best_action(
        self,
        state_key: Tuple,
        valid_actions: List[int],
    ) -> int:
        if state_key not in self.table:
            return random.choice(valid_actions)

        q_values = [self.table[state_key][a] for a in valid_actions]
        max_q = max(q_values)
        best_actions = [
            a for a, q in zip(valid_actions, q_values) if q == max_q
        ]
        return random.choice(best_actions)

    def get_max_q_value(self, state_key: Tuple) -> float:
        if state_key not in self.table:
            return 0.0
        return max(
            self.table[state_key].values()) if self.table[state_key] else 0.0

    def size(self) -> int:
        return len(self.table)


class TrainingVisualizer:

    def __init__(self):
        self.episodes = []
        self.scores = []
        self.avg_scores = []
        self.epsilons = []
        self.q_table_sizes = []

    def update_metrics(self, metrics: TrainingMetrics):
        self.episodes.append(metrics.episode)
        self.scores.append(metrics.score)
        self.avg_scores.append(metrics.avg_score_last_100)
        self.epsilons.append(metrics.epsilon)
        self.q_table_sizes.append(metrics.q_table_size)

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

        plt.subplot(2, 2, 3)
        plt.plot(self.episodes, self.q_table_sizes)
        plt.xlabel('Episode')
        plt.ylabel('Q-Table Size')
        plt.title('Q-Table Growth')

        plt.tight_layout()
        plt.savefig('storage/q/training_progress.png')
        plt.close()

    def save_plots(self, filepath: str):
        self.plot_progress()


class QTableTrainer:

    def __init__(
        self,
        config: TrainingConfig,
        board_width: int,
        board_height: int,
    ):
        self.config = config
        self.board_width = board_width
        self.board_height = board_height

        self.q_table = QTable()
        self.visualizer = TrainingVisualizer()
        self.metrics_history: List[TrainingMetrics] = []
        self.epsilon = config.epsilon_start
        self.best_score = 0
        self.actions = [0, 1, 2, 3]  # up, down, left, right

    def train(self) -> None:
        for episode in range(self.config.max_episodes):
            metrics = self._run_episode()
            self.metrics_history.append(metrics)
            self.visualizer.update_metrics(metrics)

            if episode % 10 == 0:
                print(
                    f"Episode {episode}: Score={metrics.score}, ",
                    f"Avg100={metrics.avg_score_last_100:.2f}, ",
                    f"Epsilon={metrics.epsilon:.3f}, ",
                    f"Q-Table Size={metrics.q_table_size}",
                )

            if episode % 50 == 0:
                self.visualizer.plot_progress()

            if self._should_stop_training():
                print(f"Training stopped early at episode {episode}")
                break

        self.save_q_table(
            f'storage/q/snake_table_{self.board_width}x{self.board_height}.pkl'
        )
        self.visualizer.save_plots('storage/q/final_training_progress.png')

    def _run_episode(self) -> TrainingMetrics:
        game = SnakeGame(self.board_width, self.board_height)
        state = self._game_to_state(game)
        steps = 0
        last_distance = self._manhattan_distance(game.snake[-1], game.food)

        for step in range(self.config.max_steps_per_episode):
            state_key = state.to_state_key()
            action = self._select_action(state_key, game)

            old_score = len(game.snake)
            action_name = ['up', 'down', 'left', 'right'][action]
            status = getattr(game, f"move_{action_name}")()

            reward = 0
            done = False

            if status == 'game_over':
                reward = -20
                done = True
            elif status == 'food':
                reward = 20
                state.steps_since_food = 0
            else:
                # Reward for moving towards food
                new_distance = self._manhattan_distance(
                    game.snake[-1],
                    game.food,
                )
                if new_distance < last_distance:
                    reward = 2
                else:
                    reward = -1

                # Penalty for taking too long
                state.steps_since_food += 1
                if state.steps_since_food > 50:
                    reward -= 0.5

                last_distance = new_distance

            if done:
                next_state_key = None
            else:
                next_state = self._game_to_state(game)
                next_state_key = next_state.to_state_key()

            # Q-learning update
            self._update_q_value(state_key, action, reward, next_state_key)

            if done:
                break

            state = next_state
            steps += 1

        self.epsilon = max(self.config.epsilon_end,
                           self.epsilon * self.config.epsilon_decay)

        score = len(game.snake) - 1
        if score > self.best_score:
            self.best_score = score

        recent_scores = [m.score for m in self.metrics_history[-99:]] + [score]
        avg_score = sum(recent_scores) / len(recent_scores)

        return TrainingMetrics(
            episode=len(self.metrics_history),
            score=score,
            steps=steps,
            epsilon=self.epsilon,
            avg_score_last_100=avg_score,
            max_score=self.best_score,
            q_table_size=self.q_table.size(),
        )

    def _select_action(self, state_key: Tuple, game: SnakeGame) -> int:
        valid_actions = self._get_valid_actions(game)

        if not valid_actions:
            return random.choice(self.actions)

        if random.random() < self.epsilon:
            return random.choice(valid_actions)

        return self.q_table.get_best_action(state_key, valid_actions)

    def _get_valid_actions(self, game: SnakeGame) -> List[int]:
        valid_actions = []
        head_x, head_y = game.snake[-1]

        for i, (dx, dy) in enumerate([(0, 1), (0, -1), (-1, 0), (1, 0)]):
            next_x = (head_x + dx) % game.w
            next_y = (head_y + dy) % game.h
            if (next_x, next_y) not in game.snake[:-1]:
                valid_actions.append(i)

        return valid_actions

    def _update_q_value(
        self,
        state_key: Tuple,
        action: int,
        reward: float,
        next_state_key: Optional[Tuple],
    ) -> None:
        old_q = self.q_table.get_q_value(state_key, action)

        if next_state_key is None:
            next_max_q = 0
        else:
            next_max_q = self.q_table.get_max_q_value(next_state_key)

        new_q = (1 - self.config.learning_rate) * old_q + \
                self.config.learning_rate * (reward + self.config.gamma * next_max_q)

        self.q_table.set_q_value(state_key, action, new_q)

    def _manhattan_distance(
        self,
        pos1: Tuple[int, int],
        pos2: Tuple[int, int],
    ) -> int:
        return abs(pos1[0] - pos2[0]) + abs(pos1[1] - pos2[1])

    def _should_stop_training(self) -> bool:
        if len(self.metrics_history) < 100:
            return False

        recent_avg = sum(m.score for m in self.metrics_history[-100:]) / 100
        if recent_avg > self.config.early_stop_threshold:
            return True

        return False

    def save_q_table(self, filepath: str) -> None:
        with open(filepath, 'wb') as f:
            pickle.dump(
                {
                    'q_table': dict(self.q_table.table),
                    'board_width': self.board_width,
                    'board_height': self.board_height,
                    'best_score': self.best_score,
                }, f)

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


class QTableLoader:

    @staticmethod
    def get_table_filename(board_width: int, board_height: int) -> str:
        return f'storage/q/snake_table_{board_width}x{board_height}.pkl'

    @staticmethod
    def load_q_table(filepath: str) -> Dict:
        with open(filepath, 'rb') as f:
            return pickle.load(f)

    @staticmethod
    def table_exists(board_width: int, board_height: int) -> bool:
        filename = QTableLoader.get_table_filename(board_width, board_height)
        return os.path.exists(filename)


class QLearningAI:

    def __init__(self):
        self.board_width = 20
        self.board_height = 20
        self.q_table = QTable()
        self._load_trained_table()
        self.actions = ['up', 'down', 'left', 'right']

    def _load_trained_table(self) -> None:
        if QTableLoader.table_exists(self.board_width, self.board_height):
            filename = QTableLoader.get_table_filename(
                self.board_width,
                self.board_height,
            )
            checkpoint = QTableLoader.load_q_table(filename)
            self.q_table.table = defaultdict(lambda: defaultdict(float),
                                             checkpoint['q_table'])

    def next_move(self, game: SnakeGame) -> str:
        state = self._game_to_state(game)
        state_key = state.to_state_key()

        # Get valid actions
        valid_actions = []
        head_x, head_y = game.snake[-1]

        for i, (dx, dy) in enumerate([(0, 1), (0, -1), (-1, 0), (1, 0)]):
            next_x = (head_x + dx) % game.w
            next_y = (head_y + dy) % game.h
            if (next_x, next_y) not in game.snake[:-1]:
                valid_actions.append(i)

        if not valid_actions:
            return random.choice(self.actions)

        action_idx = self.q_table.get_best_action(state_key, valid_actions)
        return self.actions[action_idx]

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


class QRunnerAgent(QLearningAI):

    def __init__(self, board_width: int, board_height: int):
        self.board_width = board_width
        self.board_height = board_height
        self.q_table = QTable()
        self._load_trained_table()
        self.actions = ['up', 'down', 'left', 'right']


if __name__ == '__main__':
    config = TrainingConfig()
    trainer = QTableTrainer(config, 20, 20)
    trainer.train()
