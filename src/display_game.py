import pygame
import argparse
from game_state import SnakeGame
from nn_agent import NNRunnerAgent
from q_agent import QRunnerAgent
from astar_agent import AStarAI

CELL_SIZE = 20
FPS = 120


class SnakeDisplay:

    def __init__(self, w, h, agent_type='astar'):
        pygame.init()
        self.font = pygame.font.SysFont(None, 36)
        self.screen = pygame.display.set_mode((w * CELL_SIZE, h * CELL_SIZE))
        pygame.display.set_caption('A Smart Snake')
        self.clock = pygame.time.Clock()
        self.game = SnakeGame(w, h)

        # Initialize the appropriate agent
        match agent_type:
            case 'nn':
                self.agent = NNRunnerAgent(board_width=w, board_height=h)
            case 'q':
                self.agent = QRunnerAgent(board_width=w, board_height=h)
            case 'astar':
                self.agent = AStarAI()

    def draw(self):
        self.screen.fill((0, 0, 0))

        # Draw food
        if self.game.food:
            fx, fy = self.game.food
            pygame.draw.rect(
                self.screen,
                (255, 0, 0),
                (
                    fx * CELL_SIZE,
                    fy * CELL_SIZE,
                    CELL_SIZE,
                    CELL_SIZE,
                ),
            )

        # Draw snake
        for x, y in self.game.snake:
            pygame.draw.rect(
                self.screen,
                (0, 255, 0),
                (
                    x * CELL_SIZE,
                    y * CELL_SIZE,
                    CELL_SIZE,
                    CELL_SIZE,
                ),
            )

        # Draw score
        score_text = self.font.render(
            f"Score: {self.game.score}",
            True,
            (255, 255, 255),
        )

        self.screen.blit(score_text, (10, 10))

        pygame.display.flip()

    def run(self):
        running = True
        while running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                    break
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_q:
                        running = False
                    elif event.key == pygame.K_r:
                        self.game = SnakeGame(self.game.w, self.game.h)
                    elif event.key == pygame.K_p:
                        pygame.time.wait(1000)

            self.clock.tick(FPS)
            move = self.agent.next_move(self.game)
            status = getattr(self.game, f"move_{move}")()
            if status == 'game_over':
                running = False
            self.draw()
        pygame.quit()


def main():
    parser = argparse.ArgumentParser(
        description='Run Snake Game with AI agent')

    parser.add_argument(
        '--agent',
        type=str,
        default='astar',
        choices=['nn', 'q', 'astar'],
        help='Type of AI agent to use',
    )

    args = parser.parse_args()

    SnakeDisplay(20, 20, args.agent).run()


if __name__ == '__main__':
    main()
