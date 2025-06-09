import pygame
import sys
import argparse
from game_state import SnakeGame
from ai_agent import AI
from nn_agent import NeuralNetAI
from q_agent import QLearningAI
from astar_agent import AStarAI

CELL_SIZE = 20
FPS = 120

class SnakeDisplay:
    def __init__(self, w, h, agent_type='random'):
        pygame.init()
        self.screen = pygame.display.set_mode((w*CELL_SIZE, h*CELL_SIZE))
        pygame.display.set_caption('A Smart Snake')
        self.clock = pygame.time.Clock()
        self.game = SnakeGame(w, h)
        
        # Initialize the appropriate agent
        if agent_type == 'nn':
            self.agent = NeuralNetAI()
        elif agent_type == 'q':
            self.agent = QLearningAI()
        elif agent_type == 'astar':
            self.agent = AStarAI()
        else:
            self.agent = AI()

    def draw(self):
        self.screen.fill((0,0,0))
        if self.game.food:
            fx, fy = self.game.food
            pygame.draw.rect(self.screen, (255,0,0), 
                           (fx*CELL_SIZE, fy*CELL_SIZE, CELL_SIZE, CELL_SIZE))
        for x,y in self.game.snake:
            pygame.draw.rect(self.screen, (0,255,0), 
                           (x*CELL_SIZE, y*CELL_SIZE, CELL_SIZE, CELL_SIZE))
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
                        pygame.time.wait(1000)  # Simple pause

            self.clock.tick(FPS)
            move = self.agent.next_move(self.game)
            status = getattr(self.game, f"move_{move}")()
            if status == 'game_over':
                running = False
            self.draw()
        pygame.quit()

def main():
    parser = argparse.ArgumentParser(description='Run Snake Game with AI agent')
    parser.add_argument('--agent', type=str, default='random',
                      choices=['random', 'nn', 'q', 'astar'],
                      help='Type of AI agent to use')
    args = parser.parse_args()
    
    SnakeDisplay(20, 20, args.agent).run()

if __name__ == '__main__':
    main()
