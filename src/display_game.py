import pygame
from game_state import SnakeGame
from ai_agent import AI

CELL_SIZE = 20
FPS = 5

class SnakeDisplay:
    def __init__(self, w, h):
        pygame.init()
        self.screen = pygame.display.set_mode((w*CELL_SIZE, h*CELL_SIZE))
        pygame.display.set_caption('A Smart Snake')
        self.clock = pygame.time.Clock()
        self.game = SnakeGame(w, h)
        self.agent = AI()

    def draw(self):
        self.screen.fill((0,0,0))
        if self.game.food:
            fx, fy = self.game.food
            pygame.draw.rect(self.screen, (255,0,0), (fx*CELL_SIZE, fy*CELL_SIZE, CELL_SIZE, CELL_SIZE))
        for x,y in self.game.snake:
            pygame.draw.rect(self.screen, (0,255,0), (x*CELL_SIZE, y*CELL_SIZE, CELL_SIZE, CELL_SIZE))
        pygame.display.flip()

    def run(self):
        running = True
        while running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                    break

            self.clock.tick(FPS)
            move = self.agent.next_move()
            status = getattr(self.game, f"move_{move}")()
            if status == 'game_over':
                running = False
            self.draw()
        pygame.quit()

if __name__ == '__main__':
    SnakeDisplay(20,20).run()
