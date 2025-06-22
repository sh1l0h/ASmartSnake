import random

class SnakeGame:

    def __init__(self, w, h):
        self.w = w
        self.h = h
        init = (w // 2, h // 2)
        self.snake = [init]
        self.food = self._place_food()
        self.game_over = False
        self.score = 0

    def _place_food(self):
        total = self.w * self.h - len(self.snake)
        if total <= 0:
            return None
        r = random.randrange(total)
        snake_set = set(self.snake)
        count = -1
        for x in range(self.w):
            for y in range(self.h):
                if (x, y) in snake_set:
                    continue
                count += 1
                if count == r:
                    return (x, y)

    def _move(self, direction):
        """
        Internal move handler on a toroidal grid.
        Returns:
          - 'food' if food eaten,
          - 'move' if moved normally,
          - 'game_over' if collision with self.
        """
        if self.game_over:
            return 'game_over'

        dx, dy = direction
        head_x, head_y = self.snake[-1]
        new_head = ((head_x + dx) % self.w, (head_y + dy) % self.h)

        if new_head in self.snake:
            self.game_over = True
            return 'game_over'

        self.snake.append(new_head)
        if new_head == self.food:
            self.food = self._place_food()
            self.score += 1
            return 'food'
        else:
            self.snake.pop(0)
            return 'move'

    def move_up(self):
        return self._move((0, 1))

    def move_down(self):
        return self._move((0, -1))

    def move_left(self):
        return self._move((-1, 0))

    def move_right(self):
        return self._move((1, 0))
