import random

class AI:

    def next_move(self):
        moves = ['up', 'down', 'left', 'right']
        return random.choice(moves)
