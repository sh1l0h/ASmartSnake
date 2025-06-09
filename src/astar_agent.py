import heapq


class AStarAI:
    def __init__(self):
        self.actions = ['up', 'down', 'left', 'right']
        self.directions = {
            'up': (0, 1),
            'down': (0, -1),
            'left': (-1, 0),
            'right': (1, 0)
        }

    def heuristic(self, a, b):
        # Manhattan distance
        return abs(a[0] - b[0]) + abs(a[1] - b[1])

    def get_neighbors(self, pos, w, h):
        neighbors = []
        for move, (dx, dy) in self.directions.items():
            nx, ny = (pos[0] + dx) % w, (pos[1] + dy) % h
            neighbors.append(((nx, ny), move))
        return neighbors

    def reconstruct_path(self, came_from, current):
        path = [current]
        while current in came_from:
            current = came_from[current]
            path.append(current)
        path.reverse()
        return path

    def next_move(self, game):
        w, h = game.w, game.h
        start = game.snake[-1]
        goal = game.food
        snake_body = set(game.snake)
        open_set = []
        heapq.heappush(
            open_set, (0 + self.heuristic(start, goal), 0, start, None)
        )
        came_from = {}
        g_score = {start: 0}
        move_from = {}
        while open_set:
            _, cost, current, prev = heapq.heappop(open_set)
            if current == goal:
                # Reconstruct path
                path = []
                while prev is not None:
                    path.append((current, move_from[current]))
                    current, prev = prev
                path.reverse()
                if path:
                    return path[0][1]
                else:
                    # Default if already at food
                    return 'up'
            for neighbor, move in self.get_neighbors(current, w, h):
                if neighbor in snake_body and neighbor != goal:
                    continue
                tentative_g = cost + 1
                if neighbor not in g_score or tentative_g < g_score[neighbor]:
                    g_score[neighbor] = tentative_g
                    priority = tentative_g + self.heuristic(neighbor, goal)
                    heapq.heappush(
                        open_set, (priority, tentative_g, neighbor, (current, prev))
                    )
                    came_from[neighbor] = current
                    move_from[neighbor] = move
        # If no path found, just move up
        return 'up'
