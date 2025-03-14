import pygame
import numpy as np
import heapq
import copy

class Navigate:
    def __init__(self, size=64, buffer_size=5, cell_size=5):
        self.size = size
        self.buffer_size = buffer_size  # Grid-based buffer calculations (treated as squares)
        self.cell_size = cell_size
        self.start = None
        self.goal = None
        self.path = None
        self.boulders = set()
        self.buffer_positions = set()
        self.draw_buffer_positions = None

    def convert_to_pygame_coords(self, coords):
        return [(32 - y, x + 32) for x, y in coords]

    def selection(self, boulder_coords):
        """Displays the terrain and allows user to select start and goal points."""
        pygame.init()
        screen_size = (self.size * self.cell_size, self.size * self.cell_size)
        screen = pygame.display.set_mode(screen_size)
        pygame.display.set_caption("Select Start and Goal Points")

        selected_points = []
        self.boulders = set(self.convert_to_pygame_coords(boulder_coords))

        # Store buffer zones as grid squares (but visualize as circles)
        self.buffer_positions = set()
        for bx, by in self.boulders:
            for dx in range(-self.buffer_size, self.buffer_size + 1):
                for dy in range(-self.buffer_size, self.buffer_size + 1):
                    if 0 <= bx + dx < self.size and 0 <= by + dy < self.size:
                        self.buffer_positions.add((bx + dx, by + dy))
        self.buffer_positions -= self.boulders  # Ensure buffers don't include boulders

        def draw_grid():
            # Store buffer zones as grid squares (but visualize as circles)
            if self.draw_buffer_positions is None:
                self.draw_buffer_positions = set()
                for bx, by in self.boulders:
                    for dx in range(-self.buffer_size, self.buffer_size + 1):
                        for dy in range(-self.buffer_size, self.buffer_size + 1):
                            if 0 <= bx + dx < self.size and 0 <= by + dy < self.size:
                                self.buffer_positions.add((bx + dx, by + dy))
                self.draw_buffer_positions -= self.boulders  # Ensure buffers don't include boulders
                
            screen.fill((255, 255, 255))
            for x in range(self.size):
                for y in range(self.size):
                    if (x, y) in self.boulders:
                        color = (163, 59, 36)  # Dark Red (boulders)
                    elif (x, y) in self.draw_buffer_positions:
                        color = (255, 140, 0)  # Orange (buffer zones)
                    else:
                        color = (19, 156, 94)  # Green (walkable terrain)
                    pygame.draw.rect(screen, color, (y * self.cell_size, x * self.cell_size, self.cell_size, self.cell_size))

            # Draw buffer circles (display only)
            for bx, by in self.boulders:
                pygame.draw.circle(screen, (255, 140, 0), 
                                   (by * self.cell_size + self.cell_size // 2, bx * self.cell_size + self.cell_size // 2), 
                                   self.buffer_size * self.cell_size)

            # Draw selected start/goal points
            for idx, (px, py) in enumerate(selected_points):
                pygame.draw.circle(screen, (0, 0, 255) if idx == 0 else (0, 255, 0),
                                   (py * self.cell_size + self.cell_size // 2, px * self.cell_size + self.cell_size // 2), 4)

        running = True
        while running:
            draw_grid()
            pygame.display.flip()
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.MOUSEBUTTONDOWN:
                    mx, my = pygame.mouse.get_pos()
                    grid_x, grid_y = my // self.cell_size, mx // self.cell_size
                    if (grid_x, grid_y) not in self.boulders and (grid_x, grid_y) not in self.draw_buffer_positions:
                        selected_points.append((grid_x, grid_y))
                    if len(selected_points) == 2:
                        self.start, self.goal = selected_points
                        running = False
        pygame.quit()

    def astar(self, start, goal):
        """Finds a path using A* algorithm avoiding boulders and square buffer zones."""
        if start in self.boulders or start in self.buffer_positions or goal in self.boulders or goal in self.buffer_positions:
            return None  # Path not possible if start/goal are in obstacles

        directions = [(0, 1), (1, 0), (0, -1), (-1, 0), (-1, -1), (-1, 1), (1, -1), (1, 1)]
        open_set = []
        heapq.heappush(open_set, (0, start))
        came_from = {}
        g_score = {start: 0}
        f_score = {start: abs(start[0] - goal[0]) + abs(start[1] - goal[1])}  # Manhattan Distance

        while open_set:
            _, current = heapq.heappop(open_set)

            if current == goal:
                path = []
                while current in came_from:
                    path.append(current)
                    current = came_from[current]
                path.append(start)
                return path[::-1]  # Return reversed path

            for dx, dy in directions:
                neighbor = (current[0] + dx, current[1] + dy)

                # Ensure within bounds and not an obstacle
                if 0 <= neighbor[0] < self.size and 0 <= neighbor[1] < self.size:
                    if neighbor in self.boulders or neighbor in self.buffer_positions:
                        continue  # Skip obstacles

                    # Manhattan distance for cost
                    tentative_g_score = g_score[current] + 1  # Grid-based, so each move costs 1

                    if neighbor not in g_score or tentative_g_score < g_score[neighbor]:
                        came_from[neighbor] = current
                        g_score[neighbor] = tentative_g_score
                        f_score[neighbor] = tentative_g_score + abs(neighbor[0] - goal[0]) + abs(neighbor[1] - goal[1])

                        heapq.heappush(open_set, (f_score[neighbor], neighbor))

        return None  # No path found
    
    def simplify_path(self, path, smoothing_factor=0.3, zigzag_threshold=0.7):
        """Smooths out the path, reduces unnecessary waypoints, and eliminates clustered zigzagging."""
        if not path or len(path) < 3:
            return path  # If the path is too short, return it as-is.

        # Step 1: Remove redundant waypoints (keep only major directional changes)
        reduced_path = [path[0]]
        for i in range(1, len(path) - 1):
            prev, curr, next = path[i - 1], path[i], path[i + 1]

            # Detect zigzag pattern: rapid alternation in movement direction
            prev_dx, prev_dy = prev[0] - curr[0], prev[1] - curr[1]
            next_dx, next_dy = curr[0] - next[0], curr[1] - next[1]

            is_zigzag = (abs(prev_dx + next_dx) <= zigzag_threshold) and (abs(prev_dy + next_dy) <= zigzag_threshold)

            # If movement direction changes significantly and isn't a zigzag, keep the point
            if not ((prev_dx == next_dx and prev_dy == next_dy) or is_zigzag):
                reduced_path.append(curr)

        reduced_path.append(path[-1])

        # Step 2: Apply path smoothing using interpolation
        smoothed_path = [reduced_path[0]]
        for i in range(1, len(reduced_path) - 1):
            prev, curr, next = reduced_path[i - 1], reduced_path[i], reduced_path[i + 1]

            # Compute a new position slightly adjusted towards neighbors
            new_x = curr[0] * (1 - smoothing_factor) + (prev[0] + next[0]) / 2 * smoothing_factor
            new_y = curr[1] * (1 - smoothing_factor) + (prev[1] + next[1]) / 2 * smoothing_factor

            smoothed_path.append((int(round(new_x)), int(round(new_y))))

        smoothed_path.append(reduced_path[-1])
        smoothed_path.append(self.goal)
        return smoothed_path

    
    def plotPath(self):
        """Finds and plots path using A* and saves it as PNG."""
        if self.start is None or self.goal is None:
            print("Start or Goal is not set.")
            return

        path = self.astar(self.start, self.goal)
        if not path:
            print("No valid path found.")
            return

        self.path = self.simplify_path(path)
        pygame.init()
        screen = pygame.display.set_mode((self.size * self.cell_size, self.size * self.cell_size))
        pygame.display.set_caption("Path Plot")

        def draw_grid():
            for x in range(self.size):
                for y in range(self.size):
                    color = (19, 156, 94)  # Default Green
                    if (x, y) in self.boulders:
                        color = (163, 59, 36)  # Boulders
                    elif (x, y) in self.draw_buffer_positions:
                        color = (255, 140, 0)  # Buffer
                    pygame.draw.rect(screen, color, (y * self.cell_size, x * self.cell_size, self.cell_size, self.cell_size))

            # Draw buffer circles (visual only)
            for bx, by in self.boulders:
                pygame.draw.circle(screen, (255, 140, 0), 
                                   (by * self.cell_size + self.cell_size // 2, bx * self.cell_size + self.cell_size // 2), 
                                   self.buffer_size * self.cell_size)

            # Draw the path as a line
            if len(self.path) > 1:
                path_points = [(py * self.cell_size + self.cell_size // 2, 
                                px * self.cell_size + self.cell_size // 2) for px, py in self.path]
                pygame.draw.lines(screen, (0, 0, 0), False, path_points, 2)  # Connect the dots with a line

            # Draw waypoints as dots
            for px, py in self.path:
                pygame.draw.circle(screen, (0, 0, 0), (py * self.cell_size + self.cell_size // 2, px * self.cell_size + self.cell_size // 2), 3)

        draw_grid()
        pygame.display.flip()
        pygame.image.save(screen, "V3_Env_Model/AstarPath.png")
        pygame.quit()
