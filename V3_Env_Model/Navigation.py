import numpy as np
import pygame
import heapq
import random

class Navigate:
    def __init__(self, size=128):
        self.size = size
        self.grid = None
        self.start = None
        self.goal = None

    def generate_grid(self, num_boulder_regions=5, boulder_size=15, max_cost=10, buffer_size=10):
        """Generate terrain with cost regions and apply a buffer around high-cost zones."""
        grid = np.random.uniform(1, 3, (self.size, self.size))  # Base terrain with slight variation
        
        # Create high-cost boulder regions
        high_cost_positions = set()
        for _ in range(num_boulder_regions):
            while True:
                x, y = np.random.randint(0, self.size - boulder_size, size=2)

                # Ensure no overlap with previously placed obstacles
                if any((i, j) in high_cost_positions for i in range(x, x+boulder_size) for j in range(y, y+boulder_size)):
                    continue
                
                break
            
            # Assign high cost to boulder region
            grid[x:x+boulder_size, y:y+boulder_size] = np.random.uniform(6, max_cost, (boulder_size, boulder_size))
            high_cost_positions.update((i, j) for i in range(x, x+boulder_size) for j in range(y, y+boulder_size))

        # Expand buffer around high-cost zones
        buffer_positions = set()
        for i, j in list(high_cost_positions):
            for dx in range(-buffer_size, buffer_size+1):
                for dy in range(-buffer_size, buffer_size+1):
                    if 0 <= i+dx < self.size and 0 <= j+dy < self.size:
                        if grid[i+dx, j+dy] < 6:  # Only modify if it's NOT already a boulder
                            grid[i+dx, j+dy] = max(grid[i+dx, j+dy], 4)  # Medium-cost (buffer zone)
                            buffer_positions.add((i+dx, j+dy))

        # Set Start at Top-Left and Goal at Bottom-Right, ensuring they are outside obstacles
        self.start = self.find_valid_position(grid, high_cost_positions, buffer_positions, (20, 20))
        self.goal = self.find_valid_position(grid, high_cost_positions, buffer_positions, (self.size - 20, self.size - 20))

        self.grid = grid

    def find_valid_position(self, grid, obstacles, buffers, preferred_position):
        """Find a valid position near the preferred position, avoiding obstacles and buffers."""
        x, y = preferred_position
        search_radius = 5  # Expand search if the preferred spot is blocked
        
        for dx in range(-search_radius, search_radius+1):
            for dy in range(-search_radius, search_radius+1):
                nx, ny = x + dx, y + dy
                if 0 <= nx < self.size and 0 <= ny < self.size:
                    if (nx, ny) not in obstacles and (nx, ny) not in buffers and grid[nx, ny] < 4:
                        return (nx, ny)

        # Fallback: Find any random open space
        return self.find_valid_position(grid, obstacles, buffers, (random.randint(0, self.size - 1), random.randint(0, self.size - 1)))

    def astar(self, start, goal):
        """A* algorithm with buffer-aware navigation."""
        size = self.size

        def heuristic(a, b):
            return abs(a[0] - b[0]) + abs(a[1] - b[1])  # Manhattan distance

        open_set = []
        heapq.heappush(open_set, (0, start))
        came_from = {}
        g_score = {start: 0}
        f_score = {start: heuristic(start, goal)}

        while open_set:
            _, current = heapq.heappop(open_set)

            if current == goal:
                path = []
                while current in came_from:
                    path.append(current)
                    current = came_from[current]
                path.append(start)
                return path[::-1]

            for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                neighbor = (current[0] + dx, current[1] + dy)
                if 0 <= neighbor[0] < size and 0 <= neighbor[1] < size:
                    # Avoid only high-cost areas (boulders, not buffer)
                    if self.grid[neighbor] >= 6:
                        continue  

                    tentative_g = g_score[current] + self.grid[neighbor]
                    if neighbor not in g_score or tentative_g < g_score[neighbor]:
                        g_score[neighbor] = tentative_g
                        f_score[neighbor] = tentative_g + heuristic(neighbor, goal)
                        heapq.heappush(open_set, (f_score[neighbor], neighbor))
                        came_from[neighbor] = current
        return []

    def draw_grid(self, screen, cell_size, path):
        """Draw the grid with color-coded terrain and A* path."""
        for x in range(self.size):
            for y in range(self.size):
                cost = self.grid[x, y]
                if cost < 2:
                    color = (19, 156, 94)  # Green (low-cost)
                elif cost < 4:
                    color = (109, 178, 67)  # Light Orange (low-medium)
                elif cost < 6:
                    color = (186, 156, 47)  # Orange (buffer zones)
                else:
                    color = (163, 59, 36)  # Red (boulders)

                pygame.draw.rect(screen, color, (y * cell_size, x * cell_size, cell_size, cell_size))

        # Draw path in black
        for x, y in path:
            pygame.draw.rect(screen, (0, 0, 0), (y * cell_size, x * cell_size, cell_size, cell_size))

        # Draw start and goal
        pygame.draw.rect(screen, (0, 255, 0), (self.start[1] * cell_size, self.start[0] * cell_size, cell_size, cell_size))  # Green start
        pygame.draw.rect(screen, (0, 0, 255), (self.goal[1] * cell_size, self.goal[0] * cell_size, cell_size, cell_size))  # Blue goal

    def getGrid(self):
        return self.grid
    
    def interactivePlot(self):
        """Run Pygame visualization."""
        self.generate_grid()
        path = self.astar(self.start, self.goal)

        pygame.init()
        cell_size = 5
        screen = pygame.display.set_mode((self.size * cell_size, self.size * cell_size))
        pygame.display.set_caption("A* Pathfinding with Buffer Zone")
        running = True
        while running:
            self.step()
            
        def step(self, running, robot_pos):
            screen.fill((255, 255, 255))
            self.draw_grid(screen, cell_size, path)
            pygame.display.flip()

            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False

        pygame.quit()

nav = Navigate()
nav.interactivePlot()
