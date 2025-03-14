import pygame
import numpy as np
import heapq
import copy

class Navigate:
    def __init__(self, size=128, buffer_size=3):
        self.size = size
        self.buffer_size = buffer_size
        self.start = None
        self.goal = None
        self.path = None
        self.boulders = set()
        self.buffer_positions = set()
    
    def convert_to_pygame_coords(self, coords):
        coords_copy = copy.deepcopy(coords)
        for i in range(len(coords)):
            coords[i] = (64 - coords_copy[i][1], coords_copy[i][0] + 64 )
        return (coords)
    
    def selection(self, boulder_coords, boulder_buffer=6, size=128, cell_size=5):
        """Displays the 128x128 terrain and allows user to select start and goal points via clicks."""
        pygame.init()
        screen_size = (size * cell_size, size * cell_size)
        screen = pygame.display.set_mode(screen_size)
        pygame.display.set_caption("Select Start and Goal Points")

        selected_points = set()

        # Use the existing conversion function
        boulder_coords = set(self.convert_to_pygame_coords(boulder_coords))

        # Compute buffer zones around boulders
        buffer_positions = set()
        for bx, by in boulder_coords:
            for dx in range(-boulder_buffer, boulder_buffer + 1):
                for dy in range(-boulder_buffer, boulder_buffer + 1):
                    if 0 <= bx + dx < size and 0 <= by + dy < size:
                        buffer_positions.add((bx + dx, by + dy))
        buffer_positions -= boulder_coords  # Ensure buffers don't include boulders

        def draw_grid():
            """Draws the terrain grid."""
            for x in range(size):
                for y in range(size):
                    if (x, y) in boulder_coords:
                        color = (163, 59, 36)  # Dark Red (boulders)
                    elif (x, y) in buffer_positions:
                        color = (255, 140, 0)  # Orange (buffer zones)
                    else:
                        color = (19, 156, 94)  # Green (walkable terrain)

                    pygame.draw.rect(screen, color, (y * cell_size, x * cell_size, cell_size, cell_size))

            # Draw selected points
            for idx, (px, py) in enumerate(selected_points):
                pygame.draw.circle(screen, (0, 0, 255) if idx == 0 else (0, 255, 0),
                                (py * cell_size + cell_size // 2, px * cell_size + cell_size // 2), 4)

        running = True
        while running:
            screen.fill((255, 255, 255))
            draw_grid()
            pygame.display.flip()

            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False

                elif event.type == pygame.MOUSEBUTTONDOWN:
                    mx, my = pygame.mouse.get_pos()
                    grid_x, grid_y = my // cell_size, mx // cell_size  # Convert click position to grid

                    # Ensure the point is valid (not on boulders or buffer zones)
                    if (grid_x, grid_y) not in boulder_coords and (grid_x, grid_y) not in buffer_positions:
                        selected_points.add((grid_x, grid_y))

                    if len(selected_points) == 2:
                        self.start, self.goal = list(selected_points)
                        running = False  # Exit loop after selecting both points

        pygame.quit()
