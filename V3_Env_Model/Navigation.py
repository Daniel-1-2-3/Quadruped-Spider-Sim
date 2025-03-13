import pygame
import numpy as np

class Navigate:
    def __init__(self, terrain_data, size=128, buffer_size=4):
        self.size = size
        self.buffer_size = buffer_size
        self.terrain_data = terrain_data.reshape(size, size)
        self.start = None
        self.goal = None

    def selection(self, cell_size=5):
        """Displays terrain and allows user to select start and goal points via clicks."""
        pygame.init()
        screen_size = (self.size * cell_size, self.size * cell_size)
        screen = pygame.display.set_mode(screen_size)
        pygame.display.set_caption("Select Start and Goal Points")

        selected_points = []

        # Identify boulders (height != 0)
        boulders = set(zip(*np.where(self.terrain_data != 0)))

        # Identify buffer zones around boulders
        buffer_positions = set()
        for bx, by in boulders:
            for dx in range(-self.buffer_size, self.buffer_size + 1):
                for dy in range(-self.buffer_size, self.buffer_size + 1):
                    if (0 <= bx + dx < self.size) and (0 <= by + dy < self.size):
                        buffer_positions.add((bx + dx, by + dy))
        buffer_positions -= boulders  # Remove boulders from buffer list

        def draw_grid():
            """Draws the terrain grid with color-coded regions."""
            for x in range(self.size):
                for y in range(self.size):
                    if (x, y) in boulders:
                        color = (163, 59, 36)  # Red (boulders)
                    elif (x, y) in buffer_positions:
                        color = (186, 156, 47)  # Orange (buffer)
                    else:
                        color = (19, 156, 94)  # Green (normal ground)

                    pygame.draw.rect(screen, color, (y * cell_size, x * cell_size, cell_size, cell_size))

            # Draw selected points
            for idx, (px, py) in enumerate(selected_points):
                pygame.draw.circle(screen, (0, 0, 255) if idx == 0 else (0, 255, 0),
                                   (py * cell_size + cell_size // 2, px * cell_size + cell_size // 2), cell_size // 2)

        running = True
        while running:
            screen.fill((255, 255, 255))
            draw_grid()
            pygame.display.flip()

            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False

                elif event.type == pygame.MOUSEBUTTONDOWN:
                    if len(selected_points) < 2:
                        mx, my = pygame.mouse.get_pos()
                        grid_x, grid_y = my // cell_size, mx // cell_size

                        # Ensure the point is valid (not on boulders)
                        if (grid_x, grid_y) not in boulders:
                            selected_points.append((grid_x, grid_y))

                        if len(selected_points) == 2:
                            self.start, self.goal = selected_points
                            running = False  # Exit loop after selecting both points

        pygame.quit()
