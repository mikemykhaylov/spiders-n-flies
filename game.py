import pygame
import random
from enum import Enum
from typing import List, Tuple

class moves(Enum):
    UP = 'UP'
    DOWN = 'DOWN'
    LEFT = 'LEFT'
    RIGHT = 'RIGHT'
    NONE = 'NONE'

class GridEnvironment:
    grid_size = 10

    def __init__(self, k: int, spider_positions: List[Tuple[int, int]], wait_delay: int = 1000):
        self.wait_delay = wait_delay
        self.reset(k, spider_positions)

    def reset(self, k: int, spider_positions: List[Tuple[int, int]]) -> None:
        """Reset environment with new flies and spider positions"""
        self.grid = [[False for _ in range(self.grid_size)] for _ in range(self.grid_size)]
        self.spiders = [pos for pos in spider_positions]

        # Place flies in random unique positions
        all_cells = [(x, y) for x in range(self.grid_size) for y in range(self.grid_size)]
        # Remove spider positions
        for pos in self.spiders:
            if pos in all_cells:
                all_cells.remove(pos)

        random.shuffle(all_cells)
        self.flies = sorted(all_cells[:k])
        for x, y in all_cells[:k]:
            self.grid[x][y] = True

    def move_spider(self, spider_index: int, direction: moves) -> None:
        """Move specified spider in one of 4 directions"""
        new_x, new_y = self.apply_spider_move(self.spiders[spider_index], direction)
        self.spiders[spider_index] = (new_x, new_y)

        # Remove fly if spider lands on it
        if self.grid[new_x][new_y]:
            self.grid[new_x][new_y] = False
            self.flies.remove((new_x, new_y))

    @staticmethod
    def apply_spider_move(spider_pos: Tuple[int, int], move: moves) -> Tuple[int, int]:
        """Return new spider position after applying move"""
        x, y = spider_pos
        dx, dy = 0, 0

        match move:
            case moves.NONE:
                return spider_pos
            case moves.UP:
                dy = -1
            case moves.DOWN:
                dy = 1
            case moves.LEFT:
                dx = -1
            case moves.RIGHT:
                dx = 1
            case _:
                raise ValueError("Invalid direction. Use: UP, DOWN, LEFT, RIGHT")

        new_x = max(0, min(GridEnvironment.grid_size - 1, x + dx))
        new_y = max(0, min(GridEnvironment.grid_size - 1, y + dy))
        return (new_x, new_y)

    def wait(self):
        pygame.time.wait(self.wait_delay)

class GridVisualization:
    def __init__(self, env: GridEnvironment, cell_size: int = 80):
        self.env = env
        self.cell_size = cell_size
        self.window_size = self.cell_size * self.env.grid_size

        pygame.init()
        self.screen = pygame.display.set_mode((self.window_size, self.window_size))
        pygame.display.set_caption("Spiders and Flies Environment")

        # Load images
        self.spider_img = pygame.image.load('src/spider.png')
        self.fly_img = pygame.image.load('src/fruit-fly.png')

        # Scale images to fit cell size
        self.spider_img = pygame.transform.smoothscale(self.spider_img, (self.cell_size//2, self.cell_size//2))
        self.fly_img = pygame.transform.smoothscale(self.fly_img, (self.cell_size//2, self.cell_size//2))

        # Colors
        self.background = (255, 255, 255)
        self.grid_lines = (0, 0, 0)

    def draw_grid(self) -> None:
        # Draw grid lines
        for i in range(self.env.grid_size + 1):
            pygame.draw.line(self.screen, self.grid_lines,
                             (0, i * self.cell_size),
                             (self.window_size, i * self.cell_size))
            pygame.draw.line(self.screen, self.grid_lines,
                             (i * self.cell_size, 0),
                             (i * self.cell_size, self.window_size))

    def draw_flies(self) -> None:
        for x in range(self.env.grid_size):
            for y in range(self.env.grid_size):
                if self.env.grid[x][y]:
                    pos = (x * self.cell_size + self.cell_size//4,
                          y * self.cell_size + self.cell_size//4)
                    self.screen.blit(self.fly_img, pos)

    def draw_spiders(self) -> None:
        # Find overlapping spiders
        overlapping_positions = {}
        for i, spider in enumerate(self.env.spiders):
            x, y = spider
            if (x,y) in overlapping_positions:
                overlapping_positions[(x,y)].append(i)
            else:
                overlapping_positions[(x,y)] = [i]

        # Draw spiders with offsets if needed
        for i, spider in enumerate(self.env.spiders):
            x, y = spider
            if (x,y) in overlapping_positions and len(overlapping_positions[(x,y)]) > 1:
                offset = self.cell_size//8
                if overlapping_positions[(x,y)][0] == i:
                    # First spider offset one way
                    pos = (x * self.cell_size + self.cell_size//4 - offset,
                          y * self.cell_size + self.cell_size//4 - offset)
                else:
                    # Second spider offset opposite way
                    pos = (x * self.cell_size + self.cell_size//4 + offset,
                          y * self.cell_size + self.cell_size//4 + offset)
            else:
                pos = (x * self.cell_size + self.cell_size//4,
                      y * self.cell_size + self.cell_size//4)
            self.screen.blit(self.spider_img, pos)

    def update_display(self) -> None:
        self.screen.fill(self.background)
        self.draw_grid()
        self.draw_flies()
        self.draw_spiders()
        pygame.display.flip()
