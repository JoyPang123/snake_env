import random

import gym
import numpy as np
import pygame
from gym import spaces

# Set the screen size
SCREEN_WIDTH, SCREEN_HEIGHT = 600, 400

# Make a grid with 10x10 pixels
GRID_SIZE = 10

# Get a grid size
GRID_WIDTH = int(SCREEN_WIDTH / GRID_SIZE)
GRID_HEIGHT = int(SCREEN_HEIGHT / GRID_SIZE)

# Set up the direction
UP = (0, -1)
DOWN = (0, 1)
LEFT = (-1, 0)
RIGHT = (1, 0)

# Action dict
action_dict = {
    0: UP,
    1: DOWN,
    2: LEFT,
    3: RIGHT
}

# Set time interval
FPS = 10


def draw_box(surf, color, pos):
    """Draw rectangle on surface
    Args:
    surf: Surface to draw rectangle
    color: The color of the rectangle
    pos: The top-left place of rectangle
    Return: None
    """

    # Create a rectangle object
    r = pygame.Rect((pos[0], pos[1]), (GRID_SIZE, GRID_SIZE))
    # Draw on the surface
    pygame.draw.rect(surf, color, r)


class Apple(object):
    def __init__(self):
        self.position = (0, 0)
        self.color = (255, 0, 0)
        self.randomize()

    def randomize(self):
        """Randomly set the position of the apple
        Args:
            self: Instance itself
        Return: None
        """
        self.position = (random.randint(0, GRID_WIDTH - 1) * GRID_SIZE, random.randint(0, GRID_HEIGHT - 1) * GRID_SIZE)

    def draw(self, surf):
        """Draw the apple on the surface
        Args:
            self: Instance itself
            surf: Surface to draw on
        Return: None
        """
        draw_box(surf, self.color, self.position)


# noinspection PyAttributeOutsideInit
class SnakeAgent(object):
    def __init__(self):
        self.restart()
        self.color = (0, 0, 0)

    def get_head_position(self):
        """Return the head position (index 0) of snake
        Args:
            self: Instance itself
        Return: None
        """
        return self.positions[0]

    def restart(self):
        """Restart the game
        Args:
            self: Instance itself
        Return: None
        """
        # Reset the information of snake
        self.length = 1
        self.positions = [((SCREEN_WIDTH / 2), (SCREEN_HEIGHT / 2))]
        self.direction = random.choice([UP, DOWN, LEFT, RIGHT])

    def point(self, pt):
        """Set the moving direction of snake
        Args:
            self: Instance itself
            pt: Direction of snake, include "UP", "DOWN", "LEFT" and "RIGHT"

        Return: None
        """
        # The snake couldn't change its moving direction oppositely
        #  if the length is greater than 1
        if (self.length <= 1) or \
                ((pt[0] * -1, pt[1] * -1) != self.direction):
            self.direction = pt

    def move(self):
        """Update the snake position according to its moving
        direction
        Args:
            self: Instance itself

        Return: None
        """
        # Get current position
        cur = self.positions[0]
        # Direction to move
        point_x, point_y = self.direction
        # New position
        new = ((cur[0] + point_x * GRID_SIZE), (cur[1] + point_y * GRID_SIZE))

        done = False
        # Check whether the snake will crash into itself
        if len(self.positions) > 2 and new in self.positions[2:]:
            done = True
        elif (new[0] < 0) | (new[0] >= SCREEN_WIDTH) | (new[1] < 0) | (new[1] >= SCREEN_HEIGHT):
            done = True
        else:
            self.positions.insert(0, new)
            if len(self.positions) > self.length:
                self.positions.pop()

        return done

    def draw(self, surf):
        """Draw the snake on the surface

        Args:
            self: Instance itself
            surf:,
        """
        for p in self.positions:
            draw_box(surf, self.color, p)


def check_eat(snake, apple):
    """Check whether snake eat the apple
    Args:
        snake: Snake object, used to obtain the head position of snake
        apple: Apple object, used to obtain the position of apple
    Return: None
    """

    if snake.get_head_position() == apple.position:
        # After eating an apple, increase length
        #  and regenerate the apple
        snake.length += 1
        apple.randomize()


class SnakeEnv(gym.Env):
    def __init__(self, seed=None, max_iter=1000):
        self.seed = seed

        # Initialize the render screen to None
        self.screen = None

        # The maximum iteration for a env is `max_iter` steps (default to 1000)
        self.max_iter = max_iter
        self.iter_count = 0

        # Initialize the agent and the target
        self.snake = SnakeAgent()
        self.apple = Apple()

        # Only up, down, left, right
        self.action_space = spaces.Discrete(4)
        # Set up two different kind of observation - frame and action
        self.observation_space = spaces.Dict({
            "1D_state": spaces.Discrete(GRID_WIDTH * GRID_HEIGHT),
            "frame": spaces.Box(
                0, 255, (GRID_WIDTH, GRID_HEIGHT, 3),
                dtype=np.uint8
            )
        })

    def step(self, choose_act):
        self.snake.point(action_dict[choose_act])
        self.iter_count += 1
        done = self.snake.move()
        new_obs = self._get_frame()
        reward = 0

        if self.iter_count == self.max_iter:
            done = True

        return new_obs, reward, done, None

    def reset(self):
        # Reset the agent and the target
        self.snake.restart()
        self.apple.randomize()
        self.iter_count = 0

        # Get the current state
        curr_frame = self._get_frame()
        curr_1D_state = self._get_1D_state()

        return {
            "1D_state": curr_1D_state,
            "frame": curr_frame,
        }

    def render(self, mode="human"):
        if self.screen is None:
            pygame.init()
            self.screen = pygame.display.set_mode(
                (SCREEN_WIDTH, SCREEN_HEIGHT)
            )
        clock = pygame.time.Clock()

        # Fill the background with white color
        surface = pygame.Surface(self.screen.get_size())
        surface.fill((255, 255, 255))

        # Draw snake and apple on screen
        self.snake.draw(surface)
        self.apple.draw(surface)

        # Show the score
        font = pygame.font.Font(None, 36)
        text = font.render(str(self.snake.length - 1), True, (10, 10, 10))
        text_pos = text.get_rect()
        text_pos.centerx = 20

        # Add object to the screen
        surface.blit(text, text_pos)
        self.screen.blit(surface, (0, 0))

        # Update the screen
        pygame.display.flip()
        pygame.display.update()

        clock.tick(FPS + int(self.snake.length / 3))

    def close(self):
        pygame.quit()

    def _get_frame(self):
        # Create white frame
        empty_frame = np.full((GRID_WIDTH, GRID_HEIGHT, 3), 255)

        # Put in snake body with color black
        for x, y in self.snake.positions:
            x, y = int(x), int(y)
            empty_frame[x:x + GRID_SIZE, y:y + GRID_SIZE] = self.snake.color

        # Put in apple with color red
        apple_x, apple_y = self.apple.position
        empty_frame[apple_x:apple_x + GRID_SIZE, apple_y:apple_y + GRID_SIZE] = self.apple.color

        return empty_frame

    def _get_1D_state(self):
        return self.snake.positions[0][0] + self.snake.positions[0][1]

    def _get_distance(self):
        snake_head_x, snake_head_y = self.snake.positions[0]
        apple_x, apple_y = self.apple.position

        return abs(snake_head_x - apple_x) + abs(snake_head_y - apple_y)
