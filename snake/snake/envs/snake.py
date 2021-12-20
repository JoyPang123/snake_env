import random
import time

import gym
import numpy as np
from gym import spaces

SCREEN_WIDTH, SCREEN_HEIGHT = 200, 200

# Make a grid with 10x10 pixels
GRID_SIZE = 20

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


def draw_box_gl(color, pos):
    """Draw rectangle on surface
    Args:
    pos: The top-left place of rectangle
    color: Color to draw
    Return: None
    """
    pyglet.gl.glBegin(pyglet.gl.GL_QUADS)
    pyglet.gl.glColor3d(*color)
    pyglet.gl.glVertex2d(pos[0], pos[1])
    pyglet.gl.glVertex2d(pos[0] + GRID_SIZE, pos[1])
    pyglet.gl.glVertex2d(pos[0] + GRID_SIZE, pos[1] + GRID_SIZE)
    pyglet.gl.glVertex2d(pos[0], pos[1] + GRID_SIZE)
    pyglet.gl.glEnd()


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
        apple.snake_positions = snake.positions
        apple.randomize()

        return True

    return False


def check_crush(snake, walls):
    """Check whether snake eat the apple
    Args:
        snake: Snake object, used to obtain the head position of snake
        walls: Apple object, used to obtain the position of apple
    Return: None
    """

    for wall in walls:
        if snake.get_head_position() == wall.position:
            return True
    return False


def hamming_dis(point1, point2):
    return abs(point1[0] - point2[0]) / GRID_WIDTH + abs(point1[1] - point2[1]) / GRID_HEIGHT


class Apple(object):
    def __init__(self, snake_positions):
        self.position = (0, 0)
        self.color = (255, 0, 0)
        self.snake_positions = snake_positions
        self.randomize()

    def randomize(self):
        """Randomly set the position of the apple
        Args:
            self: Instance itself
        Return: None
        """
        self.position = (random.randint(0, GRID_WIDTH - 1) * GRID_SIZE, random.randint(0, GRID_HEIGHT - 1) * GRID_SIZE)
        for snake_pos in self.snake_positions:
            if self.position == snake_pos:
                self.randomize()

    def draw(self, surf):
        """Draw the apple on the surface
        Args:
            self: Instance itself
            surf: Surface to draw on
        Return: None
        """
        draw_box(surf, self.color, self.position)

    def draw_gl(self):
        """Draw the wall on the surface
        Args:
            self: Instance itself
        Return: None
        """
        draw_box_gl(self.color, self.position)


class Wall(object):
    def __init__(self, apple_pos, snake_pos):
        self.position = (0, 0)
        self.apple_pos = apple_pos
        self.snake_pos = snake_pos
        self.color = (0, 0, 0)
        self.randomize()

    def randomize(self):
        """Randomly set the position of the wall
        Args:
            self: Instance itself
        Return: None
        """
        self.position = (random.randint(0, GRID_WIDTH - 1) * GRID_SIZE, random.randint(0, GRID_HEIGHT - 1) * GRID_SIZE)
        if self.position == self.apple_pos or self.position == self.snake_pos:
            self.randomize()

    def draw(self, surf):
        """Draw the wall on the surface
        Args:
            self: Instance itself
            surf: Surface to draw on
        Return: None
        """
        draw_box(surf, self.color, self.position)

    def draw_gl(self):
        """Draw the wall on the surface
        Args:
            self: Instance itself
        Return: None
        """
        draw_box_gl(self.color, self.position)


# noinspection PyAttributeOutsideInit
class SnakeAgent(object):
    def __init__(self):
        self.restart()
        self.head_color = (0, 0, 255)
        self.body_color = (0, 0, 0)

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
            surf: Surface to draw
        """
        draw_box(surf, self.head_color, self.positions[0])
        for p in self.positions[1:]:
            draw_box(surf, self.body_color, p)

    def draw_gl(self):
        """Draw the wall on the surface
        Args:
            self: Instance itself
        Return: None
        """
        draw_box_gl(self.head_color, self.positions[0])
        for p in self.positions[1:]:
            draw_box_gl(self.body_color, p)


class SnakeEnv(gym.Env):
    def __init__(self, seed=None, mode="cheat", max_iter=1000, render="pygame"):
        self.seed = seed

        assert mode in ["cheat", "hardworking"], "You're mode should be in [cheat, hardworking]."
        assert render in ["pygame", "pyglet"], "You're mode should be in [pygame, pyglet]"

        # Initialize the render screen to None
        self.screen = None

        # The maximum iteration for an env is `max_iter` steps (default to 1000)
        self.max_iter = max_iter
        self.mode = mode
        self.render_mode = render
        self.iter_count = 0

        if self.render_mode == "pygame":
            import pygame
        else:
            import pyglet

        # Initialize the agent and the target
        self.snake = SnakeAgent()
        self.apple = Apple(self.snake.positions)
        self.walls = [Wall(self.apple.position, self.snake.get_head_position()) for _ in range(5)]
        self.last_head_pos = self.snake.get_head_position()

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

        # Curr state
        self.state = None

    def step(self, choose_act):
        """The reward are defined as follows:
        1. Done: (crush on itself or crush on walls) -10
        2. Reach the apple: 20
        3. Default per-step reward of -1
        """

        self.snake.point(action_dict[choose_act])
        self.iter_count += 1
        done = self.snake.move()
        reach = check_eat(self.snake, self.apple)
        crush = check_crush(self.snake, self.walls)
        new_obs = {
            "frame": self._get_frame(),
            "1D-state": self._get_1D_state()
        }

        if self.iter_count == self.max_iter or crush:
            done = True

        if done:
            reward = -10
        elif reach:
            reward = 20
        elif self.mode == "cheat":
            cur_dis = hamming_dis(self.snake.get_head_position(), self.apple.position)
            last_dis = hamming_dis(self.last_head_pos, self.apple.position)
            reward = -(cur_dis - last_dis)
        else:
            reward = -0.03 * self.snake.length

        self.last_head_pos = self.snake.get_head_position()
        self.state = new_obs

        return new_obs, reward, done, None

    def reset(self):
        # Reset the agent and the target
        self.snake.restart()
        self.apple.randomize()
        self.iter_count = 0

        # Get the current state
        curr_frame = self._get_frame()
        curr_1D_state = self._get_1D_state()

        self.last_head_pos = self.snake.get_head_position()

        self.state = {
            "1D_state": curr_1D_state,
            "frame": curr_frame,
        }

        return self.state

    def render(self, mode="human"):
        if self.render_mode == "pygame":
            if self.screen is None:
                pygame.init()
                pygame.display.set_caption("Snake")
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
            for wall in self.walls:
                wall.draw(surface)

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
        else:
            if self.screen is None:
                # Set up window and its white background
                self.screen = pyglet.canvas.get_display()
                screen = self.screen.get_screens()
                config = screen[0].get_best_config()  # selecting the first screen
                context = config.create_context(None)
                self.window = pyglet.window.Window(
                    SCREEN_WIDTH, SCREEN_HEIGHT,
                    display=self.screen, context=context,
                    config=config
                )
                self.window.set_caption("Snake")
                pyglet.gl.glClearColor(1, 1, 1, 1)

            self.window.clear()
            self.window.switch_to()
            self.window.dispatch_events()

            pyglet.text.Label(
                str(self.snake.length - 1), x=20, y=SCREEN_HEIGHT - 20,
                color=(10, 10, 10, 255), font_size=20
            ).draw()
            self.snake.draw_gl()
            self.apple.draw_gl()
            for wall in self.walls:
                wall.draw_gl()

            self.window.flip()
            time.sleep(1 / (FPS + int(self.snake.length / 3)))

    def close(self):
        if self.render_mode == "pygame":
            pygame.quit()
        else:
            self.screen.close()
        self.screen = None

    def _get_frame(self):
        # Create white frame
        empty_frame = np.full((SCREEN_WIDTH, SCREEN_HEIGHT, 3), 255, dtype=np.uint8)

        # Put in snake body with color black
        head_y, head_x = int(self.snake.positions[0][0]), int(self.snake.positions[0][1])
        empty_frame[head_x:head_x + GRID_SIZE, head_y:head_y + GRID_SIZE] = self.snake.head_color
        for x, y in self.snake.positions[1:]:
            # x, y axis should interchange
            y, x = int(x), int(y)
            empty_frame[x:x + GRID_SIZE, y:y + GRID_SIZE] = self.snake.body_color

        # Put in walls with color black
        for wall in self.walls:
            # x, y axis should interchange
            y, x = int(wall.position[0]), int(wall.position[1])
            empty_frame[x:x + GRID_SIZE, y:y + GRID_SIZE] = wall.color

        # Put in apple with color red
        apple_y, apple_x = self.apple.position
        empty_frame[apple_x:apple_x + GRID_SIZE, apple_y:apple_y + GRID_SIZE] = self.apple.color

        return empty_frame

    def _get_1D_state(self):
        return self.snake.positions[0][0] + self.snake.positions[0][1]

    def _get_distance(self):
        snake_head_x, snake_head_y = self.snake.positions[0]
        apple_x, apple_y = self.apple.position

        return abs(snake_head_x - apple_x) / SCREEN_WIDTH + abs(snake_head_y - apple_y) / SCREEN_HEIGHT
