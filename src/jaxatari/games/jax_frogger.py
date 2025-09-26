# Gruppemittglieder:
# - Matias Heredia
# - Santiago Ramirez

import os
import time
import sys
import pygame
from functools import partial
from typing import NamedTuple, Tuple
import jax.lax
import jax
import jax.numpy as jnp
import chex
from functools import partial
from typing import NamedTuple, Tuple
import jaxatari.spaces as spaces
from jaxatari.environment import JaxEnvironment, JAXAtariAction as Action

class FroggerConstants(NamedTuple):
    WIDTH: int = 160
    HEIGHT: int = 210
    NUM_LANES: int = 13
    LANE_HEIGHT: int = HEIGHT // NUM_LANES
    FROG_SIZE: Tuple[int, int] = (10, 10)
    CAR_SIZE: Tuple[int, int] = (20, 10)
    LOG_SIZE: Tuple[int, int] = (30, 10)
    FROG_START_X: int = WIDTH // 2
    FROG_START_Y: int = (NUM_LANES - 1) * LANE_HEIGHT
    GOAL_Y: int = 0

class FroggerState(NamedTuple):
    frog_x: chex.Array
    frog_y: chex.Array
    cars: chex.Array
    logs: chex.Array
    step_counter: chex.Array
    game_over: chex.Array
    reached_goal: chex.Array

class FroggerObservation(NamedTuple):
    frog: Tuple[int, int]
    cars: chex.Array
    logs: chex.Array

class FroggerInfo(NamedTuple):
    time: chex.Array
    BACKGROUND_COLOR = (0, 0, 0)

class JaxFrogger(JaxEnvironment[FroggerState, FroggerObservation, FroggerInfo, FroggerConstants]):
    def _get_observation(self, state: FroggerState) -> FroggerObservation:
        return FroggerObservation(
            frog=(state.frog_x, state.frog_y),
            cars=state.cars,
            logs=state.logs
        )
    def __init__(self, consts: FroggerConstants = None):
        if consts is None:
            consts = FroggerConstants()
        super().__init__(consts)
        self.consts = consts
        self.action_set = [Action.NOOP, Action.UP, Action.DOWN, Action.LEFT, Action.RIGHT]
    def step(self, state, action):
        # Use Action constants for comparison
        frog_x = jax.lax.cond(action == Action.LEFT, lambda x: x - self.consts.LANE_HEIGHT, lambda x: x, state.frog_x)
        frog_x = jax.lax.cond(action == Action.RIGHT, lambda x: x + self.consts.LANE_HEIGHT, lambda x: x, frog_x)
        frog_y = jax.lax.cond(action == Action.UP, lambda y: y - self.consts.LANE_HEIGHT, lambda y: y, state.frog_y)
        frog_y = jax.lax.cond(action == Action.DOWN, lambda y: y + self.consts.LANE_HEIGHT, lambda y: y, frog_y)
        frog_x = jnp.clip(frog_x, 0, self.consts.WIDTH - self.consts.FROG_SIZE[0])
        frog_y = jnp.clip(frog_y, 0, self.consts.HEIGHT - self.consts.FROG_SIZE[1])
        frog_y = frog_y.astype(int)
        cars = state.cars
        for lane in range(self.consts.NUM_LANES):
            if lane % 2 == 0:
                cars = cars.at[lane].set(jnp.roll(cars[lane], 1))
            elif lane % 2 == 1:
                cars = cars.at[lane].set(jnp.roll(cars[lane], -1))
        logs = state.logs
        for lane in range(self.consts.NUM_LANES):
            if lane % 2 == 0:
                logs = logs.at[lane].set(jnp.roll(logs[lane], 1))
            elif lane % 2 == 1:
                logs = logs.at[lane].set(jnp.roll(logs[lane], -1))
        lane_index = frog_y // self.consts.LANE_HEIGHT
        car_slice = jax.lax.dynamic_slice(
            cars,
            (lane_index, frog_x),
            (1, self.consts.FROG_SIZE[0])
        )
        collision = jnp.any(car_slice)
        reached_goal = frog_y == self.consts.GOAL_Y
        logs_slice = jax.lax.dynamic_slice(
            logs,
            (lane_index, frog_x),
            (1, self.consts.FROG_SIZE[0])
        )
        on_log = jnp.any(logs_slice)
        def move_with_log(frog_x):
            return jax.lax.cond(
                lane_index % 2 == 0,
                lambda x: x + 1,
                lambda x: x - 1,
                frog_x
            )
        frog_x = jax.lax.cond(on_log, move_with_log, lambda x: x, frog_x)
        frog_x = jnp.clip(frog_x, 0, self.consts.WIDTH - self.consts.FROG_SIZE[0])
        in_water = jnp.logical_and(lane_index >= 1, lane_index <= 5)
        collision = jnp.logical_or(collision, jnp.logical_and(in_water, jnp.logical_not(on_log)))
        new_state = state._replace(
            frog_x=frog_x,
            frog_y=frog_y,
            cars=cars,
            logs=logs,
            step_counter=state.step_counter + 1,
            game_over=collision,
            reached_goal=reached_goal
        )
        reward = jnp.where(reached_goal, 1.0, 0.0)
        observation = self._get_observation(new_state)
        info = FroggerInfo(time=new_state.step_counter)
        return new_state, observation, reward, collision, reached_goal, info

    def action_space(self):
        return spaces.Discrete(len(self.action_set))

    def observation_space(self):
        return spaces.Box(
            low=0,
            high=255,
            shape=(self.consts.HEIGHT, self.consts.WIDTH, 3),
            dtype=jnp.uint8
        )

    def reset(self, key=None):
        frog_x = jnp.array(self.consts.FROG_START_X)
        frog_y = jnp.array(self.consts.FROG_START_Y)
        cars = jnp.zeros((self.consts.NUM_LANES, self.consts.WIDTH))
        logs = jnp.zeros((self.consts.NUM_LANES, self.consts.WIDTH))
        # place cars in lanes
        for lane in range(self.consts.NUM_LANES):
            if lane == 7:
                car_positions = [30, 130]
            elif lane == 8:
                car_positions = [30, 130]
            elif lane == 9:
                car_positions = [0, 60, 120, 170]
            elif lane == 10:
                car_positions = [50, 130]
            elif lane == 11:
                car_positions = [10, 90]
            else:
                car_positions = []
            for car_x in car_positions:
                cars = cars.at[lane, car_x].set(2)
                if self.consts.CAR_SIZE[0] > 1:
                    cars = cars.at[lane, car_x+1:car_x+self.consts.CAR_SIZE[0]].set(jnp.ones(self.consts.CAR_SIZE[0]-1))

        # place logs in lanes
        for lane in range(self.consts.NUM_LANES):
            if lane == 1:
                log_positions = [80, 110]
            elif lane == 2:
                log_positions = [20, 80]
            elif lane == 3:
                log_positions = [80]
            elif lane == 4:
                log_positions = [20, 100, 130]
            elif lane == 5:
                log_positions = [50, 130]
            else:
                log_positions = []
            for log_x in log_positions:
                logs = logs.at[lane, log_x].set(2)
                if self.consts.LOG_SIZE[0] > 1:
                    logs = logs.at[lane, log_x+1:log_x+self.consts.LOG_SIZE[0]].set(jnp.ones(self.consts.LOG_SIZE[0]-1))

        # Finalize and return initial state and observation
        state = FroggerState(
            frog_x=frog_x,
            frog_y=frog_y,
            cars=cars,
            logs=logs,
            step_counter=jnp.array(0),
            game_over=jnp.array(False),
            reached_goal=jnp.array(False)
        )
        observation = self._get_observation(state)
        return state, observation


def load_sprites():
    """Load all sprites for the Frogger game."""
    game = JaxFrogger()
    pink_car = pygame.image.load(os.path.join("src", "jaxatari", "games", "sprites", "frogger", "pink_car.png")).convert_alpha()
    green_car = pygame.image.load(os.path.join("src", "jaxatari", "games", "sprites", "frogger", "green_car.png")).convert_alpha()
    purple_car = pygame.image.load(os.path.join("src", "jaxatari", "games", "sprites", "frogger", "purple_car.png")).convert_alpha()
    log_sprite = pygame.image.load(os.path.join("src", "jaxatari", "games", "sprites", "frogger", "small_log.png")).convert_alpha()
    frog_up = pygame.image.load(os.path.join("src", "jaxatari", "games", "sprites", "frogger", "frog_up.png")).convert_alpha()
    frog_down = pygame.image.load(os.path.join("src", "jaxatari", "games", "sprites", "frogger", "frog_down.png")).convert_alpha()
    frog_left = pygame.image.load(os.path.join("src", "jaxatari", "games", "sprites", "frogger", "frog_left.png")).convert_alpha()
    frog_right = pygame.image.load(os.path.join("src", "jaxatari", "games", "sprites", "frogger", "frog_right.png")).convert_alpha()
    happy_frog = pygame.image.load(os.path.join("src", "jaxatari", "games", "sprites", "frogger", "happy_frog.png")).convert_alpha()

    # Scale sprites to match game dimensions using game.consts
    pink_car = pygame.transform.scale(pink_car, (game.consts.CAR_SIZE[0] * 3, game.consts.CAR_SIZE[1] * 3))
    green_car = pygame.transform.scale(green_car, (game.consts.CAR_SIZE[0] * 3, game.consts.CAR_SIZE[1] * 3))
    purple_car = pygame.transform.scale(purple_car, (game.consts.CAR_SIZE[0] * 3, game.consts.CAR_SIZE[1] * 3))
    log_sprite = pygame.transform.scale(log_sprite, (game.consts.LOG_SIZE[0] * 3, game.consts.LOG_SIZE[1] * 3))
    frog_up = pygame.transform.scale(frog_up, (game.consts.FROG_SIZE[0] * 3, game.consts.FROG_SIZE[1] * 3))
    frog_down = pygame.transform.scale(frog_down, (game.consts.FROG_SIZE[0] * 3, game.consts.FROG_SIZE[1] * 3))
    frog_left = pygame.transform.scale(frog_left, (game.consts.FROG_SIZE[0] * 3, game.consts.FROG_SIZE[1] * 3))
    frog_right = pygame.transform.scale(frog_right, (game.consts.FROG_SIZE[0] * 3, game.consts.FROG_SIZE[1] * 3))
    happy_frog = pygame.transform.scale(happy_frog, (game.consts.FROG_SIZE[0] * 3, game.consts.FROG_SIZE[1] * 3))
    return pink_car, green_car, purple_car, log_sprite, frog_up, frog_down, frog_right, frog_left, happy_frog

if __name__ == "__main__":
    # Initialize Pygame
    pygame.init()
    # Define action constants for main block
    NOOP = Action.NOOP
    UP = Action.UP
    DOWN = Action.DOWN
    LEFT = Action.LEFT
    RIGHT = Action.RIGHT
    BACKGROUND_COLOR = (0, 0, 0)
    game = JaxFrogger()
    screen = pygame.display.set_mode((game.consts.WIDTH * 3, game.consts.HEIGHT * 3))  # Scale up for better visibility
    pygame.display.set_caption("Frogger Game")
    clock = pygame.time.Clock()

    # Initialize the game
    state, observation = game.reset()

    # Before the game loop
    homes_filled = [False] * 5  # Track filled blue squares
    patch_count = 9
    patch_width = game.consts.WIDTH * 3 // patch_count
    blue_home_x = [i * patch_width for i in range(0, 9, 2)]  # x positions of blue squares in lane 0

    # Game loop
    running = True
    frozen = False  # New: track if the game is frozen after game over
    lives = 4  # New: initialize player lives

    last_direction = UP
    while running:
        action = NOOP
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_r:
                    state, observation = game.reset()
                    lives = 4
                    frozen = False
                    homes_filled = [False] * 5
                elif not frozen:
                    if event.key == pygame.K_UP:
                        action = UP
                        last_direction = UP
                    elif event.key == pygame.K_DOWN:
                        action = DOWN
                        last_direction = DOWN
                    elif event.key == pygame.K_LEFT:
                        action = LEFT
                        last_direction = LEFT
                    elif event.key == pygame.K_RIGHT:
                        action = RIGHT
                        last_direction = RIGHT
            elif event.type == pygame.MOUSEBUTTONDOWN and frozen:
                state, observation = game.reset()
                lives = 4
                frozen = False
                homes_filled = [False] * 5

        # Only step the game if not frozen
        if not frozen:
            prev_game_over = getattr(state, 'game_over', False)
            state, observation, reward, collision, reached_goal, info = game.step(state, action)
            # If just died (collision or water), decrease lives and reset frog if lives remain
            if not prev_game_over and state.game_over:
                lives -= 1
                if lives > 0:
                    state = state._replace(
                        frog_x=jnp.array(game.consts.FROG_START_X),
                        frog_y=jnp.array(game.consts.FROG_START_Y),
                        game_over=jnp.array(False)
                    )
                else:
                    frozen = True

        # Render the game
        screen.fill((0,0,0))  # Clear the screen
        # Draw lane backgrounds
        for lane in range(game.consts.NUM_LANES):
            if lane == 0:
                # Draw 5 blue water patches and 4 green grass patches, evenly spaced
                patch_count = 9
                patch_width = game.consts.WIDTH * 3 // patch_count
                for i in range(patch_count):
                    color = (0, 0, 255) if i % 2 == 0 else (0, 128, 0)
                    pygame.draw.rect(
                        screen,
                        color,
                        pygame.Rect(
                            i * patch_width,
                            lane * game.consts.LANE_HEIGHT * 3,
                            patch_width,
                            game.consts.LANE_HEIGHT * 3,
                        ),
                    )
                continue
            elif lane == 6 or lane == 12:
                lane_color = (235, 229, 100)  # Light Green
            elif 1 <= lane <= 5:
                lane_color = (0, 0, 255)    # Black
            elif 7 <= lane <= 11:
                lane_color = (0, 0, 0)  # Blue
            else:
                lane_color = BACKGROUND_COLOR  # fallback
            pygame.draw.rect(
                screen,
                lane_color,
                pygame.Rect(
                    0,
                    lane * game.consts.LANE_HEIGHT * 3,
                    game.consts.WIDTH * 3,
                    game.consts.LANE_HEIGHT * 3,
                ),
            )

        # Load sprites
        pink_car, green_car, purple_car, log_sprite, frog_up, frog_down, frog_right, frog_left, happy_frog = load_sprites()

        # Draw logs
        for lane in range(game.consts.NUM_LANES):
            log_starts = jnp.where(state.logs[lane] == 2)[0]
            for x in log_starts:
                screen.blit(
                    log_sprite,
                    (int(x) * 3, int(lane * game.consts.LANE_HEIGHT) * 3),
                )

        # Draw cars
        for lane in range(game.consts.NUM_LANES):
            car_starts = jnp.where(state.cars[lane] == 2)[0]
            for x in car_starts:
                screen.blit(
                    pink_car,  # Use pink car sprite
                    (int(x) * 3, int(lane * game.consts.LANE_HEIGHT) * 3),
                )

        # Draw the frog
        # Ensure last_direction is always defined
        if 'last_direction' not in locals():
            last_direction = UP
        frog_sprite = {
            UP: frog_up,
            DOWN: frog_down,
            LEFT: frog_left,
            RIGHT: frog_right,
        }.get(last_direction, frog_up)
        screen.blit(
            frog_sprite,
            (int(state.frog_x) * 3, int(state.frog_y) * 3),
        )

        # Draw lives as white dots in the bottom left corner
        for i in range(lives):
            pygame.draw.rect(
                screen,
                (255, 255, 255),
                pygame.Rect(
                    10 + i * 20,  # Spacing between lives
                    game.consts.HEIGHT * 3 - 30,  # Position at the bottom
                    10,  # Width of the life dot
                    10,  # Height of the life dot
                ),
            )

        # Check for reaching a blue home square in lane 0
        if not frozen and int(state.frog_y) == 0:
            frog_center_x = int(state.frog_x) * 3 + (game.consts.FROG_SIZE[0] * 3) // 2
            on_blue = False
            for idx, home_x in enumerate(blue_home_x):
                # Only allow filling if not already filled and frog is centered on the blue patch
                if not homes_filled[idx] and abs(frog_center_x - (home_x + patch_width // 2)) < patch_width // 2:
                    homes_filled[idx] = True
                    on_blue = True
                    # Reset frog to start, but keep it visible in the home
                    state = state._replace(
                        frog_x=jnp.array(game.consts.FROG_START_X),
                        frog_y=jnp.array(game.consts.FROG_START_Y),
                        game_over=jnp.array(False)
                    )
                    break
            if not on_blue:
                # If not on a blue patch, treat as death (fall in water/grass)
                lives -= 1
                if lives > 0:
                    state = state._replace(
                        frog_x=jnp.array(game.consts.FROG_START_X),
                        frog_y=jnp.array(game.consts.FROG_START_Y),
                        game_over=jnp.array(False)
                    )
                else:
                    frozen = True

        # Win condition
        if all(homes_filled):
            frozen = True
            font = pygame.font.SysFont(None, 72)
            text = font.render("Winner!", True, (0, 255, 0))
            text_rect = text.get_rect(center=(game.consts.WIDTH * 3 // 2, game.consts.HEIGHT * 3 // 2 - 30))
            screen.blit(text, text_rect)

            font2 = pygame.font.SysFont(None, 60)
            smiley = font2.render(":)", True, (0, 255, 0))
            smiley_rect = smiley.get_rect(center=(game.consts.WIDTH * 3 // 2, game.consts.HEIGHT * 3 // 2 + 30))
            screen.blit(smiley, smiley_rect)

            font3 = pygame.font.SysFont(None, 36)
            text3 = font3.render("Press R or click to restart", True, (255, 255, 255))
            text3_rect = text3.get_rect(center=(game.consts.WIDTH * 3 // 2, game.consts.HEIGHT * 3 // 2 + 90))
            screen.blit(text3, text3_rect)

        # If game over, freeze and show message
        if lives == 0 and state.game_over:
            frozen = True
            font = pygame.font.SysFont(None, 72)
            text = font.render("Game Over", True, (255, 0, 0))
            text_rect = text.get_rect(center=(game.consts.WIDTH * 3 // 2, game.consts.HEIGHT * 3 // 2))
            screen.blit(text, text_rect)

            font2 = pygame.font.SysFont(None, 36)
            text2 = font2.render("Press R or click to restart", True, (255, 255, 255))
            text2_rect = text2.get_rect(center=(game.consts.WIDTH * 3 // 2, game.consts.HEIGHT * 3 // 2 + 60))
            screen.blit(text2, text2_rect)

        # Draw frogs in filled home slots (lane 0)
        for idx, filled in enumerate(homes_filled):
            if filled:
                screen.blit(
                    happy_frog,  # Show happy frog sprite
                    (blue_home_x[idx] + (patch_width - game.consts.FROG_SIZE[0] * 3) // 2, 0)
                )

        # Update the display
        pygame.display.flip()

        # Cap the frame rate
        clock.tick(30)