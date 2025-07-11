# Gruppemittglieder:
# - Hannah Sebastian
# - Matias Heredia
# - Santiago Ramirez
import os
import time
import sys
from functools import partial
from typing import NamedTuple, Tuple
import jax.lax
import jax.numpy as jnp
import chex
import pygame
from gymnax.environments import spaces
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from jaxatari.rendering import atraJaxis as aj
from jaxatari.environment import JaxEnvironment

# Constants for Frogger environment
WIDTH = 160
HEIGHT = 210
NUM_LANES = 13
LANE_HEIGHT = HEIGHT // NUM_LANES 
FROG_SIZE = (10, 10)
FROG_VEL = 2
CAR_SIZE = (20, 10)
LOG_SIZE = (30, 10)
FROG_START_X = WIDTH // 2
FROG_START_Y = (NUM_LANES - 1) * LANE_HEIGHT
GOAL_Y = 0

# Colors
BACKGROUND_COLOR = 0, 0, 0
FROG_COLOR = 0, 255, 0
CAR_COLOR_RIGHT = 255, 0, 0
CAR_COLOR_LEFT = 0, 0, 255
LOG_COLOR = 139, 69, 19

# Actions
NOOP = 0
UP = 1
DOWN = 2
LEFT = 3
RIGHT = 4

UP_COOLDOWN_FRAMES = 10  # Number of frames to wait before allowing another UP move
up_cooldown_counter = 0

DOWN_COOLDOWN_FRAMES = 10  # Number of frames to wait before allowing another DOWN move
down_cooldown_counter = 0

last_direction = UP  # or whatever you want as the default

def get_human_action() -> chex.Array:
    """Get human action from keyboard input."""

    keys = pygame.key.get_pressed()
    if keys[pygame.K_UP]:
        return jnp.array(UP)
    elif keys[pygame.K_DOWN]:
        return jnp.array(DOWN)
    elif keys[pygame.K_LEFT]:
        return  jnp.array(LEFT) 
    elif keys[pygame.K_RIGHT]:
        return jnp.array(RIGHT)
    else:
        return jnp.array(NOOP)


class FroggerState(NamedTuple):
    frog_x: chex.Array
    frog_y: chex.Array
    cars: chex.Array  # Positions of cars in each lane
    logs: chex.Array  # Positions of logs in the river
    step_counter: chex.Array
    game_over: chex.Array
    reached_goal: chex.Array

class FroggerObservation(NamedTuple):
    frog: Tuple[int, int]
    cars: chex.Array
    logs: chex.Array

class FroggerInfo(NamedTuple):
    time: jnp.ndarray

class JaxFrogger(JaxEnvironment[FroggerState, FroggerObservation, FroggerInfo]):  
    def __init__(self):
        super().__init__()
        self.action_set = {NOOP, DOWN, LEFT, RIGHT, UP}
   
    @partial(jax.jit, static_argnums=(0,))
    def reset(self) -> FroggerState:
        frog_x = jnp.array(FROG_START_X)
        frog_y = jnp.array(FROG_START_Y)
        cars = jnp.zeros((NUM_LANES, WIDTH))  # Initialize cars
        logs = jnp.zeros((NUM_LANES, WIDTH))  # Initialize logs
        # place cars in lanes
        for lane in range(NUM_LANES):  #Corrected car placement for collision
            if lane == 7:
                car_positions = [30, 130]
                for car_x in car_positions:
                    cars = cars.at[lane, car_x].set(2)
                    if CAR_SIZE[0] > 1:
                        cars = cars.at[lane, car_x+1:car_x+CAR_SIZE[0]].set(jnp.ones(CAR_SIZE[0]-1))

            elif lane == 8:
                car_positions = [30, 130]
                for car_x in car_positions:
                    cars = cars.at[lane, car_x].set(2)
                    if CAR_SIZE[0] > 1:
                        cars = cars.at[lane, car_x+1:car_x+CAR_SIZE[0]].set(jnp.ones(CAR_SIZE[0]-1))

            elif lane == 9:
                car_positions = [0, 60, 120, 170]
                for car_x in car_positions:
                    cars = cars.at[lane, car_x].set(2)
                    if CAR_SIZE[0] > 1:
                        cars = cars.at[lane, car_x+1:car_x+CAR_SIZE[0]].set(jnp.ones(CAR_SIZE[0]-1))

            elif lane == 10:
                car_positions = [50, 130]
                for car_x in car_positions:
                    cars = cars.at[lane, car_x].set(2)
                    if CAR_SIZE[0] > 1:
                        cars = cars.at[lane, car_x+1:car_x+CAR_SIZE[0]].set(jnp.ones(CAR_SIZE[0]-1))
            
            elif lane == 11:
                car_positions = [10, 90]
                for car_x in car_positions:
                    cars = cars.at[lane, car_x].set(2)
                    if CAR_SIZE[0] > 1:
                        cars = cars.at[lane, car_x+1:car_x+CAR_SIZE[0]].set(jnp.ones(CAR_SIZE[0]-1))
            else:
                continue

        
        # Place logs in lanes. 
        for lane in range(NUM_LANES): # Corrected log placement for frog movement on log
            if lane == 1:
                log_positions = [80, 110]
                for log_x in log_positions:
                    logs = logs.at[lane, log_x].set(2)
                    if LOG_SIZE[0] > 1:
                        logs = logs.at[lane, log_x + 1 : log_x + LOG_SIZE[0]].set(jnp.ones(LOG_SIZE[0]-1))
                
            elif lane == 2:
                log_positions = [20, 80]
                for log_x in log_positions:
                    logs = logs.at[lane, log_x].set(2)
                    if LOG_SIZE[0] > 1:
                        logs = logs.at[lane, log_x + 1 : log_x + LOG_SIZE[0]].set(jnp.ones(LOG_SIZE[0]-1))
            elif lane == 3:
                log_positions = [80]
                for log_x in log_positions:
                    logs = logs.at[lane, log_x].set(2)
                    if LOG_SIZE[0] > 1:
                        logs = logs.at[lane, log_x + 1 : log_x + LOG_SIZE[0]].set(jnp.ones(LOG_SIZE[0]-1))
            elif lane == 4:
                log_positions = [20, 100, 130]
                for log_x in log_positions:
                    logs = logs.at[lane, log_x].set(2)
                    if LOG_SIZE[0] > 1:
                        logs = logs.at[lane, log_x + 1 : log_x + LOG_SIZE[0]].set(jnp.ones(LOG_SIZE[0]-1))
            elif lane == 5: 
                log_positions = [50, 130]
                for log_x in log_positions:
                    logs = logs.at[lane, log_x].set(2)
                    if LOG_SIZE[0] > 1:
                        logs = logs.at[lane, log_x + 1 : log_x + LOG_SIZE[0]].set(jnp.ones(LOG_SIZE[0]-1))
            else:
                continue


        # TODO LATER add Turtles in line 2 and 5, for now just logs


        state = FroggerState(
            frog_x=frog_x,
            frog_y=frog_y,
            cars=cars,
            logs=logs,
            step_counter=jnp.array(0),
            game_over=jnp.array(False),
            reached_goal=jnp.array(False),
        )
        return state, self._get_observation(state)

    @partial(jax.jit, static_argnums=(0,))
    def step(self, state: FroggerState, action: chex.Array) -> Tuple[FroggerState, FroggerObservation, float, bool, FroggerInfo]:
        
        # Debug print for action
        #jax.debug.print("Action: {}", action)

        # Update frog position
        frog_x = jax.lax.cond(action == LEFT, lambda x: x - LANE_HEIGHT, lambda x: x, state.frog_x)
        frog_x = jax.lax.cond(action == RIGHT, lambda x: x + LANE_HEIGHT, lambda x: x, frog_x) 
        frog_y = jax.lax.cond(action == UP, lambda y: y - LANE_HEIGHT, lambda y: y, state.frog_y)  
        frog_y = jax.lax.cond(action == DOWN, lambda y: y + LANE_HEIGHT, lambda y: y, frog_y)

        # Debug print for frog position
        #jax.debug.print("Frog position: ({}, {})", frog_x, frog_y)

        # Clamp frog position within bounds
        frog_x = jnp.clip(frog_x, 0, WIDTH - FROG_SIZE[0])
        frog_y = jnp.clip(frog_y, 0, HEIGHT - FROG_SIZE[1]) # TODO Tries to go out of margin
        
        # Cars traveling on the screen using bool arrays:
        #
        #    <- WIDTH ->           <- WIDTH ->            <- WIDTH ->
        #     0   1   2             0   1   2              0   1   2
        # 0 |-0-|-0-|-0-|L      0 |-0-|-0-|-1-|L       0 |-0-|-1-|-0-|L
        # 1 |-0-|-0-|-0-|A      1 |-0-|-0-|-0-|A       1 |-0-|-0-|-0-|A
        # 2 |-0-|-0-|-0-|N      2 |-0-|-0-|-1-|N       2 |-0-|-1-|-0-|N
        # 3 |-0-|-0-|-0-|E      3 |-0-|-0-|-0-|E       3 |-0-|-0-|-0-|E
        # 4 |-0-|-0-|-0-|S      4 |-0-|-0-|-1-|S       4 |-0-|-1-|-0-|S

        cars = state.cars
        for lane in range(NUM_LANES):
            if lane % 2 == 0:
                # roll cars to the left
                #jax.debug.print("Cars array: {}", cars)
                cars = cars.at[lane].set(jnp.roll(cars[lane], 1))

            elif lane % 2 == 1:
                # roll cars to the right
                #jax.debug.print("Cars array: {}", cars)
                cars = cars.at[lane].set(jnp.roll(cars[lane], -1))
        
        # Debug print to verify the updated cars array
        #jax.debug.print("Updated cars array: {}", cars)

        # Update logs position
        logs = state.logs
        for lane in range(NUM_LANES):
            if lane % 2 == 0:
                # roll logs to the left
                logs = logs.at[lane].set(jnp.roll(logs[lane], 1))
            elif lane % 2 == 1:
                # roll logs to the right
                logs = logs.at[lane].set(jnp.roll(logs[lane], -1))
        
        # Debug print to verify the updated logs array 
        #jax.debug.print("Updated logs array: {}", logs)

        # Check for collisions with cars 
        lane_index = frog_y // LANE_HEIGHT
        car_slice = jax.lax.dynamic_slice(
            state.cars, 
            (lane_index, frog_x),  # Start indices for slicing
            (1, FROG_SIZE[0])      # Slice size (1 row, FROG_SIZE[0] columns)
        )
        collision = jnp.any(car_slice) 

        # Check if frog reached the goal
        reached_goal = frog_y == GOAL_Y

        # Frog is on a log
        lane_index = frog_y // LANE_HEIGHT
        logs_slice = jax.lax.dynamic_slice(
            state.logs, 
            (lane_index, frog_x),  # Start indices for slicing
            (1, FROG_SIZE[0])      # Slice size (1 row, FROG_SIZE[0] columns)
        )
        on_log = jnp.any(logs_slice) 
        
        # Frog moves with the log
        def move_with_log(frog_x):
            # Even lanes: logs move left (decrease x), odd lanes: right (increase x)
            return jax.lax.cond(
                lane_index % 2 == 0,
                lambda x: x + 1, 
                lambda x: x - 1,  
                frog_x
            )

        frog_x = jax.lax.cond(on_log, move_with_log, lambda x: x, frog_x)
        frog_x = jnp.clip(frog_x, 0, WIDTH - FROG_SIZE[0])  # Ensure frog stays in bounds

       
        # Game over if not on log 
        in_water = jnp.logical_and(lane_index >= 1, lane_index <= 5) 
        collision = jnp.logical_or(collision, jnp.logical_and(in_water, jnp.logical_not(on_log)))

        # TODO Allow traveling on logs beyond bounds
       
        # Update game state
        new_state = state._replace(
            frog_x=frog_x,
            frog_y=frog_y,
            cars=cars,
            logs=logs,
            step_counter=state.step_counter + 1,
            game_over=collision,
            reached_goal=reached_goal,
        )

        reward = jnp.where(reached_goal, 1.0, 0.0)
        observation = self._get_observation(new_state)
        info = FroggerInfo(time=new_state.step_counter)

        return new_state, observation, reward, collision, reached_goal, info

    @partial(jax.jit, static_argnums=(0,))
    def _get_observation(self, state: FroggerState) -> FroggerObservation:
        return FroggerObservation(
            frog=(state.frog_x, state.frog_y),
            cars=state.cars,
            logs=state.logs,
        )

    def action_space(self) -> spaces.Discrete:
        return spaces.Discrete(len(self.action_set))

    def observation_space(self) -> spaces.Box:
        return spaces.Box(
            low=0,
            high=255,
            shape=(HEIGHT, WIDTH, 3),
            dtype=jnp.uint8,
        )


def load_sprites():
    """Load all sprites for the Frogger game."""
    pink_car = pygame.image.load(os.path.join("src", "jaxatari", "games", "sprites", "frogger", "pink_car.png")).convert_alpha()
    green_car = pygame.image.load(os.path.join("src", "jaxatari", "games", "sprites", "frogger", "green_car.png")).convert_alpha()
    purple_car = pygame.image.load(os.path.join("src", "jaxatari", "games", "sprites", "frogger", "purple_car.png")).convert_alpha()
    log_sprite = pygame.image.load(os.path.join("src", "jaxatari", "games", "sprites", "frogger", "small_log.png")).convert_alpha()
    frog_up = pygame.image.load(os.path.join("src", "jaxatari", "games", "sprites", "frogger", "frog_up.png")).convert_alpha()
    frog_down = pygame.image.load(os.path.join("src", "jaxatari", "games", "sprites", "frogger", "frog_down.png")).convert_alpha()
    frog_left = pygame.image.load(os.path.join("src", "jaxatari", "games", "sprites", "frogger", "frog_left.png")).convert_alpha()
    frog_right = pygame.image.load(os.path.join("src", "jaxatari", "games", "sprites", "frogger", "frog_right.png")).convert_alpha()
    happy_frog = pygame.image.load(os.path.join("src", "jaxatari", "games", "sprites", "frogger", "happy_frog.png")).convert_alpha()


    # Scale sprites to match game dimensions
    pink_car = pygame.transform.scale(pink_car, (CAR_SIZE[0] * 3, CAR_SIZE[1] * 3))
    green_car = pygame.transform.scale(green_car, (CAR_SIZE[0] * 3, CAR_SIZE[1] * 3))
    purple_car = pygame.transform.scale(purple_car, (CAR_SIZE[0] * 3, CAR_SIZE[1] * 3))
    log_sprite = pygame.transform.scale(log_sprite, (LOG_SIZE[0] * 3, LOG_SIZE[1] * 3))
    frog_up = pygame.transform.scale(frog_up, (FROG_SIZE[0] * 3, FROG_SIZE[1] * 3))
    frog_down = pygame.transform.scale(frog_down, (FROG_SIZE[0] * 3, FROG_SIZE[1] * 3))
    frog_left = pygame.transform.scale(frog_left, (FROG_SIZE[0] * 3, FROG_SIZE[1] * 3))
    frog_right = pygame.transform.scale(frog_right, (FROG_SIZE[0] * 3, FROG_SIZE[1] * 3))
    happy_frog = pygame.transform.scale(happy_frog, (FROG_SIZE[0] * 3, FROG_SIZE[1] * 3))

    return pink_car, green_car, purple_car, log_sprite, frog_up, frog_down, frog_right, frog_left, happy_frog

if __name__ == "__main__":
    # Initialize Pygame
    pygame.init()
    screen = pygame.display.set_mode((WIDTH * 3, HEIGHT * 3))  # Scale up for better visibility
    pygame.display.set_caption("Frogger Game")
    clock = pygame.time.Clock()

    # Initialize the game
    game = JaxFrogger()
    state, observation = game.reset()

    # Before the game loop
    homes_filled = [False] * 5  # Track filled blue squares
    patch_count = 9
    patch_width = WIDTH * 3 // patch_count
    blue_home_x = [i * patch_width for i in range(0, 9, 2)]  # x positions of blue squares in lane 0

    # Game loop
    running = True
    frozen = False  # New: track if the game is frozen after game over
    lives = 4  # New: initialize player lives
    while running:
        # Handle events
        action = NOOP
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                # Allow reset with 'r' at any time
                if event.key == pygame.K_r:
                    state, observation = game.reset()
                    lives = 4
                    frozen = False
                    homes_filled = [False] * 5
                elif frozen:
                    pass  # No other actions when frozen
                else:
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
            prev_game_over = state.game_over
            state, observation, reward, game_over, reached_goal, info = game.step(state, jnp.array(action))
            # If just died (collision or water), decrease lives and reset frog if lives remain
            if not prev_game_over and state.game_over:
                lives -= 1
                if lives > 0:
                    # Reset frog position only, keep cars/logs and lives
                    state = state._replace(
                        frog_x=jnp.array(FROG_START_X),
                        frog_y=jnp.array(FROG_START_Y),
                        game_over=jnp.array(False)
                    )
                else:
                    frozen = True

        # Render the game
        screen.fill(BACKGROUND_COLOR)  # Clear the screen

        ## Draw lane backgrounds
        for lane in range(NUM_LANES):
            if lane == 0:
                # Draw 5 blue water patches and 4 green grass patches, evenly spaced
                patch_count = 9
                patch_width = WIDTH * 3 // patch_count
                for i in range(patch_count):
                    if i % 2 == 0:
                        color = (0, 0, 255)  # Blue (water)
                    else:
                        color = (0, 128, 0)  # Green (grass)
                    pygame.draw.rect(
                        screen,
                        color,
                        pygame.Rect(
                            i * patch_width,
                            lane * LANE_HEIGHT * 3,
                            patch_width,
                            LANE_HEIGHT * 3,
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
                   lane * LANE_HEIGHT * 3,
                   WIDTH * 3,
                   LANE_HEIGHT * 3,
               ),
           )

        
        # Load sprites
        pink_car, green_car, purple_car, log_sprite, frog_up, frog_down, frog_right, frog_left, happy_frog = load_sprites()

        # Render the game
        #screen.blit(background_sprite, (0, 0))  # Draw the background

        # Draw logs
        for lane in range(NUM_LANES):
            log_starts = jnp.where(state.logs[lane] == 2)[0]
            for x in log_starts:
                screen.blit(
                    log_sprite,
                    (int(x) * 3, int(lane * LANE_HEIGHT) * 3),
                )

        # Draw cars
        for lane in range(NUM_LANES):
            car_starts = jnp.where(state.cars[lane] == 2)[0]
            for x in car_starts:
                screen.blit(
                    pink_car,  # Use pink car sprite
                    (int(x) * 3, int(lane * LANE_HEIGHT) * 3),
                )


        # Draw the frog
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
                    HEIGHT * 3 - 30,  # Position at the bottom
                    10,  # Width of the life dot
                    10,  # Height of the life dot
                ),
            )

        # Check for reaching a blue home square in lane 0
        if not frozen and int(state.frog_y) == 0:
            frog_center_x = int(state.frog_x) * 3 + (FROG_SIZE[0] * 3) // 2
            on_blue = False
            for idx, home_x in enumerate(blue_home_x):
                # Only allow filling if not already filled and frog is centered on the blue patch
                if not homes_filled[idx] and abs(frog_center_x - (home_x + patch_width // 2)) < patch_width // 2:
                    homes_filled[idx] = True
                    on_blue = True
                    # Reset frog to start, but keep it visible in the home
                    state = state._replace(
                        frog_x=jnp.array(FROG_START_X),
                        frog_y=jnp.array(FROG_START_Y),
                        game_over=jnp.array(False)
                    )
                    break
            if not on_blue:
                # If not on a blue patch, treat as death (fall in water/grass)
                lives -= 1
                if lives > 0:
                    state = state._replace(
                        frog_x=jnp.array(FROG_START_X),
                        frog_y=jnp.array(FROG_START_Y),
                        game_over=jnp.array(False)
                    )
                else:
                    frozen = True

        # Win condition
        if all(homes_filled):
            frozen = True
            font = pygame.font.SysFont(None, 72)
            text = font.render("Winner!", True, (0, 255, 0))
            text_rect = text.get_rect(center=(WIDTH * 3 // 2, HEIGHT * 3 // 2 - 30))
            screen.blit(text, text_rect)

            font2 = pygame.font.SysFont(None, 60)
            smiley = font2.render(":)", True, (0, 255, 0))
            smiley_rect = smiley.get_rect(center=(WIDTH * 3 // 2, HEIGHT * 3 // 2 + 30))
            screen.blit(smiley, smiley_rect)

            font3 = pygame.font.SysFont(None, 36)
            text3 = font3.render("Press R or click to restart", True, (255, 255, 255))
            text3_rect = text3.get_rect(center=(WIDTH * 3 // 2, HEIGHT * 3 // 2 + 90))
            screen.blit(text3, text3_rect)

        # If game over, freeze and show message
        if lives == 0 and state.game_over:
            frozen = True
            font = pygame.font.SysFont(None, 72)
            text = font.render("Game Over", True, (255, 0, 0))
            text_rect = text.get_rect(center=(WIDTH * 3 // 2, HEIGHT * 3 // 2))
            screen.blit(text, text_rect)

            font2 = pygame.font.SysFont(None, 36)
            text2 = font2.render("Press R or click to restart", True, (255, 255, 255))
            text2_rect = text2.get_rect(center=(WIDTH * 3 // 2, HEIGHT * 3 // 2 + 60))
            screen.blit(text2, text2_rect)

        # Draw frogs in filled home slots (lane 0)
        for idx, filled in enumerate(homes_filled):
            if filled:
                screen.blit(
                    happy_frog,  # Show happy frog sprite
                    (blue_home_x[idx] + (patch_width - FROG_SIZE[0] * 3) // 2, 0)
                )

        # Update the display
        pygame.display.flip()

        # Cap the frame rate
        clock.tick(30)

    pygame.quit()