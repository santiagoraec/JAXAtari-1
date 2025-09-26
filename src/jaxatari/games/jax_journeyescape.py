# Gruppemittglieder:
# - Santiago Ramirez
# - Matias Heredia

import os
import sys
import time
from functools import partial
from typing import NamedTuple, Tuple
import jax
import jax.numpy as jnp
import chex
import pygame
import random
from gymnax.environments import spaces

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from jaxatari.environment import JaxEnvironment

# Constants for Journey Escape Environment
WINDOW_WIDTH = 170 * 3
WINDOW_HEIGHT = 250 * 3
WIDTH = 170
HEIGHT = 250

# Entity sizes
BAND_MEMBER_SIZE = (20, 40)  # Player
ESCAPE_VEHICLE_SIZE = (50, 40)  # Goal
GROUPIE_SIZE = (20, 30)  # Enemy
PROMOTER_SIZE = (20, 30)  # Enemy
PHOTOGRAPHER_SIZE = (20, 30)  # Enemy
BARRIER_SIZE = (60, 30)  # Obstacle
ROADIE_SIZE = (20, 30)  # Ally
MANAGER_SIZE = (20, 20)  # Ally
 
# Entity types for the grid system
ENTITY_NONE = 0
ENTITY_GROUPIE = 1
ENTITY_PROMOTER = 2
ENTITY_PHOTOGRAPHER = 3
ENTITY_BARRIER = 4
ENTITY_ROADIE = 5
ENTITY_MANAGER = 6
ENTITY_ESCAPE_VEHICLE = 7 

# Velocities and wave costants
ENTITY_SPEED = 1  # Speed of downward movement
NUM_LANES = 8  # Number of horizontal lanes
LANE_HEIGHT = HEIGHT // NUM_LANES  # Height of each lane
LANE_SPACING = 25  # Minimum spacing between entities in a lane
MAX_ENTITIES_PER_LANE = 4  # Maximum entities per lane (adjust as needed)
BORDER_BOUNCE_SPEED = 0.5  # Speed component for bouncing off borders

DRAG_EFFECT_DURATION = 60  # 1 second at 60 FPS
DRAG_VELOCITY = 2.0         # Speed of downward drag
PHOTOGRAPHER_CYCLE_TIME = 60 # 1 second visibility cycle at 60 FPS
ROADIE_PROTECTION_TIME = 300 # 5 seconds at 60 FPS

# Entity spawn timer
SPAWN_TIMER = 120  # Frames between spawning new lanes (2 seconds at 60 FPS)
ESCAPE_VEHICLE_APPEARANCES = 2  # Number of times escape vehicle appears per game
ESCAPE_VEHICLE_MIN_TIME = 40    # Earliest time it can appear (40 seconds)

# Initial values
INIT_TIME = 60
INIT_CASH = 50000
BAND_MEMBER_START_X = WIDTH // 2
BAND_MEMBER_START_Y = HEIGHT - BAND_MEMBER_SIZE[1] - 10

# Colors
BACKGROUND_COLOR = (0, 50, 100)
BAND_MEMBER_COLOR = (255, 255, 0)  # Yellow
ESCAPE_VEHICLE_COLOR = (0, 255, 0)  # Green
GROUPIE_COLOR = (255, 0, 0)  # Red
PROMOTER_COLOR = (255, 100, 100)  # Light Red
PHOTOGRAPHER_COLOR = (200, 0, 0)  # Dark Red
BARRIER_COLOR = (100, 100, 100)  # Gray
ROADIE_COLOR = (0, 0, 255)  # Blue
MANAGER_COLOR = (0, 100, 255)  # Light Blue

# Actions
NOOP = 0
UP = 1
DOWN = 2
LEFT = 3
RIGHT = 4
UPRIGHT = 5
UPLEFT = 6
DOWNRIGHT = 7
DOWNLEFT = 8
RIGHTFIRE = 9
LEFTFIRE = 10
DOWNFIRE = 11
UPRIGHTFIRE = 12
UPLEFTFIRE = 13
DOWNRIGHTFIRE = 14
DOWNLEFTFIRE = 15

# Game state container
class JourneyEscapeState(NamedTuple):
    player_x: chex.Array
    player_y: chex.Array
    time_left: chex.Array
    cash_left: chex.Array
    # Lane and entity tracking
    lane_entities: chex.Array  # Shape: (NUM_LANES, MAX_ENTITIES_PER_LANE, 3) -> [x, y, entity_type]
    lane_velocities: chex.Array  # Shape: (NUM_LANES, 2) -> [vx, vy] for each lane
    lane_active: chex.Array  # Shape: (NUM_LANES,) -> boolean, which lanes are active
    lane_entity_counts: chex.Array  # Shape: (NUM_LANES,) -> how many entities in each lane
    
    # Escape vehicle tracking
    escape_vehicles_spawned: chex.Array  # How many escape vehicles have been spawned
    next_escape_spawn_time: chex.Array   # When the next escape vehicle should spawn
    
    # NEW: Entity effect tracking
    player_drag_velocity: chex.Array     # Player downward drag from groupies/barriers
    drag_timer: chex.Array              # How long the drag effect lasts
    photographer_visibility: chex.Array # Track photographer visibility state
    photographer_timer: chex.Array      # Timer for photographer visibility cycle
    invulnerability_timer: chex.Array   # Roadie protection timer (5 seconds)
    manager_protection: chex.Array      # Manager permanent protection until goal/game end
    manager_spawned: chex.Array         # Track if manager has been spawned this game

    # Collision tracking for one-time cash deduction
    last_groupie_collision: chex.Array   # Track if colliding with groupie last frame
    last_promoter_collision: chex.Array  # Track if colliding with promoter last frame
    last_photographer_collision: chex.Array # Track if colliding with photographer last frame

    spawn_timer: chex.Array
    
    # Game state
    game_over: chex.Array
    reached_goal: chex.Array
    step_counter: chex.Array
    rng_key: chex.Array



class JourneyEscapeObservation(NamedTuple):
    player: chex.Array
    entities: chex.Array
    time: chex.Array
    cash: chex.Array

class JourneyEscapeInfo(NamedTuple):
    time: chex.Array
    cash: chex.Array


# Get human action from keyboard input
def get_human_action() -> chex.Array:
    
    keys = pygame.key.get_pressed()
    if keys[pygame.K_UP]:
        return jnp.array(UP)
    elif keys[pygame.K_DOWN]:
        return jnp.array(DOWN)
    elif keys[pygame.K_LEFT]:
        return  jnp.array(LEFT) 
    elif keys[pygame.K_RIGHT]:
        return jnp.array(RIGHT)
    elif keys[pygame.K_UP] and keys[pygame.K_RIGHT]:
        return jnp.array(UPRIGHT)
    elif keys[pygame.K_UP] and keys[pygame.K_LEFT]:
        return jnp.array(UPLEFT)
    elif keys[pygame.K_DOWN] and keys[pygame.K_RIGHT]:
        return jnp.array(DOWNRIGHT)
    elif keys[pygame.K_DOWN] and keys[pygame.K_LEFT]:
        return jnp.array(DOWNLEFT)
    elif keys[pygame.K_RIGHT] and keys[pygame.K_SPACE]:
        return jnp.array(RIGHTFIRE)
    elif keys[pygame.K_LEFT] and keys[pygame.K_SPACE]:
        return jnp.array(LEFTFIRE)
    elif keys[pygame.K_DOWN] and keys[pygame.K_SPACE]:
        return jnp.array(DOWNFIRE)
    elif keys[pygame.K_UP] and keys[pygame.K_RIGHT] and keys[pygame.K_SPACE]:
        return jnp.array(UPRIGHTFIRE)
    elif keys[pygame.K_UP] and keys[pygame.K_LEFT] and keys[pygame.K_SPACE]:
        return jnp.array(UPLEFTFIRE)
    elif keys[pygame.K_DOWN] and keys[pygame.K_RIGHT] and keys[pygame.K_SPACE]:
        return jnp.array(DOWNRIGHTFIRE)
    elif keys[pygame.K_DOWN] and keys[pygame.K_LEFT] and keys[pygame.K_SPACE]:
        return jnp.array(DOWNLEFTFIRE)
    else:
        return jnp.array(NOOP)


# Helper function to get entity size and color by type
def get_entity_info(entity_type):
    """Get size and color for entity type"""
    if entity_type == ENTITY_GROUPIE:
        return GROUPIE_SIZE, GROUPIE_COLOR
    elif entity_type == ENTITY_PROMOTER:
        return PROMOTER_SIZE, PROMOTER_COLOR
    elif entity_type == ENTITY_PHOTOGRAPHER:
        return PHOTOGRAPHER_SIZE, PHOTOGRAPHER_COLOR
    elif entity_type == ENTITY_BARRIER:
        return BARRIER_SIZE, BARRIER_COLOR
    elif entity_type == ENTITY_ROADIE:
        return ROADIE_SIZE, ROADIE_COLOR
    elif entity_type == ENTITY_MANAGER:
        return MANAGER_SIZE, MANAGER_COLOR
    elif entity_type == ENTITY_ESCAPE_VEHICLE:
        return ESCAPE_VEHICLE_SIZE, ESCAPE_VEHICLE_COLOR
    else:
        return (0, 0), (0, 0, 0)

def spawn_lane_entities(rng_key, lane_idx, reset_counter=0, force_escape_vehicle=False, manager_already_spawned=False):
    """Spawn entities for a single lane"""
    base_seed = jax.random.fold_in(rng_key, lane_idx * 1000 + reset_counter * 7919)
    rng_key, subkey1 = jax.random.split(base_seed)
    rng_key, subkey2 = jax.random.split(rng_key)
    rng_key, subkey3 = jax.random.split(rng_key)
    rng_key, subkey4 = jax.random.split(rng_key)
    rng_key, subkey5 = jax.random.split(rng_key)
    
    # Determine entity type
    entity_type = jax.lax.cond(
        force_escape_vehicle,
        lambda: jnp.array(ENTITY_ESCAPE_VEHICLE),
        lambda: jax.lax.cond(
            manager_already_spawned,
            lambda: jax.random.randint(subkey1, shape=(), minval=1, maxval=6),  # Exclude manager (1-5)
            lambda: jax.random.randint(subkey1, shape=(), minval=1, maxval=7)   # Include manager (1-6)
        )
    )
    
    # Number of entities 
    num_entities = jax.lax.cond(
        force_escape_vehicle,
        lambda: jnp.array(1),
        lambda: jax.lax.cond(
            entity_type == ENTITY_MANAGER,
            lambda: jnp.array(1),  # Only 1 manager per lane
            lambda: jax.random.randint(subkey2, shape=(), minval=0, maxval=MAX_ENTITIES_PER_LANE + 1)
        )
    )
    
    # Random Y position variation for the lane
    y_variation = jax.random.randint(subkey5, shape=(), minval=-30, maxval=10)
    lane_y = -(LANE_HEIGHT + 20 + y_variation)

    # Generate X positions
    x_positions = jax.random.randint(
        subkey3, shape=(MAX_ENTITIES_PER_LANE,), 
        minval=10, maxval=WIDTH - 50
    )
    
    # Sort and space out X positions
    sorted_x = jnp.sort(x_positions)
    min_spacing = 20
    
    def apply_spacing(carry, x):
        prev_x = carry
        new_x = jnp.maximum(x, prev_x + min_spacing)
        return new_x, new_x
    
    _, spaced_x = jax.lax.scan(apply_spacing, sorted_x[0], sorted_x[1:])
    final_x = jnp.concatenate([sorted_x[:1], spaced_x])
    
    # Shuffle positions
    rng_key, shuffle_key = jax.random.split(rng_key)
    shuffle_indices = jax.random.permutation(shuffle_key, MAX_ENTITIES_PER_LANE)
    final_x_positions = final_x[shuffle_indices]
    
    # For escape vehicle, center it in the lane
    escape_vehicle_x = jnp.array([WIDTH // 2 - ESCAPE_VEHICLE_SIZE[0] // 2, 0, 0, 0])
    final_x_positions = jax.lax.cond(
        force_escape_vehicle,
        lambda: escape_vehicle_x,
        lambda: jnp.clip(final_x_positions, 5, WIDTH - 45)
    )

    # Create entity array
    entities = jnp.zeros((MAX_ENTITIES_PER_LANE, 3))
    entities = entities.at[:, 0].set(final_x_positions)
    entities = entities.at[:, 1].set(lane_y)
    
    # Set entity types 
    entities = jax.lax.cond(
        entity_type == ENTITY_MANAGER,
        lambda: entities.at[0, 2].set(entity_type),  # Only set first entity as manager
        lambda: entities.at[:, 2].set(entity_type)   # Set all entities to the same type
    )
    
    # Generate velocity 
    velocity_type = jax.lax.cond(
        force_escape_vehicle,
        lambda: jnp.array(0),  # Straight down for escape vehicle
        lambda: jax.random.randint(subkey4, shape=(), minval=0, maxval=5)
    )
    
    velocity = jax.lax.switch(
        velocity_type,
        [
            lambda: jnp.array([0.0, ENTITY_SPEED]),
            lambda: jnp.array([-BORDER_BOUNCE_SPEED, ENTITY_SPEED]),
            lambda: jnp.array([BORDER_BOUNCE_SPEED, ENTITY_SPEED]),
            lambda: jnp.array([-BORDER_BOUNCE_SPEED * 1.5, ENTITY_SPEED * 0.8]),
            lambda: jnp.array([BORDER_BOUNCE_SPEED * 1.5, ENTITY_SPEED * 0.8])
        ]
    )
    
    spawned_manager = jnp.any(entity_type == ENTITY_MANAGER)
    
    return rng_key, entities, velocity, num_entities, spawned_manager



# Environment
class JaxJourneyEscape(JaxEnvironment[JourneyEscapeState, JourneyEscapeObservation, JourneyEscapeInfo, None]):
    def __init__(self):
        super().__init__()
        self.action_set = {
            NOOP, UP, DOWN, LEFT, RIGHT, UPRIGHT, UPLEFT, DOWNRIGHT, DOWNLEFT,
            RIGHTFIRE, LEFTFIRE, DOWNFIRE, UPRIGHTFIRE, UPLEFTFIRE, DOWNRIGHTFIRE, DOWNLEFTFIRE
        }
        self.reset_counter = 0  
    
    @partial(jax.jit, static_argnums=(0,))
    def reset(self) -> Tuple[JourneyEscapeState, JourneyEscapeObservation]:
        
        self.reset_counter += 1
        
    
        base_time = int(time.time() * 1000000) % (2**32)
        random_offset = random.randint(0, 1000000)
        reset_component = self.reset_counter * 3571
        final_seed = (base_time + random_offset + reset_component) % (2**32)
        
        rng_key = jax.random.PRNGKey(final_seed)
        
        # Initialize empty lanes
        lane_entities = jnp.zeros((NUM_LANES, MAX_ENTITIES_PER_LANE, 3))
        lane_velocities = jnp.zeros((NUM_LANES, 2))
        lane_active = jnp.zeros(NUM_LANES, dtype=bool)
        lane_entity_counts = jnp.zeros(NUM_LANES, dtype=jnp.int32)
        
        # Spawn initial lanes (no escape vehicle initially)
        rng_key, entities_1, velocity_1, count_1, manager_1 = spawn_lane_entities(rng_key, 0, self.reset_counter, False, False)
        rng_key, entities_2, velocity_2, count_2, manager_2 = spawn_lane_entities(rng_key, 1, self.reset_counter, False, jnp.logical_or(manager_1, False))

        # Set up first two lanes
        lane_entities = lane_entities.at[0].set(entities_1)
        lane_entities = lane_entities.at[1].set(entities_2)
        lane_velocities = lane_velocities.at[0].set(velocity_1)
        lane_velocities = lane_velocities.at[1].set(velocity_2)
        lane_active = lane_active.at[0].set(True)
        lane_active = lane_active.at[1].set(True)
        lane_entity_counts = lane_entity_counts.at[0].set(count_1)
        lane_entity_counts = lane_entity_counts.at[1].set(count_2)
        
        # Calculate random times for escape vehicle appearances
        rng_key, escape_time_key = jax.random.split(rng_key)
        first_escape_time = jax.random.randint(
            escape_time_key, shape=(), 
            minval=30, maxval=ESCAPE_VEHICLE_MIN_TIME  
        )

        state = JourneyEscapeState(
            player_x=jnp.array(BAND_MEMBER_START_X, dtype=jnp.float32),
            player_y=jnp.array(BAND_MEMBER_START_Y, dtype=jnp.float32),
            time_left=jnp.array(INIT_TIME, dtype=jnp.float32),
            cash_left=jnp.array(INIT_CASH, dtype=jnp.float32),
            lane_entities=lane_entities,
            lane_velocities=lane_velocities,
            lane_active=lane_active,
            lane_entity_counts=lane_entity_counts,
            escape_vehicles_spawned=jnp.array(0, dtype=jnp.int32),
            next_escape_spawn_time=jnp.array(first_escape_time, dtype=jnp.float32),
            player_drag_velocity=jnp.array(0.0, dtype=jnp.float32),
            drag_timer=jnp.array(0, dtype=jnp.int32),
            photographer_visibility=jnp.array(True, dtype=bool),  # Start visible
            photographer_timer=jnp.array(0, dtype=jnp.int32),
            invulnerability_timer=jnp.array(0, dtype=jnp.int32),
            manager_protection=jnp.array(False, dtype=bool),
            manager_spawned=jnp.logical_or(manager_1, manager_2),
            last_groupie_collision=jnp.array(False, dtype=bool),
            last_promoter_collision=jnp.array(False, dtype=bool),
            last_photographer_collision=jnp.array(False, dtype=bool),
            spawn_timer=jnp.array(SPAWN_TIMER, dtype=jnp.int32),
            game_over=jnp.array(False),
            reached_goal=jnp.array(False),
            step_counter=jnp.array(0, dtype=jnp.int32),
            rng_key=rng_key,
        )
        
        obs = self._get_observation(state)
        return state, obs


    @partial(jax.jit, static_argnums=(0,))
    def step(
        self,
        state: JourneyEscapeState,
        action: chex.Array
    ) -> Tuple[JourneyEscapeState, JourneyEscapeObservation, float, bool, JourneyEscapeInfo]:
        
        # Update player position
        player_x = jax.lax.cond(action == LEFT, lambda x: x - 5, lambda x: x, state.player_x)
        player_x = jax.lax.cond(action == RIGHT, lambda x: x + 5, lambda x: x, player_x)
        player_y = jax.lax.cond(action == UP, lambda y: y - 5, lambda y: y, state.player_y)
        player_y = jax.lax.cond(action == DOWN, lambda y: y + 5, lambda y: y, player_y)
        
        # Handle diagonal movement
        player_x = jax.lax.cond(action == UPRIGHT, lambda x: x + 3, lambda x: x, player_x)
        player_y = jax.lax.cond(action == UPRIGHT, lambda y: y - 3, lambda y: y, player_y)
        player_x = jax.lax.cond(action == UPLEFT, lambda x: x - 3, lambda x: x, player_x)
        player_y = jax.lax.cond(action == UPLEFT, lambda y: y - 3, lambda y: y, player_y)
        player_x = jax.lax.cond(action == DOWNRIGHT, lambda x: x + 3, lambda x: x, player_x)
        player_y = jax.lax.cond(action == DOWNRIGHT, lambda y: y + 3, lambda y: y, player_y)
        player_x = jax.lax.cond(action == DOWNLEFT, lambda x: x - 3, lambda x: x, player_x)
        player_y = jax.lax.cond(action == DOWNLEFT, lambda y: y + 3, lambda y: y, player_y)
        
        # Apply drag effect if active
        player_y = jax.lax.cond(
            state.drag_timer > 0,
            lambda y: y + state.player_drag_velocity,
            lambda y: y,
            player_y
        )
        
        # Clamp player position
        player_x = jnp.clip(player_x, 0, WIDTH - BAND_MEMBER_SIZE[0])
        player_y = jnp.clip(player_y, 0, HEIGHT - BAND_MEMBER_SIZE[1])
        
        # Update photographer visibility cycle
        photographer_timer = (state.photographer_timer + 1) % (PHOTOGRAPHER_CYCLE_TIME * 2)
        photographer_visibility = photographer_timer < PHOTOGRAPHER_CYCLE_TIME
        
        # Update timers
        drag_timer = jnp.maximum(0, state.drag_timer - 1)
        invulnerability_timer = jnp.maximum(0, state.invulnerability_timer - 1)
        
        # Reset drag velocity when timer expires
        player_drag_velocity = jax.lax.cond(
            drag_timer > 0,
            lambda: state.player_drag_velocity,
            lambda: jnp.array(0.0)
        )
        
        def update_lane_positions_with_bouncing(lane_entities, lane_velocities):
            
            # Add velocity to all entity positions
            updated_positions = lane_entities[:, :, :2] + lane_velocities[:, None, :]
            
            # Get entity sizes for proper collision detection - FIXED: Include escape vehicle
            entity_types = lane_entities[:, :, 2].astype(jnp.int32)
            entity_sizes = jnp.array([
                [0, 0],         # ENTITY_NONE
                [10, 10],       # ENTITY_GROUPIE
                [10, 10],       # ENTITY_PROMOTER
                [10, 10],       # ENTITY_PHOTOGRAPHER
                [40, 15],       # ENTITY_BARRIER
                [10, 10],       # ENTITY_ROADIE
                [10, 10],       # ENTITY_MANAGER
                [25, 20],       # ENTITY_ESCAPE_VEHICLE
            ])
            all_entity_sizes = entity_sizes[entity_types]
            
            # Check for border collisions per individual entity
            left_collision = updated_positions[:, :, 0] < 0

            right_collision = updated_positions[:, :, 0] + all_entity_sizes[:, :, 0] > WIDTH
            
            # Handle left border bouncing per entity
            updated_positions = jnp.where(
                left_collision[:, :, None],
                jnp.stack([
                    jnp.zeros_like(updated_positions[:, :, 0]),  # Set X to 0
                    updated_positions[:, :, 1]  # Keep Y
                ], axis=2),
                updated_positions
            )
            
            # Handle right border bouncing per entity
            right_bounce_x = WIDTH - all_entity_sizes[:, :, 0]
            updated_positions = jnp.where(
                right_collision[:, :, None],
                jnp.stack([
                    right_bounce_x,  # Set X to right edge minus entity width
                    updated_positions[:, :, 1]  # Keep Y
                ], axis=2),
                updated_positions
            )
            
            # Update velocities PER LANE based on any collision in that lane
            lane_left_collision = jnp.any(left_collision, axis=1) 
            lane_right_collision = jnp.any(right_collision, axis=1) 
            
            # Create new velocities
            new_velocities = lane_velocities.copy()
            
            # For lanes with left collision: make X velocity positive
            new_velocities = jnp.where(
                lane_left_collision[:, None],
                jnp.stack([
                    jnp.abs(lane_velocities[:, 0]),  # Make X velocity positive
                    lane_velocities[:, 1]  # Keep Y velocity
                ], axis=1),
                new_velocities
            )
            
            # For lanes with right collision: make X velocity negative
            new_velocities = jnp.where(
                lane_right_collision[:, None],
                jnp.stack([
                    -jnp.abs(lane_velocities[:, 0]),  # Make X velocity negative
                    lane_velocities[:, 1]  # Keep Y velocity
                ], axis=1),
                new_velocities
            )
            
            # Update entity positions
            updated_entities = lane_entities.at[:, :, :2].set(updated_positions)
            
            return updated_entities, new_velocities
        
        new_lane_entities, new_lane_velocities = update_lane_positions_with_bouncing(
            state.lane_entities, state.lane_velocities
        )
        
    
        def check_lanes_active_jax(lane_entities, lane_active, lane_entity_counts):
            # Get Y positions for all entities in all lanes
            all_y_positions = lane_entities[:, :, 1]
            
            # Check which entities are on screen
            entities_on_screen = all_y_positions <= HEIGHT + 50
            
            # Create mask for valid entities
            entity_indices = jnp.arange(MAX_ENTITIES_PER_LANE)[None, :]
            lane_counts = lane_entity_counts[:, None]
            valid_entity_mask = entity_indices < lane_counts
            
            # Only consider entities that are valid AND on screen
            valid_entities_on_screen = jnp.logical_and(entities_on_screen, valid_entity_mask)
            
            # Check if each lane has any valid entities on screen
            lane_has_entities_on_screen = jnp.any(valid_entities_on_screen, axis=1)
            
            # Lane stays active if it was active AND has entities on screen
            new_active = jnp.logical_and(lane_active, lane_has_entities_on_screen)
            
            return new_active
        
        new_lane_active = check_lanes_active_jax(new_lane_entities, state.lane_active, state.lane_entity_counts)
        
        # Check if it's time to spawn an escape vehicle
        should_spawn_escape = jnp.logical_and(
            jnp.logical_and(
                state.time_left <= state.next_escape_spawn_time,
                state.escape_vehicles_spawned < ESCAPE_VEHICLE_APPEARANCES
            ),
            jnp.logical_not(jnp.any(new_lane_entities[:, :, 2] == ENTITY_ESCAPE_VEHICLE))
        )
        
        # Regular lane spawning
        spawn_timer = jnp.maximum(0, state.spawn_timer - 1)
        should_spawn_regular = jnp.logical_and(spawn_timer <= 0, jnp.logical_not(should_spawn_escape))
        
        def spawn_escape_vehicle():
            # Find first inactive lane
            inactive_lane_idx = jnp.argmax(jnp.logical_not(new_lane_active))
            
            # Spawn escape vehicle
            new_rng, new_entities, new_velocity, new_count, spawned_manager = spawn_lane_entities(
                state.rng_key, inactive_lane_idx, self.reset_counter, True, state.manager_spawned  # Force escape vehicle
            )

            # Update lane data
            updated_entities = new_lane_entities.at[inactive_lane_idx].set(new_entities)
            updated_velocities = new_lane_velocities.at[inactive_lane_idx].set(new_velocity)
            updated_active = new_lane_active.at[inactive_lane_idx].set(True)
            updated_counts = state.lane_entity_counts.at[inactive_lane_idx].set(new_count)
            
            # Calculate next escape spawn time
            new_rng, time_key = jax.random.split(new_rng)
            next_spawn_time = jax.lax.cond(
                state.escape_vehicles_spawned < ESCAPE_VEHICLE_APPEARANCES - 1,
                lambda: jax.random.randint(time_key, shape=(), minval=10, maxval=25).astype(jnp.float32),
                lambda: jnp.array(-1.0, dtype=jnp.float32)
            )
            
            return (updated_entities, updated_velocities, updated_active, updated_counts, new_rng, 
                    state.spawn_timer, state.escape_vehicles_spawned + 1, next_spawn_time, 
                    jnp.logical_or(state.manager_spawned, spawned_manager))

        def spawn_regular_lane():
            # Find first inactive lane
            inactive_lane_idx = jnp.argmax(jnp.logical_not(new_lane_active))                
                
            # Spawn regular entities
            new_rng, new_entities, new_velocity, new_count, spawned_manager = spawn_lane_entities(
                state.rng_key, inactive_lane_idx, self.reset_counter, False, state.manager_spawned
            )
        # Update lane data
            updated_entities = new_lane_entities.at[inactive_lane_idx].set(new_entities)
            updated_velocities = new_lane_velocities.at[inactive_lane_idx].set(new_velocity)
            updated_active = new_lane_active.at[inactive_lane_idx].set(True)
            updated_counts = state.lane_entity_counts.at[inactive_lane_idx].set(new_count)

            return (updated_entities, updated_velocities, updated_active, updated_counts, new_rng,
                    SPAWN_TIMER, state.escape_vehicles_spawned, state.next_escape_spawn_time,
                    jnp.logical_or(state.manager_spawned, spawned_manager))
                    
        def keep_current_lanes():
            return (new_lane_entities, new_lane_velocities, new_lane_active, state.lane_entity_counts, 
                    state.rng_key, spawn_timer, state.escape_vehicles_spawned, state.next_escape_spawn_time,
                    state.manager_spawned)

        (lane_entities, lane_velocities, lane_active, lane_entity_counts, rng_key, 
        spawn_timer, escape_vehicles_spawned, next_escape_spawn_time, manager_spawned) = jax.lax.cond(
            should_spawn_escape,
            spawn_escape_vehicle,
            lambda: jax.lax.cond(should_spawn_regular, spawn_regular_lane, keep_current_lanes)
        )
        
        def check_collisions_with_effects(player_x, player_y, lane_entities, lane_active, lane_entity_counts, photographer_visibility):
            entity_sizes = jnp.array([
                [0, 0],         # ENTITY_NONE
                [10, 10],       # ENTITY_GROUPIE
                [10, 10],       # ENTITY_PROMOTER
                [10, 10],       # ENTITY_PHOTOGRAPHER
                [40, 15],       # ENTITY_BARRIER
                [10, 10],       # ENTITY_ROADIE
                [10, 10],       # ENTITY_MANAGER
                [25, 20],       # ENTITY_ESCAPE_VEHICLE
            ])
            
            entity_x = lane_entities[:, :, 0]
            entity_y = lane_entities[:, :, 1]
            entity_types = lane_entities[:, :, 2].astype(jnp.int32)
            
            all_entity_sizes = entity_sizes[entity_types]
            
            collision_checks = jnp.logical_and(
                jnp.logical_and(
                    player_x < entity_x + all_entity_sizes[:, :, 0],
                    player_x + BAND_MEMBER_SIZE[0] > entity_x
                ),
                jnp.logical_and(
                    player_y < entity_y + all_entity_sizes[:, :, 1],
                    player_y + BAND_MEMBER_SIZE[1] > entity_y
                )
            )
            
            entity_indices = jnp.arange(MAX_ENTITIES_PER_LANE)[None, :]
            lane_counts = lane_entity_counts[:, None]
            valid_entity_mask = entity_indices < lane_counts
            lane_active_mask = lane_active[:, None]
            
            # For photographers, only count collision if they're visible
            photographer_mask = jnp.where(
                entity_types == ENTITY_PHOTOGRAPHER,
                photographer_visibility,
                True
            )
            
            valid_collisions = collision_checks & valid_entity_mask & lane_active_mask & photographer_mask
            
            # Check specific entity collisions
            groupie_collision = jnp.any(valid_collisions & (entity_types == ENTITY_GROUPIE))
            promoter_collision = jnp.any(valid_collisions & (entity_types == ENTITY_PROMOTER))
            photographer_collision = jnp.any(valid_collisions & (entity_types == ENTITY_PHOTOGRAPHER))
            barrier_collision = jnp.any(valid_collisions & (entity_types == ENTITY_BARRIER))
            roadie_collision = jnp.any(valid_collisions & (entity_types == ENTITY_ROADIE))
            manager_collision = jnp.any(valid_collisions & (entity_types == ENTITY_MANAGER))
            escape_collision = jnp.any(valid_collisions & (entity_types == ENTITY_ESCAPE_VEHICLE))
            
            return (groupie_collision, promoter_collision, photographer_collision, barrier_collision,
                    roadie_collision, manager_collision, escape_collision)
        
        (groupie_collision, promoter_collision, photographer_collision, barrier_collision,
        roadie_collision, manager_collision, escape_collision) = check_collisions_with_effects(
            player_x, player_y, lane_entities, lane_active, lane_entity_counts, photographer_visibility
        )
        
        # Determine if player has protection
        has_protection = jnp.logical_or(invulnerability_timer > 0, state.manager_protection)
        
        # Apply collision effects only if not protected
        def apply_collision_effects():
            # Check for new collisions (not continuing from last frame)
            new_groupie_collision = jnp.logical_and(groupie_collision, jnp.logical_not(state.last_groupie_collision))
            new_promoter_collision = jnp.logical_and(promoter_collision, jnp.logical_not(state.last_promoter_collision))
            new_photographer_collision = jnp.logical_and(photographer_collision, jnp.logical_not(state.last_photographer_collision))
            
            # Calculate cash changes - only deduct on NEW collisions
            new_cash = state.cash_left
            new_cash = jax.lax.cond(new_groupie_collision, lambda c: jnp.maximum(0, c - 300), lambda c: c, new_cash)
            new_cash = jax.lax.cond(new_promoter_collision, lambda c: jnp.maximum(0, c - 2000), lambda c: c, new_cash)
            new_cash = jax.lax.cond(new_photographer_collision, lambda c: jnp.maximum(0, c - 600), lambda c: c, new_cash)
            
            # Set drag effects (groupies and barriers)
            new_drag_timer = jax.lax.cond(
                jnp.logical_or(groupie_collision, barrier_collision),
                lambda: jnp.array(DRAG_EFFECT_DURATION),
                lambda: drag_timer
            )
            
            new_drag_velocity = jax.lax.cond(
                jnp.logical_or(groupie_collision, barrier_collision),
                lambda: jnp.array(DRAG_VELOCITY),
                lambda: player_drag_velocity
            )
            
            # Set invulnerability from roadie
            new_invulnerability = jax.lax.cond(
                roadie_collision,
                lambda: jnp.array(ROADIE_PROTECTION_TIME),
                lambda: invulnerability_timer
            )
            
            # Set manager protection
            new_manager_protection = jax.lax.cond(
                manager_collision,
                lambda: jnp.array(True),
                lambda: state.manager_protection
            )
            
            # Any enemy collision (for game over, only if not protected)
            enemy_collision = jnp.logical_or(
                jnp.logical_or(groupie_collision, promoter_collision),
                jnp.logical_or(photographer_collision, barrier_collision)
            )
            
            return new_cash, new_drag_timer, new_drag_velocity, new_invulnerability, new_manager_protection, enemy_collision
        
        def no_collision_effects():
            return state.cash_left, drag_timer, player_drag_velocity, invulnerability_timer, state.manager_protection, jnp.array(False)
        
        # Apply effects only if player doesn't have protection for enemy collisions
        (cash_left, drag_timer, player_drag_velocity, invulnerability_timer, 
        manager_protection, enemy_collision) = jax.lax.cond(
            has_protection,
            no_collision_effects,
            apply_collision_effects
        )
        
        # Roadie and manager effects - only protection, no cash rewards
        invulnerability_timer = jax.lax.cond(
            roadie_collision,
            lambda: jnp.array(ROADIE_PROTECTION_TIME),
            lambda: invulnerability_timer
        )
        
        manager_protection = jax.lax.cond(
            manager_collision,
            lambda: jnp.array(True),
            lambda: manager_protection
        )
        
        # Calculate reward No rewards for collisions
        reward = 0.0
        #reward = jax.lax.cond(roadie_collision, lambda r: r + 100, lambda r: r, reward)
        #reward = jax.lax.cond(manager_collision, lambda r: r + 500, lambda r: r, reward)
        #reward = jax.lax.cond(escape_collision, lambda r: r + 1000, lambda r: r, reward)
        #reward = jax.lax.cond(jnp.logical_and(enemy_collision, jnp.logical_not(has_protection)), lambda r: r - 500, lambda r: r, reward)
        
        # Update time
        time_left = jax.lax.cond(
            state.step_counter % 60 == 0,
            lambda t: jnp.maximum(0, t - 1),
            lambda t: t,
            state.time_left
        )

        # Game over conditions
        game_over = jnp.logical_or(cash_left <= 0, time_left <= 0)

        reached_goal = escape_collision

        new_state = state._replace(
            player_x=player_x,
            player_y=player_y,
            time_left=time_left,
            cash_left=cash_left,
            lane_entities=lane_entities,
            lane_velocities=lane_velocities,
            lane_active=lane_active,
            lane_entity_counts=lane_entity_counts,
            escape_vehicles_spawned=escape_vehicles_spawned,
            next_escape_spawn_time=next_escape_spawn_time,
            player_drag_velocity=player_drag_velocity,
            drag_timer=drag_timer,
            photographer_visibility=photographer_visibility,
            photographer_timer=photographer_timer,
            invulnerability_timer=invulnerability_timer,
            manager_protection=manager_protection,
            manager_spawned=manager_spawned,
            last_groupie_collision=groupie_collision,
            last_promoter_collision=promoter_collision,
            last_photographer_collision=photographer_collision,
            spawn_timer=spawn_timer,
            game_over=game_over,
            reached_goal=reached_goal,
            step_counter=state.step_counter + 1,
            rng_key=rng_key
        )
        
        obs = self._get_observation(new_state)
        done = jnp.logical_or(game_over, reached_goal)
        info = JourneyEscapeInfo(time=time_left, cash=cash_left)
        
        return new_state, obs, reward, done, info


    # Keep the _get_observation method the same
    def _get_observation(self, state: JourneyEscapeState) -> JourneyEscapeObservation:
        player = jnp.array([state.player_x, state.player_y])
        entities = state.lane_entities.flatten()
        return JourneyEscapeObservation(
            player=player,
            entities=entities,
            time=state.time_left,
            cash=state.cash_left,
        )

def load_journey_escape_sprites():
    """Load all sprites for the Journey Escape game from PNG files."""
    
    sprites = {}
    
    sprite_files = {
        'band_member': 'band_member_still.png',
        'band_member_left': 'band_member_left.png', 
        'band_member_right': 'band_member_right.png',
        'groupie': 'groupie.png',
        'promoter': 'promoter.png',
        'photographer': 'photographer.png',
        'barrier': 'barrier.png',
        'roadie': 'rodie.png',  # Note: filename is 'rodie.png'
        'manager': 'manager.png',
        'escape_vehicle': 'vehicle.png',
    }
    
    # Load each sprite
    for sprite_name, filename in sprite_files.items():
        sprite_path = os.path.join("src", "jaxatari", "games", "sprites", "journeyescape", filename)
        try:
            sprite = pygame.image.load(sprite_path).convert_alpha()
            sprites[sprite_name] = sprite
        except pygame.error as e:
            print(f"Warning: Could not load sprite {filename}: {e}")
            # Fallback to colored rectangle if sprite fails to load
            if sprite_name == 'band_member' or 'band_member' in sprite_name:
                size, color = BAND_MEMBER_SIZE, BAND_MEMBER_COLOR
            else:
                size, color = get_entity_info_by_name(sprite_name)
            
            fallback_surface = pygame.Surface(size)
            fallback_surface.fill(color)
            sprites[sprite_name] = fallback_surface
    
    # Scale sprites up for better visibility (similar to Frogger approach)
    scale_factor = 4
    for sprite_name in sprites:
        original_size = sprites[sprite_name].get_size()
        new_size = (original_size[0] * scale_factor, original_size[1] * scale_factor)
        sprites[sprite_name] = pygame.transform.scale(sprites[sprite_name], new_size)
    
    return sprites

def get_entity_info_by_name(sprite_name):
    """Helper function to get entity info for fallback sprites."""
    info_map = {
        'groupie': (GROUPIE_SIZE, GROUPIE_COLOR),
        'promoter': (PROMOTER_SIZE, PROMOTER_COLOR),
        'photographer': (PHOTOGRAPHER_SIZE, PHOTOGRAPHER_COLOR),
        'barrier': (BARRIER_SIZE, BARRIER_COLOR),
        'roadie': (ROADIE_SIZE, ROADIE_COLOR),
        'manager': (MANAGER_SIZE, MANAGER_COLOR),
        'escape_vehicle': (ESCAPE_VEHICLE_SIZE, ESCAPE_VEHICLE_COLOR),
    }
    return info_map.get(sprite_name, ((10, 10), (255, 255, 255)))

def get_entity_sprite(entity_type, sprites):
    """Get the appropriate sprite for an entity type."""
    sprite_map = {
        ENTITY_GROUPIE: 'groupie',
        ENTITY_PROMOTER: 'promoter', 
        ENTITY_PHOTOGRAPHER: 'photographer',
        ENTITY_BARRIER: 'barrier',
        ENTITY_ROADIE: 'roadie',
        ENTITY_MANAGER: 'manager',
        ENTITY_ESCAPE_VEHICLE: 'escape_vehicle'
    }
    sprite_name = sprite_map.get(entity_type, 'groupie')  # Default to groupie if unknown
    return sprites[sprite_name]

def run_game():
    pygame.init()
    screen = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT))
    pygame.display.set_caption("Journey Escape - Enhanced Effects")
    clock = pygame.time.Clock()
    
    # Load sprites
    sprites = load_journey_escape_sprites()
    
    env = JaxJourneyEscape()
    state, obs = env.reset()
        
    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_r and (state.game_over or state.reached_goal):
                    state, obs = env.reset()
        
        if not state.game_over and not state.reached_goal:
            action = get_human_action()
            state, obs, reward, done, info = env.step(state, action)
        
        # Clear screen
        screen.fill(BACKGROUND_COLOR)
        
        # Draw lane entities
        for lane_idx in range(NUM_LANES):
            if state.lane_active[lane_idx]:
                for entity_idx in range(state.lane_entity_counts[lane_idx]):
                    entity = state.lane_entities[lane_idx, entity_idx]
                    entity_x, entity_y, entity_type = entity[0], entity[1], entity[2]
                    
                    # Skip drawing photographers when they're invisible
                    if int(entity_type) == ENTITY_PHOTOGRAPHER and not state.photographer_visibility:
                        continue
                    
                    # Only draw if entity is on screen
                    entity_size, entity_color = get_entity_info(int(entity_type))
                    if -entity_size[1] <= entity_y <= HEIGHT + entity_size[1]:
                        x, y = int(entity_x * 3), int(entity_y * 3)
                        
                        # Use sprite instead of rectangle
                        entity_sprite = get_entity_sprite(int(entity_type), sprites)
                        screen.blit(entity_sprite, (x, y))
        
        # Draw player with protection indication
        px, py = int(state.player_x * 3), int(state.player_y * 3)
        
        # Get base player sprite
        player_sprite = sprites['band_member'].copy()
        
        # Change player sprite tint if protected
        if state.invulnerability_timer > 0:
            # Tint white for roadie protection
            player_sprite.fill((255, 255, 255, 128), special_flags=pygame.BLEND_RGBA_MULT)
        elif state.manager_protection:
            # Tint gold for manager protection  
            player_sprite.fill((255, 215, 0, 128), special_flags=pygame.BLEND_RGBA_MULT)
        
        screen.blit(player_sprite, (px, py))
        
        # Draw UI
        font = pygame.font.Font(None, 36)
        small_font = pygame.font.Font(None, 24)
        
        time_text = font.render(f"Time: {int(state.time_left)}", True, (255, 255, 255))
        cash_text = font.render(f"Cash: ${int(state.cash_left)}", True, (255, 255, 255))
        screen.blit(time_text, (10, 10))
        screen.blit(cash_text, (10, 50))
        
        # Protection status
        if state.invulnerability_timer > 0:
            protection_text = small_font.render(f"Roadie Protection: {state.invulnerability_timer // 60 + 1}s", True, (255, 255, 255))
            screen.blit(protection_text, (10, 90))
        elif state.manager_protection:
            protection_text = small_font.render("Manager Protection: ACTIVE", True, (255, 215, 0))
            screen.blit(protection_text, (10, 90))
        
        # Drag effect status
        if state.drag_timer > 0:
            drag_text = small_font.render(f"Dragged: {state.drag_timer // 60 + 1}s", True, (255, 100, 100))
            screen.blit(drag_text, (10, 115))
        
        # Game over/win messages
        if state.game_over:
            game_over_text = font.render("GAME OVER! Press R to restart", True, (255, 0, 0))
            screen.blit(game_over_text, (WINDOW_WIDTH // 2 - 200, WINDOW_HEIGHT // 2))
        elif state.reached_goal:
            win_text = font.render("YOU ESCAPED! Press R to restart", True, (0, 255, 0))
            screen.blit(win_text, (WINDOW_WIDTH // 2 - 200, WINDOW_HEIGHT // 2))
        
        pygame.display.flip()
        clock.tick(60)
    
    pygame.quit()

if __name__ == "__main__":
    run_game()
