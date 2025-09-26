# Gruppemittglieder:
# - Matias Heredia
# - Santiago Ramirez

import jax
import numpy as np
import pygame
import jaxatari.spaces as spaces
import sys
import os
from functools import partial

import jax.numpy as jnp

# from jaxatari.rendering import atraJaxis as aj
from jaxatari.environment import JaxEnvironment

GRID_SIZE = 4
WINDOW_SIZE = (400, 600)
CELL_SIZE = WINDOW_SIZE[0] // GRID_SIZE
COLORS = {
    "bg": (30, 30, 30),
    "grid": (200, 200, 200),
    "X": (255, 80, 80),
    "O": (80, 80, 255),
    "highlight": (255, 255, 0),
}

ACTION_MEANINGS = [
    "NOOP", "FIRE", "UP", "RIGHT", "LEFT", "DOWN",
    "UPRIGHT", "UPLEFT", "DOWNRIGHT", "DOWNLEFT"
]

class JaxTicTacToe3D(JaxEnvironment):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 10}

    def __init__(self, obs_type="rgb", full_action_space=False):
        super().__init__()
        self.obs_type = obs_type
        self.full_action_space = full_action_space
        self._action_space = spaces.Discrete(18 if full_action_space else 10)
        if obs_type == "rgb":
            self._observation_space = spaces.Box(0, 255, (210, 160, 3), dtype=np.uint8)
        elif obs_type == "ram":
            self._observation_space = spaces.Box(0, 255, (128,), dtype=np.uint8)
        elif obs_type == "grayscale":
            self._observation_space = spaces.Box(0, 255, (210, 160), dtype=np.uint8)
        else:
            raise ValueError("Unknown obs_type")
        self.window = None
        self.clock = None
        self.cursor_circle = [0, 0, 0]  # Arrow keys, draws circle, places O
        self.cursor_x = [0, 0, 0]       # WASD keys, draws X, places X
        self.turn = 2  # 2 for O (circle), 1 for X
        self.game_mode = "pvp"  # "pvp" or "pve"
        self.winner = None
        self.reset()

    def observation_space(self):
        return self._observation_space

    def action_space(self):
        return self._action_space

    # Add image_space and render stubs for wrapper compatibility
    def image_space(self):
        # Return a Box matching the expected image shape
        return spaces.Box(0, 255, (210, 160, 3), dtype=np.uint8)

    def render(self, state=None):
        # Return a blank image for compatibility
        return jnp.zeros((210, 160, 3), dtype=jnp.uint8)

    def reset(self, key=None):
        self.board = jnp.zeros((GRID_SIZE, GRID_SIZE, GRID_SIZE), dtype=jnp.int8)
        self.current_player = 1
        self.done = False
        self.last_action = 0
        self.cursor = [0, 0, 0]
        self.cursor_circle = [0, 0, 0]
        self.cursor_x = [0, 0, 0]
        self.turn = 2
        self.winner = None
        return self._get_obs(), self.get_state()

    def step(self, state, action):
        if self.done:
            return self._get_obs(), state, 0.0, True, {}

        self.last_action = action
        reward = 0.0

        # Move cursor or place mark
        if action == 0:  # NOOP
            pass
        elif action == 1:  # FIRE (place mark)
            x, y, z = self.cursor
            if self.board[x, y, z] == 0:
                self.board = self.board.at[x, y, z].set(self.current_player)
                if self._check_win(self.current_player):
                    reward = 1.0
                    self.done = True
                    self.winner = "X" if self.current_player == 1 else "O"
                elif jnp.all(self.board != 0):
                    self.done = True  # Draw
                else:
                    self.current_player = 3 - self.current_player
            else:
                reward = -0.1  # Invalid move
        else:
            self._move_cursor(action)

        return self._get_obs(), self.get_state(), reward, self.done, {}

    def get_state(self):
        # Return a simple state representation (can be expanded for logging/wrappers)
        return {
            "board": self.board,
            "current_player": self.current_player,
            "done": self.done,
            "last_action": self.last_action,
            "cursor": self.cursor,
            "winner": self.winner
        }

    @property
    def observation_space(self):
        return self._observation_space

    @observation_space.setter
    def observation_space(self, value):
        self._observation_space = value

    @property
    def action_space(self):
        return self._action_space

    @action_space.setter
    def action_space(self, value):
        self._action_space = value

    def reset(self, key=None):
        self.board = jnp.zeros((GRID_SIZE, GRID_SIZE, GRID_SIZE), dtype=jnp.int8)
        self.current_player = 1
        self.done = False
        self.last_action = 0
        self.cursor = [0, 0, 0]
        self.cursor_circle = [0, 0, 0]
        self.cursor_x = [0, 0, 0]
        self.turn = 2
        self.winner = None
        return self._get_obs(), {}

    @partial(jax.jit, static_argnums=(0,))
    def step(self, state, action):
        # For compatibility, state is ignored, using self attributes
        if self.done:
            return self._get_obs(), 0.0, True, False, {}

        self.last_action = action
        reward = 0.0

        # Move cursor or place mark
        if action == 0:  # NOOP
            pass
        elif action == 1:  # FIRE (place mark)
            x, y, z = self.cursor
            if self.board[x, y, z] == 0:
                self.board = self.board.at[x, y, z].set(self.current_player)
                if self._check_win(self.current_player):
                    reward = 1.0
                    self.done = True
                    self.winner = "X" if self.current_player == 1 else "O"
                elif jnp.all(self.board != 0):
                    self.done = True  # Draw
                else:
                    self.current_player = 3 - self.current_player
            else:
                reward = -0.1  # Invalid move
        else:
            self._move_cursor(action)

        return self._get_obs(), reward, self.done, False, {}

    def reset(self, seed=None, options=None):
        self.board = jnp.zeros((GRID_SIZE, GRID_SIZE, GRID_SIZE), dtype=jnp.int8)
        self.current_player = 1
        self.done = False
        self.last_action = 0
        self.cursor = [0, 0, 0]
        self.cursor_circle = [0, 0, 0]
        self.cursor_x = [0, 0, 0]
        self.turn = 2
        self.winner = None
        return self._get_obs(), {}

    @partial(jax.jit, static_argnums=(0,))
    def step(self, action):
        if self.done:
            return self._get_obs(), 0.0, True, False, {}

        self.last_action = action
        reward = 0.0

        # Move cursor or place mark
        if action == 0:  # NOOP
            pass
        elif action == 1:  # FIRE (place mark)
            x, y, z = self.cursor
            if self.board[x, y, z] == 0:
                self.board = self.board.at[x, y, z].set(self.current_player)
                if self._check_win(self.current_player):
                    reward = 1.0
                    self.done = True
                    self.winner = "X" if self.current_player == 1 else "O"
                elif jnp.all(self.board != 0):
                    self.done = True  # Draw
                else:
                    self.current_player = 3 - self.current_player
            else:
                reward = -0.1  # Invalid move
        else:
            self._move_cursor(action)

        return self._get_obs(), reward, self.done, False, {}

    def _move_cursor(self, action):
        dx, dy, dz = 0, 0, 0
        if action == 2:  # UP
            dy = -1
        elif action == 3:  # RIGHT
            dx = 1
        elif action == 4:  # LEFT
            dx = -1
        elif action == 5:  # DOWN
            dy = 1
        elif action == 6:  # UPRIGHT
            dx, dy = 1, -1
        elif action == 7:  # UPLEFT
            dx, dy = -1, -1
        elif action == 8:  # DOWNRIGHT
            dx, dy = 1, 1
        elif action == 9:  # DOWNLEFT
            dx, dy = -1, 1
        # For full_action_space, add more dz moves if needed
        x, y, z = self.cursor
        x = np.clip(x + dx, 0, GRID_SIZE - 1)
        y = np.clip(y + dy, 0, GRID_SIZE - 1)
        self.cursor = [x, y, z]

    def _check_win(self, player):
        b = np.array(self.board)
        for x in range(GRID_SIZE):
            for y in range(GRID_SIZE):
                for z in range(GRID_SIZE):
                    if b[x, y, z] != player:
                        continue
                    # Check all 13 directions in 3D
                    for dx, dy, dz in [
                        (1, 0, 0), (0, 1, 0), (0, 0, 1),
                        (1, 1, 0), (1, 0, 1), (0, 1, 1),
                        (1, 1, 1), (1, -1, 0), (1, 0, -1),
                        (0, 1, -1), (1, -1, 1), (1, 1, -1), (1, -1, -1)
                    ]:
                        try:
                            if all(
                                0 <= x + i * dx < GRID_SIZE and
                                0 <= y + i * dy < GRID_SIZE and
                                0 <= z + i * dz < GRID_SIZE and
                                b[x + i * dx, y + i * dy, z + i * dz] == player
                                for i in range(4)
                            ):
                                return True
                        except IndexError:
                            continue
        return False

    def _get_obs(self):
        if self.obs_type == "rgb":
            return self._render_rgb()
        elif self.obs_type == "ram":
            ram = np.zeros(128, dtype=np.uint8)
            flat = np.array(self.board).flatten()
            ram[:flat.size] = flat
            return ram
        elif self.obs_type == "grayscale":
            rgb = self._render_rgb()
            return np.dot(rgb[..., :3], [0.299, 0.587, 0.114]).astype(np.uint8)
        else:
            raise ValueError("Unknown obs_type")

    def render(self, mode="human"):
        if self.obs_type != "rgb":
            raise NotImplementedError("Only rgb rendering supported")
        if self.window is None:
            pygame.init()
            self.window = pygame.display.set_mode(WINDOW_SIZE)
            self.clock = pygame.time.Clock()

        surf = pygame.Surface(WINDOW_SIZE)
        surf.fill((180, 255, 220))  # Background

        # 3D Layer projection parameters
        dx = 20  # right shift per layer
        dy = 120  # down shift per layer
        h_line_length = 150
        cell_w = h_line_length / 4
        cell_h = 20
        origin_x = 80
        origin_y = 40
        end_x = 300
        end_y = 300
        
        for z in range(GRID_SIZE):
            ox = origin_x
            oy = origin_y + z * dy
            ex = end_x + z * dx
            ey = origin_y + z * dy

            #Draw grid
            for x in range(GRID_SIZE + 1):
                start = (ox + x * cell_w, ey)
                end = (start[0] + 4 * dx, oy + GRID_SIZE * cell_h)
                pygame.draw.line(surf, (0, 80, 0), start, end, 1)

            for y in range(GRID_SIZE + 1):
                start = (ox + y * dx, oy + y * cell_h)
                end = (start[0] + h_line_length, start[1])
                pygame.draw.line(surf, (0, 80, 0), start, end, 1)

            # Draw marks and cursor
            for x in range(GRID_SIZE):
                for y in range(GRID_SIZE):
                    v = self.board[x, y, z]
                    # Calculate cell parallelogram top-left
                    px = ox + x * cell_w + y * dx
                    py = oy + y * cell_h

                    # Parallelogram corners for the cell
                    p1 = (px, py)
                    p2 = (px + cell_w, py)
                    p3 = (px + cell_w + dx, py + cell_h)
                    p4 = (px + dx, py + cell_h)

                    # Center for ellipse
                    ecx = int((p1[0] + p3[0]) / 2)
                    ecy = int((p1[1] + p3[1]) / 2)
                    ellipse_width = int((cell_w + dx) * 0.6)
                    ellipse_height = int(cell_h * 0.7)

                    # Draw X mark
                    if v == 1:
                        pygame.draw.line(surf, (0, 80, 0), p1, p3, 2)
                        pygame.draw.line(surf, (0, 80, 0), p2, p4, 2)
                    # Draw O mark
                    elif v == 2:
                        pygame.draw.ellipse(
                            surf, (0, 80, 0),
                            pygame.Rect(
                                ecx - ellipse_width // 2,
                                ecy - ellipse_height // 2,
                                ellipse_width,
                                ellipse_height
                            ),
                            2
                        )

                    # Draw X cursor
                    if self.game_mode == "pvp" and [x, y, z] == self.cursor_x:
                        if v == 1:
                            cursor_color = (100, 0, 0)  # Much darker red for overlap
                        else:
                            cursor_color = (255, 150, 150)  # Much lighter red for normal
                        pygame.draw.polygon(surf, cursor_color, [p1, p2, p3, p4], 2)
                        pygame.draw.line(surf, cursor_color, p1, p3, 2)
                        pygame.draw.line(surf, cursor_color, p2, p4, 2)

                    # Draw O cursor
                    if [x, y, z] == self.cursor_circle:
                        if v == 2:
                            cursor_color = (0, 0, 100)  # Much darker blue for overlap
                        else:
                            cursor_color = (150, 150, 255)  # Much lighter blue for normal
                        pygame.draw.polygon(surf, cursor_color, [p1, p2, p3, p4], 2)
                        pygame.draw.ellipse(
                            surf, cursor_color,
                            pygame.Rect(
                                ecx - ellipse_width // 2,
                                ecy - ellipse_height // 2,
                                ellipse_width,
                                ellipse_height
                            ),
                            2
                        )

        # Display winner or draw message
        if hasattr(self, "done") and self.done:
            font = pygame.font.SysFont(None, 48)
            winner_text = f"{self.winner} won!" if hasattr(self, "winner") else "Draw!"
            text = font.render(winner_text, True, (255, 0, 0))
            surf.blit(text, (WINDOW_SIZE[0] // 2 - text.get_width() // 2, 20))
            font2 = pygame.font.SysFont(None, 36)
            text2 = font2.render("Press R to reset", True, (0, 0, 0))
            surf.blit(text2, (WINDOW_SIZE[0] // 2 - text2.get_width() // 2, 70))

        # Draw controls at the bottom
        font = pygame.font.SysFont(None, 28)
        controls_y = WINDOW_SIZE[1] - 110

        # WASD player (X)
        wasd_text1 = font.render("W", True, (255, 0, 0))
        wasd_text2 = font.render("A   S   D", True, (255, 0, 0))
        wasd_text3 = font.render("J: Place X", True, (255, 0, 0))
        surf.blit(wasd_text1, (45, controls_y))
        surf.blit(wasd_text2, (20, controls_y + 30))
        surf.blit(wasd_text3, (20, controls_y + 60))

        # Arrow keys player (O)
        
        arrow_text1 = font.render("^", True, (0, 0, 255))
        arrow_text2 = font.render("<  v   >", True, (0, 0, 255))
        arrow_text3 = font.render("Space: Place O", True, (0, 0, 255))
        surf.blit(arrow_text1, (WINDOW_SIZE[0] // 2 + CELL_SIZE * 1.23, controls_y))
        surf.blit(arrow_text2, (WINDOW_SIZE[0] // 2 + CELL_SIZE , controls_y + 30))
        surf.blit(arrow_text3, (WINDOW_SIZE[0] // 2 + CELL_SIZE * 0.5, controls_y + 60))

        # Display game mode
        font_mode = pygame.font.SysFont(None, 32)
        mode_text = font_mode.render("PvP" if self.game_mode == "pvp" else "PvE", True, (0, 0, 0))
        surf.blit(mode_text, (WINDOW_SIZE[0] - mode_text.get_width() - 20, 20))

        self.window.blit(surf, (0, 0))
        pygame.display.flip()
        self.clock.tick(self.metadata["render_fps"])


    def _render_rgb(self):
        # Simple 2D rendering for observation
        img = np.zeros((210, 160, 3), dtype=np.uint8)
        # Draw grid and marks
        scale_x = 160 // GRID_SIZE
        scale_y = 210 // GRID_SIZE
        for z in range(GRID_SIZE):
            for x in range(GRID_SIZE):
                for y in range(GRID_SIZE):
                    px = x * scale_x + z * 2
                    py = y * scale_y + z * 2
                    v = self.board[x, y, z]
                    if v == 1:
                        img[py:py+scale_y, px:px+scale_x] = [255, 80, 80]
                    elif v == 2:
                        img[py:py+scale_y, px:px+scale_x] = [80, 80, 255]
        return img

    def close(self):
        if self.window is not None:
            pygame.quit()
            self.window = None

# Register environment for gymnasium
def make_jax_tictactoe3d_env(**kwargs):
    return JaxTicTacToe3D(**kwargs)

def ai_move(env):
    board = np.array(env.board)
    directions = [
        (1, 0, 0), (0, 1, 0), (0, 0, 1),
        (1, 1, 0), (1, 0, 1), (0, 1, 1),
        (1, 1, 1), (1, -1, 0), (1, 0, -1),
        (0, 1, -1), (1, -1, 1), (1, 1, -1), (1, -1, -1)
    ]

    # 1. Try to win
    for x in range(GRID_SIZE):
        for y in range(GRID_SIZE):
            for z in range(GRID_SIZE):
                if board[x, y, z] == 0:
                    board[x, y, z] = 1
                    if env._check_win(1):
                        env.board = env.board.at[x, y, z].set(1)
                        env.turn = 2
                        if env._check_win(1):
                            env.done = True
                            env.winner = "X"
                        return
                    board[x, y, z] = 0

    # 2. Block O's immediate win
    for x in range(GRID_SIZE):
        for y in range(GRID_SIZE):
            for z in range(GRID_SIZE):
                if board[x, y, z] == 0:
                    board[x, y, z] = 2
                    if env._check_win(2):
                        env.board = env.board.at[x, y, z].set(1)
                        env.turn = 2
                        return
                    board[x, y, z] = 0

    # 3. Block any line with 3 O's and 1 empty (potential win next turn)
    for x in range(GRID_SIZE):
        for y in range(GRID_SIZE):
            for z in range(GRID_SIZE):
                for dx, dy, dz in directions:
                    cells = []
                    for i in range(4):
                        nx, ny, nz = x + i * dx, y + i * dy, z + i * dz
                        if 0 <= nx < GRID_SIZE and 0 <= ny < GRID_SIZE and 0 <= nz < GRID_SIZE:
                            cells.append((nx, ny, nz))
                        else:
                            break
                    if len(cells) == 4:
                        values = [board[a, b, c] for a, b, c in cells]
                        if values.count(2) == 3 and values.count(0) == 1:
                            # Block the empty cell
                            idx = values.index(0)
                            bx, by, bz = cells[idx]
                            env.board = env.board.at[bx, by, bz].set(1)
                            env.turn = 2
                            return

    # 4. Otherwise, pick random
    empties = np.argwhere(board == 0)
    if len(empties) > 0:
        x, y, z = empties[np.random.choice(len(empties))]
        env.board = env.board.at[x, y, z].set(1)
        env.turn = 2
        if env._check_win(1):
            env.done = True
            env.winner = "X"

if __name__ == "__main__":
    env = JaxTicTacToe3D()
    obs, _ = env.reset()
    running = True
    while running:
        env.render()
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                # Always allow game mode change
                if event.key == pygame.K_g:
                    env.game_mode = "pve" if env.game_mode == "pvp" else "pvp"
                    obs, _ = env.reset()
                    env.done = False
                    env.turn = 2
                    env.cursor_circle = [0, 0, 0]
                    env.cursor_x = [0, 0, 0]
                elif env.done:
                    if event.key == pygame.K_r:
                        obs, _ = env.reset()
                        env.done = False
                        env.turn = 2
                        env.cursor_circle = [0, 0, 0]
                        env.cursor_x = [0, 0, 0]
                else:
                    # Move circle cursor with arrows
                    if event.key == pygame.K_LEFT:
                        env.cursor_circle[0] = (env.cursor_circle[0] - 1) % GRID_SIZE
                    elif event.key == pygame.K_RIGHT:
                        env.cursor_circle[0] = (env.cursor_circle[0] + 1) % GRID_SIZE
                    elif event.key == pygame.K_UP:
                        if env.cursor_circle[1] == 0:
                            if env.cursor_circle[2] == 0:
                                env.cursor_circle[2] = GRID_SIZE - 1
                                env.cursor_circle[1] = GRID_SIZE - 1
                            else:
                                env.cursor_circle[2] -= 1
                                env.cursor_circle[1] = GRID_SIZE - 1
                        else:
                            env.cursor_circle[1] -= 1
                    elif event.key == pygame.K_DOWN:
                        if env.cursor_circle[1] == GRID_SIZE - 1:
                            env.cursor_circle[2] = (env.cursor_circle[2] + 1) % GRID_SIZE
                            env.cursor_circle[1] = 0
                        else:
                            env.cursor_circle[1] += 1
                    # Move X cursor with WASD
                    elif event.key == pygame.K_a:
                        env.cursor_x[0] = (env.cursor_x[0] - 1) % GRID_SIZE
                    elif event.key == pygame.K_d:
                        env.cursor_x[0] = (env.cursor_x[0] + 1) % GRID_SIZE
                    elif event.key == pygame.K_w:
                        if env.cursor_x[1] == 0:
                            if env.cursor_x[2] == 0:
                                env.cursor_x[2] = GRID_SIZE - 1
                                env.cursor_x[1] = GRID_SIZE - 1
                            else:
                                env.cursor_x[2] -= 1
                                env.cursor_x[1] = GRID_SIZE - 1
                        else:
                            env.cursor_x[1] -= 1
                    elif event.key == pygame.K_s:
                        if env.cursor_x[1] == GRID_SIZE - 1:
                            env.cursor_x[2] = (env.cursor_x[2] + 1) % GRID_SIZE
                            env.cursor_x[1] = 0
                        else:
                            env.cursor_x[1] += 1
                    # Place O with space (circle's turn)
                    elif event.key == pygame.K_SPACE and env.turn == 2:
                        x, y, z = env.cursor_circle
                        if env.board[x, y, z] == 0:
                            env.board = env.board.at[x, y, z].set(2)
                            if env._check_win(2):
                                env.done = True
                                env.winner = "O"
                            else:
                                env.turn = 1
                                if env.game_mode == "pve" and not env.done:
                                    ai_move(env)
                    # Place X with J (X's turn)
                    elif event.key == pygame.K_j and env.turn == 1 and env.game_mode == "pvp":
                        x, y, z = env.cursor_x
                        if env.board[x, y, z] == 0:
                            env.board = env.board.at[x, y, z].set(1)
                            if env._check_win(1):
                                env.done = True
                                env.winner = "X"
                            else:
                                env.turn = 2
