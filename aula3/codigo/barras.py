import pygame
import sys
import random

# --- Constants ---
SCREEN_WIDTH = 850
SCREEN_HEIGHT = 500
V_BAR_WIDTH = 60
V_BAR_MAX_HEIGHT = 400
V_BAR_Y = (SCREEN_HEIGHT - V_BAR_MAX_HEIGHT) // 2
RED_BAR_X = SCREEN_WIDTH // 2 - 120
BLUE_BAR_X = SCREEN_WIDTH // 2 + 60
BAR_BORDER_RADIUS = 20
MAX_VALUE = 60

# --- Colors ---
BAR_BG_COLOR = (48, 48, 48)
# Red Gradient
RED_START = (255, 120, 120)
RED_END = (220, 50, 50)
# Blue Gradient
BLUE_START = (120, 120, 255)
BLUE_END = (50, 50, 220)


def draw_bar(surface, x, y, width, height, value, max_value, start_color, end_color):
    # Draw the gray background for the bar
    pygame.draw.rect(surface, BAR_BG_COLOR, (x, y, width, height), border_radius=BAR_BORDER_RADIUS)

    # Calculate the height of the active part of the bar
    active_bar_height = max(0, int((value / max_value) * height))
    if active_bar_height <= 0:
        return # Don't draw the active bar if the value is zero or less

    # Define the rectangle for the active bar (fills from the bottom up)
    active_rect = pygame.Rect(x, y + height - active_bar_height, width, active_bar_height)

    # --- Create the gradient surface ---
    gradient_surface = pygame.Surface(active_rect.size, pygame.SRCALPHA)
    for j in range(active_bar_height):
        # Calculate the interpolation factor 't'. This is based on the position within
        # the *full* bar height to keep the gradient color consistent.
        # The 'y' position in the full bar is (height - active_bar_height + j)
        t = (height - active_bar_height + j) / height
        r = start_color[0] + (end_color[0] - start_color[0]) * t
        g = start_color[1] + (end_color[1] - start_color[1]) * t
        b = start_color[2] + (end_color[2] - start_color[2]) * t
        color = (int(r), int(g), int(b))
        pygame.draw.line(gradient_surface, color, (0, j), (width, j))

    # --- Create the mask for rounded corners ---
    mask = pygame.Surface(active_rect.size, pygame.SRCALPHA)
    # Only round the top corners if the bar is full
    top_radius = BAR_BORDER_RADIUS if value >= max_value-(max_value//10) else 0
    pygame.draw.rect(mask, (255, 255, 255), (0, 0, width, active_bar_height),
                     border_top_left_radius=top_radius,
                     border_top_right_radius=top_radius,
                     border_bottom_left_radius=BAR_BORDER_RADIUS,
                     border_bottom_right_radius=BAR_BORDER_RADIUS)

    # Apply the mask to the gradient
    gradient_surface.blit(mask, (0, 0), special_flags=pygame.BLEND_RGBA_MULT)

    # Blit the final gradient onto the main screen
    surface.blit(gradient_surface, active_rect.topleft)


# --- Smoke Particle Class ---
class SmokeParticle:
    def __init__(self, x, y):
        self.x = x + random.randint(-5, 5) # Start in a slightly random x position
        self.y = y
        self.radius = random.randint(3, 6) # Start with a random size
        self.alpha = 120 # Start semi-transparent (0-255)
        self.color = (200, 200, 200) # Light gray smoke color
        self.x_vel = random.uniform(-0.2, 1.2) # Slow horizontal drift
        self.y_vel = random.uniform(-1.6, -1.4) # Move upwards
        self.growth = 0.1 # How much the radius increases per frame
        self.fade_rate = 1.0 # How quickly it fades

    def update(self):
        """Update particle position, size, and transparency."""
        self.x += self.x_vel
        self.y += self.y_vel
        self.radius += self.growth
        self.alpha -= self.fade_rate

    def draw(self, surface):
        """
        Draws the particle as a soft, transparent circle.
        This is done by creating a temporary surface for each particle.
        """
        if self.alpha > 0 and self.radius > 0:
            # Create a temporary surface with per-pixel alpha
            temp_surface = pygame.Surface((self.radius * 2, self.radius * 2), pygame.SRCALPHA)
            # Draw the circle in the center of the temporary surface
            pygame.draw.circle(temp_surface, (*self.color, int(self.alpha)), (self.radius, self.radius), self.radius)
            # Blit the temporary surface onto the main screen
            surface.blit(temp_surface, (self.x - self.radius, self.y - self.radius), special_flags=pygame.BLEND_RGBA_ADD)