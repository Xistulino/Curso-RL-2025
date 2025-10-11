import numpy as np
import pygame
from typing import Tuple, Optional

MAX_ANGLE = 30

# muita física, não precisa entender, o que importa é o agente.py :)
class CartPoleEnv:
    def __init__(self, render_mode: Optional[str] = None):
        self.gravity = 9.8
        self.mass_cart = 1.0
        self.mass_pole = 0.1
        self.length = 0.5
        self.force_mag = 10.0
        self.dt = 0.02
        
        self.total_mass = self.mass_cart + self.mass_pole
        self.pole_mass_length = self.mass_pole * self.length
        
        self.x_threshold = 2.4
        self.theta_threshold = MAX_ANGLE * np.pi / 180
        self.max_steps = 500
        
        self.state = None
        self.steps = 0
        
        self.render_mode = render_mode
        self.screen = None
        self.clock = None
        self.screen_width = 600
        self.screen_height = 400
        
    def reset(self) -> np.ndarray:
        self.state = np.random.uniform(low=-0.05, high=0.05, size=4)
        self.steps = 0
        
        if self.render_mode == "human":
            self._render_frame()
        
        return self.state.copy()
    
    # recebe uma ação, retorna o próximo estado, uma recompensa e um booleano de se acabou
    def step(self, action: int) -> Tuple[np.ndarray, float, bool, dict]:
        x, x_dot, theta, theta_dot = self.state
        force = self.force_mag if action == 1 else -self.force_mag
        
        cos_theta = np.cos(theta)
        sin_theta = np.sin(theta)
        
        temp = (force + self.pole_mass_length * theta_dot**2 * sin_theta) / self.total_mass
        theta_acc = (self.gravity * sin_theta - cos_theta * temp) / (
            self.length * (4.0/3.0 - self.mass_pole * cos_theta**2 / self.total_mass)
        )
        x_acc = temp - self.pole_mass_length * theta_acc * cos_theta / self.total_mass
        
        x = x + self.dt * x_dot
        x_dot = x_dot + self.dt * x_acc
        theta = theta + self.dt * theta_dot
        theta_dot = theta_dot + self.dt * theta_acc
        
        self.state = np.array([x, x_dot, theta, theta_dot])
        self.steps += 1
        
        done = bool(
            x < -self.x_threshold or x > self.x_threshold or
            theta < -self.theta_threshold or theta > self.theta_threshold or
            self.steps >= self.max_steps
        )
        
        reward = 1.0 if not done else 0.0
        
        if self.render_mode == "human":
            self._render_frame()
        
        return self.state.copy(), reward, done, {}
    
    def _render_frame(self):
        if self.screen is None:
            pygame.init()
            self.screen = pygame.display.set_mode((self.screen_width, self.screen_height))
            pygame.display.set_caption("CartPole - Semi-Gradient SARSA")
            self.clock = pygame.time.Clock()
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.close()
                return
        
        self.screen.fill((255, 255, 255))
        
        x, _, theta, _ = self.state
        
        scale = self.screen_width / (2 * self.x_threshold)
        cart_y = self.screen_height - 100
        cart_x = self.screen_width / 2 + x * scale
        
        track_y = cart_y + 20
        pygame.draw.line(self.screen, (0, 0, 0), 
                        (0, track_y), (self.screen_width, track_y), 2)
        
        cart_width = 50
        cart_height = 30
        cart_rect = pygame.Rect(
            cart_x - cart_width/2, 
            cart_y - cart_height/2,
            cart_width, 
            cart_height
        )
        pygame.draw.rect(self.screen, (0, 100, 200), cart_rect)
        pygame.draw.rect(self.screen, (0, 0, 0), cart_rect, 2)
        
        pole_length = 2 * self.length * scale * 1.5
        pole_end_x = cart_x + pole_length * np.sin(theta)
        pole_end_y = cart_y - pole_length * np.cos(theta)
        
        pygame.draw.line(self.screen, (200, 0, 0),
                        (cart_x, cart_y),
                        (pole_end_x, pole_end_y), 6)
        
        pygame.draw.circle(self.screen, (50, 50, 50),
                          (int(cart_x), int(cart_y)), 5)
        
        font = pygame.font.Font(None, 28)
        info_text = [
            f"Steps: {self.steps}",
            f"Position: {x:.2f}",
            f"Angle: {theta * 180 / np.pi:.1f}°",
        ]
        
        for i, text in enumerate(info_text):
            text_surface = font.render(text, True, (0, 0, 0))
            self.screen.blit(text_surface, (10, 10 + i * 30))
        
        pygame.display.flip()
        self.clock.tick(50)
    
    def close(self):
        if self.screen is not None:
            pygame.quit()
            self.screen = None


