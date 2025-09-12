import pygame
import numpy as np
import sys
import time

from loja import Loja
from loja import Agent

# Initialize Pygame
pygame.init()

class EpisodeVisualizer:
    def __init__(self, episode, width=1000, height=700):
        self.episode = episode
        self.width = width
        self.height = height
        
        # Create display
        self.screen = pygame.display.set_mode((width, height))
        pygame.display.set_caption("Monte Carlo Episódio")
        
        # Colors
        self.WHITE = (255, 255, 255)
        self.BLACK = (0, 0, 0)
        self.RED = (220, 50, 50)
        self.BLUE = (50, 100, 220)
        self.GREEN = (50, 180, 50)
        self.GRAY = (100, 100, 100)
        self.LIGHT_GRAY = (230, 230, 230)
        
        # Bigger fonts
        self.title_font = pygame.font.Font(None, 48)
        self.font = pygame.font.Font(None, 32)
        self.small_font = pygame.font.Font(None, 24)
        
        # Playback state
        self.current_step = 0
        self.running = True
        self.speed = 3.0  # Faster - 3 steps per second
        self.last_step_time = 0
        
        # Calculate cumulative profits
        self.cumulative_profits = []
        total = 0
        for step_data in self.episode:
            total += step_data[2]  # reward is now at index 2
            self.cumulative_profits.append(total)
        
        # Layout
        self.graph_x = 50
        self.graph_y = 120
        self.graph_width = 450
        self.graph_height = 250
        
        self.info_x = 550
        self.info_y = 120
        
        # Get max stock for scaling - we only have current states now
        all_states = [step[0] for step in episode]
        if all_states:
            all_stocks = [max(state[0], state[1]) for state in all_states]
            self.max_stock = max(all_stocks)
        else:
            self.max_stock = 50
    
    def draw_graph(self):
        if self.current_step == 0:
            return
        
        # Graph background
        graph_rect = pygame.Rect(self.graph_x, self.graph_y, self.graph_width, self.graph_height)
        pygame.draw.rect(self.screen, self.WHITE, graph_rect)
        pygame.draw.rect(self.screen, self.BLACK, graph_rect, 3)
        
        # Title
        title = self.font.render("Níveis do estoque", True, self.BLACK)
        self.screen.blit(title, (self.graph_x, self.graph_y - 40))
        
        # Get current data - only current states available
        current_episode = self.episode[:self.current_step]
        states = [step[0] for step in current_episode]
        
        if len(states) < 1:
            return
            
        stock_A = [state[0] for state in states]
        stock_B = [state[1] for state in states]
        
        if len(stock_A) < 2:
            # If only one point, just show current stock levels as bars
            if len(stock_A) == 1:
                y_scale = self.graph_height / max(self.max_stock, 1)
                bar_width = 40
                
                # Stock A bar
                bar_height_A = stock_A[0] * y_scale
                bar_rect_A = pygame.Rect(self.graph_x + 100, 
                                       self.graph_y + self.graph_height - bar_height_A,
                                       bar_width, bar_height_A)
                pygame.draw.rect(self.screen, self.RED, bar_rect_A)
                
                # Stock B bar
                bar_height_B = stock_B[0] * y_scale
                bar_rect_B = pygame.Rect(self.graph_x + 200, 
                                       self.graph_y + self.graph_height - bar_height_B,
                                       bar_width, bar_height_B)
                pygame.draw.rect(self.screen, self.BLUE, bar_rect_B)
                
                # Labels
                label_A = self.small_font.render(f"A: {stock_A[0]}", True, self.BLACK)
                self.screen.blit(label_A, (self.graph_x + 95, self.graph_y + self.graph_height + 10))
                
                label_B = self.small_font.render(f"B: {stock_B[0]}", True, self.BLACK)
                self.screen.blit(label_B, (self.graph_x + 195, self.graph_y + self.graph_height + 10))
            return
        
        # Scale
        x_scale = self.graph_width / max(len(self.episode), 1)
        y_scale = self.graph_height / max(self.max_stock, 1)
        
        # Grid lines
        for i in range(0, int(self.max_stock) + 1, max(1, int(self.max_stock // 5))):
            y = self.graph_y + self.graph_height - (i * y_scale)
            pygame.draw.line(self.screen, self.LIGHT_GRAY, 
                           (self.graph_x, y), (self.graph_x + self.graph_width, y), 1)
            # Label
            label = self.small_font.render(str(i), True, self.GRAY)
            self.screen.blit(label, (self.graph_x - 35, y - 10))
        
        # Draw lines
        points_A = [(self.graph_x + i * x_scale, 
                    self.graph_y + self.graph_height - stock_A[i] * y_scale) 
                   for i in range(len(stock_A))]
        points_B = [(self.graph_x + i * x_scale, 
                    self.graph_y + self.graph_height - stock_B[i] * y_scale) 
                   for i in range(len(stock_B))]
        
        pygame.draw.lines(self.screen, self.RED, False, points_A, 4)
        pygame.draw.lines(self.screen, self.BLUE, False, points_B, 4)
        
        # Current points
        current_x = self.graph_x + (len(states) - 1) * x_scale
        current_y_A = self.graph_y + self.graph_height - stock_A[-1] * y_scale
        current_y_B = self.graph_y + self.graph_height - stock_B[-1] * y_scale
        
        pygame.draw.circle(self.screen, self.RED, (int(current_x), int(current_y_A)), 8)
        pygame.draw.circle(self.screen, self.BLUE, (int(current_x), int(current_y_B)), 8)
        
        # Legend
        legend_y = self.graph_y + self.graph_height + 20
        pygame.draw.line(self.screen, self.RED, (self.graph_x, legend_y), (self.graph_x + 30, legend_y), 4)
        text = self.font.render("Produto A", True, self.BLACK)
        self.screen.blit(text, (self.graph_x + 40, legend_y - 12))
        
        pygame.draw.line(self.screen, self.BLUE, (self.graph_x + 200, legend_y), (self.graph_x + 230, legend_y), 4)
        text = self.font.render("Produto B", True, self.BLACK)
        self.screen.blit(text, (self.graph_x + 240, legend_y - 12))
    
    def draw_info(self):
        """Draw current step info"""
        if self.current_step >= len(self.episode):
            # Episode finished
            text = self.title_font.render("EPISÓDIO COMPLETO", True, self.GREEN)
            self.screen.blit(text, (self.info_x, self.info_y))
            
            final_profit = self.cumulative_profits[-1] if self.cumulative_profits else 0
            color = self.GREEN if final_profit >= 0 else self.RED
            text = self.title_font.render(f"Lucro final: ${final_profit:.0f}", True, color)
            self.screen.blit(text, (self.info_x, self.info_y + 60))
            return
        
        state, action, reward = self.episode[self.current_step]  # Updated format
        cumulative = self.cumulative_profits[self.current_step]
        
        y = self.info_y
        
        # Step number
        text = self.font.render(f"Etapa {self.current_step + 1}/{len(self.episode)}", True, self.BLACK)
        self.screen.blit(text, (self.info_x, y))
        y += 50
        
        # Current stock
        text = self.font.render(f"Estoque A: {state[0]}", True, self.RED)
        self.screen.blit(text, (self.info_x, y))
        y += 35
        
        text = self.font.render(f"Estoque B: {state[1]}", True, self.BLUE)
        self.screen.blit(text, (self.info_x, y))
        y += 50
        
        # Decision
        text = self.font.render("Pedidos agente:", True, self.BLACK)
        self.screen.blit(text, (self.info_x, y))
        y += 35
        
        text = self.font.render(f"  A: {action[0]} units", True, self.GREEN)
        self.screen.blit(text, (self.info_x, y))
        y += 35
        
        text = self.font.render(f"  B: {action[1]} units", True, self.GREEN)
        self.screen.blit(text, (self.info_x, y))
        y += 50
        
        # Results
        text = self.font.render(f"Lucro etapa: ${reward:.0f}", True, self.GREEN if reward >= 0 else self.RED)
        self.screen.blit(text, (self.info_x, y))
        y += 40
        
        text = self.font.render(f"Lucro total: ${cumulative:.0f}", True, self.GREEN if cumulative >= 0 else self.RED)
        self.screen.blit(text, (self.info_x, y))
    
    def handle_events(self):
        """Simple event handling - just exit on close/escape"""
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    self.running = False
    
    def run(self):
        """Simple playback loop"""
        clock = pygame.time.Clock()
        
        while self.running:
            current_time = time.time()
            
            self.handle_events()
            
            # Auto-advance steps
            if (current_time - self.last_step_time) > (1.0 / self.speed):
                if self.current_step < len(self.episode):
                    self.current_step += 1
                elif self.current_step >= len(self.episode) + 10:  # Show final screen for a bit
                    self.running = False
                self.last_step_time = current_time
            
            # Draw
            self.screen.fill(self.WHITE)
            
            # Title
            title = self.title_font.render("Monte Carlo Episódio", True, self.BLACK)
            title_rect = title.get_rect(center=(self.width // 2, 40))
            self.screen.blit(title, title_rect)
            
            self.draw_graph()
            self.draw_info()
            
            pygame.display.flip()
            clock.tick(60)
        
        pygame.quit()
        sys.exit()

def visualize_episode(episode):
    """Run episode visualization once and exit"""
    visualizer = EpisodeVisualizer(episode)
    visualizer.run()

if __name__ == "__main__":
    
    env = Loja()
    agente = Agent(env)

    
    agente.MCControl(episodes=500)

    tests = 1000
    policy_profit = 0
    random_profit = 0

    for _ in range(tests):
        episode = agente.generatePolicyEpisode()
        random = agente.generateBehaviorEpisode()

        for (state, action, reward) in episode: 
            policy_profit += reward

        for (state, action, reward) in random: 
            random_profit += reward

    print(f'Policy: {policy_profit/tests}, Random: {random_profit/tests}')

    visualize_episode(agente.generatePolicyEpisode())
