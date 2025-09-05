import pygame
import random
import sys
import numpy as np
from rl import Agent
from barras import draw_bar, SmokeParticle

pygame.init()

# Screen setup
WIDTH, HEIGHT = 850, 500
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Pixel Nuclear Usine")

clock = pygame.time.Clock()

# Colors
BLACK = (0, 0, 0)
WHITE = (255, 255, 255)
GRAY = (100, 100, 100)
DARK_GRAY = (50, 50, 50)
GREEN = (0, 200, 0)
RED = (200, 0, 0)
BLUE = (0, 0, 200)
YELLOW = (200, 200, 0)

V_BAR_WIDTH = 60
V_BAR_MAX_HEIGHT = 350
V_BAR_Y = (HEIGHT - V_BAR_MAX_HEIGHT) // 2
RED_BAR_X = 550
BLUE_BAR_X = 650
BAR_BORDER_RADIUS = 20
MAX_VALUE = 60

# --- Colors ---
BAR_BG_COLOR = (210, 210, 215)
# Red Gradient
RED_START = (255, 120, 120)
RED_END = (220, 50, 50)
# Blue Gradient
BLUE_START = (120, 120, 255)
BLUE_END = (50, 50, 220)

# Smoke particles list
smoke_particles = []
SMOKE_SPAWN_POINT = (265, 190) # Fine-tuned position for the provided image

def scale(img: pygame.Surface, factor):
    w, h = img.get_width() * factor, img.get_height() * factor
    return pygame.transform.scale(img, (int(w), int(h)))


IMAGE = pygame.image.load('smoke.png').convert_alpha()







# Draw nuclear usine
def draw_usine(surface):
    # main building
    pygame.draw.rect(surface, DARK_GRAY, (80, 250, 120, 150))
    pygame.draw.rect(surface, GRAY, (200, 200, 80, 200))
    # cooling towers
    pygame.draw.rect(surface, GRAY, (100, 150, 40, 100))
    pygame.draw.rect(surface, GRAY, (160, 170, 40, 80))
    pygame.draw.rect(surface, GRAY, (220, 140, 60, 120))



# Setup
smokes = []
energy_level = 0
usine_img = pygame.image.load("nuke-usine.jpg").convert()
usine_img = pygame.transform.scale(usine_img, (300, 300))  # resize if needed
usine_img.set_colorkey((28, 28, 28))

temp = 0





#Treinando o Agente
agente = Agent()

for i in range(20):
    print(i)
    agente.policy_evaluation()
    if(agente.policy_improvement()):
        print("-----Otimo!")
        break
print()





temp = 15
agua = 8

while True:
    screen.fill(BLACK)

    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            pygame.quit()
            sys.exit()



    # --- Draw everything ---
    screen.blit(usine_img, (120, 160))

    # Create new smoke particles
    if random.random() > 0.1: # Randomly spawn particles
        smoke_particles.append(SmokeParticle(SMOKE_SPAWN_POINT[0], SMOKE_SPAWN_POINT[1]))
    
    # Update and remove dead particles
    for particle in smoke_particles:
        particle.update()
    smoke_particles = [p for p in smoke_particles if p.alpha > 0]

    # Draw smoke particles
    for particle in smoke_particles:
        particle.draw(screen)

    # Bars for values
    draw_bar(screen, RED_BAR_X, V_BAR_Y, V_BAR_WIDTH, V_BAR_MAX_HEIGHT, temp, MAX_VALUE, RED_START, RED_END)
    # Draw the blue bar
    draw_bar(screen, BLUE_BAR_X, V_BAR_Y, V_BAR_WIDTH, V_BAR_MAX_HEIGHT, agua, MAX_VALUE, BLUE_START, BLUE_END)


    #Atualizando valores
    agua_v = agua
    agua += agente.policy[int(agua)][int(temp)]
    temp = temp+int(0.1*temp)-int(0.2*agua_v)

    #Mudando a temperatura aleatoriamente
    if(random.randint(0,20) == 17):
        temp = 20

    pygame.display.flip()
    clock.tick(10)