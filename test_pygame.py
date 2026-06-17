import pygame
import sys

# 1. Initialize Pygame
pygame.init()

# 2. Set up the display window
SCREEN_WIDTH = 1820
SCREEN_HEIGHT = 980
screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
# screen = pygame.display.set_mode((0, 0), pygame.FULLSCREEN)
pygame.display.set_caption("Simple Pygame Example")

# 3. Setup game clock for controlling frame rate
clock = pygame.time.Clock()

# 4. Define colors (RGB format)
BACKGROUND_COLOR = (30, 30, 40)       # Dark gray-blue
RECT_COLOR = (0, 255, 128)            # Mint green

# 5. Define a rectangle to draw (x, y, width, height)
player_rect = pygame.Rect(350, 250, 100, 100)

# Main Game Loop
running = True
while running:
    # 6. Event Handling Loop
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

    # 7. Game Logic (Optional: add movement or calculations here)

    # 8. Drawing / Rendering
    screen.fill(BACKGROUND_COLOR) # Clear screen with background color
    
    # Draw a simple shape on the screen
    pygame.draw.rect(screen, RECT_COLOR, player_rect)

    # 9. Update the display
    pygame.display.flip()

    # 10. Frame rate control (Caps the game at 60 FPS)
    clock.tick(60)

# Clean up and close the program safely
pygame.quit()
sys.exit()