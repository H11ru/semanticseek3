import pygame
import numpy as np
import translator
import random
# Initialize pygame
pygame.init()

# Constants
WIDTH, HEIGHT = 1200, 800
BG_COLOR = (20, 20, 30)
TEXT_COLOR = (200, 200, 200)
POINT_COLOR = (100, 150, 255)
AXIS_COLOR = (60, 60, 80)
FPS = 60

# Sample words to visualize
WORDS = [
    "cat", "dog", "kitten", "puppy",
    "car", "vehicle", "truck", "bicycle",
    "apple", "orange", "banana", "fruit",
    "computer", "laptop", "keyboard", "mouse",
    "happy", "joyful", "sad", "angry",
    "run", "walk", "sprint", "jog",
    "ocean", "sea", "water", "lake",
    "mountain", "hill", "valley", "peak"
]

# Setup display
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Embedding Visualizer")
clock = pygame.time.Clock()
font = pygame.font.Font(None, 24)
small_font = pygame.font.Font(None, 18)

# Load embeddings
print("Loading embeddings...")
embeddings_dict = translator.embed_words(WORDS)
embeddings = np.array([embeddings_dict[word] for word in WORDS])
num_dimensions = embeddings.shape[1]

# Pick random axes
axis1, axis2 = random.sample(range(num_dimensions), 2)

running = True
while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
        elif event.type == pygame.KEYDOWN:
            if event.key == pygame.K_SPACE:
                # Pick new random axes
                axis1, axis2 = random.sample(range(num_dimensions), 2)
            if event.key == pygame.K_a:
                axis1 = (axis1 - (1 if not pygame.key.get_mods() & pygame.KMOD_SHIFT else 9999999999999999999))
                if axis1 < 0:
                    axis1 = 0
            if event.key == pygame.K_d:
                axis1 = (axis1 + (1 if not pygame.key.get_mods() & pygame.KMOD_SHIFT else 9999999999999999999))
                if axis1 >= num_dimensions:
                    axis1 = num_dimensions - 1
            if event.key == pygame.K_w:
                axis2 = (axis2 - (1 if not pygame.key.get_mods() & pygame.KMOD_SHIFT else 9999999999999999999))
                if axis2 < 0:
                    axis2 = 0
            if event.key == pygame.K_s:
                axis2 = (axis2 + (1 if not pygame.key.get_mods() & pygame.KMOD_SHIFT else 9999999999999999999))
                if axis2 >= num_dimensions:
                    axis2 = num_dimensions - 1
    
    # Clear screen
    screen.fill(BG_COLOR)
    
    # Draw axes
    center_x, center_y = WIDTH // 2, HEIGHT // 2
    pygame.draw.line(screen, AXIS_COLOR, (50, center_y), (WIDTH - 50, center_y), 1)
    pygame.draw.line(screen, AXIS_COLOR, (center_x, 50), (center_x, HEIGHT - 50), 1)
    
    # Extract 2D projections
    points_2d = embeddings[:, [axis1, axis2]]
    
    # Normalize to screen coordinates
    min_vals = points_2d.min(axis=0)
    max_vals = points_2d.max(axis=0)
    range_vals = max_vals - min_vals
    range_vals[range_vals == 0] = 1  # Avoid division by zero
    
    normalized = (points_2d - min_vals) / range_vals
    screen_x = 100 + normalized[:, 0] * (WIDTH - 200)
    screen_y = 100 + normalized[:, 1] * (HEIGHT - 200)
    
    # Draw points and labels
    for i, (x, y) in enumerate(zip(screen_x, screen_y)):
        pygame.draw.circle(screen, POINT_COLOR, (int(x), int(y)), 5)
        label = small_font.render(WORDS[i], True, TEXT_COLOR)
        screen.blit(label, (int(x) + 8, int(y) - 8))
    
    # Draw axis info
    info_text = f"Axes: {axis1} x {axis2} (dimensions: {num_dimensions}) | Press SPACE for new axes"
    info_surface = font.render(info_text, True, TEXT_COLOR)
    screen.blit(info_surface, (10, 10))
    
    # Update display
    pygame.display.flip()
    clock.tick(FPS)

pygame.quit()