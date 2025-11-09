import pygame
import numpy as np
import translator
import random
import math

# Initialize pygame
pygame.init()

# Constants
WIDTH, HEIGHT = 1200, 800
BG_COLOR = (20, 20, 30)
TEXT_COLOR = (200, 200, 200)
POINT_COLOR = (100, 150, 255)
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

# Physics constants
ATTRACTION_STRENGTH = 0.01
REPULSION_STRENGTH = 5000
DAMPING = 0.85
MIN_DISTANCE = 10
TELEPORT_RADIUS = 80
CLICK_BLAST_RADIUS = 150
CLICK_BLAST_STRENGTH = 50

# Setup display
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Force-Directed Embedding Graph")
clock = pygame.time.Clock()
small_font = pygame.font.Font(None, 18)

# Load embeddings
print("Loading embeddings...")
embeddings_dict = translator.embed_words(WORDS)
embeddings = np.array([embeddings_dict[word] for word in WORDS])

# Normalize embeddings
embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)

# Calculate cosine similarity matrix
similarity_matrix = np.dot(embeddings, embeddings.T)

# Initialize random positions and velocities
positions = np.random.rand(len(WORDS), 2) * [WIDTH - 200, HEIGHT - 200] + [100, 100]
velocities = np.zeros((len(WORDS), 2))

running = True
while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
        elif event.type == pygame.MOUSEBUTTONDOWN:
            # Get mouse position in world coordinates
            mouse_x, mouse_y = event.pos
            
            # Convert screen position back to world position
            if range_x > 0 and range_y > 0:
                world_mouse = np.array([
                    (mouse_x - padding) / scale + min_x,
                    (mouse_y - padding) / scale + min_y
                ])
            else:
                world_mouse = np.array(event.pos)
            
            # Apply blast force to nearby nodes
            for i in range(len(WORDS)):
                diff = positions[i] - world_mouse
                distance = np.linalg.norm(diff)
                
                if distance < CLICK_BLAST_RADIUS:
                    if distance < 0.1:
                        distance = 0.1
                    direction = diff / distance
                    # Inverse square blast force
                    blast_force = CLICK_BLAST_STRENGTH / (distance * 0.1)
                    velocities[i] += direction * blast_force
    
    # Calculate forces
    forces = np.zeros_like(positions)
    
    for i in range(len(WORDS)):
        for j in range(len(WORDS)):
            if i == j:
                continue
            
            diff = positions[j] - positions[i]
            distance = np.linalg.norm(diff)
            
            if distance < 0.1:
                distance = 0.1
            
            direction = diff / distance
            
            # Check if too close - teleport
            if distance < MIN_DISTANCE:
                angle = random.uniform(0, 2 * math.pi)
                positions[j] = positions[i] + np.array([
                    math.cos(angle) * TELEPORT_RADIUS,
                    math.sin(angle) * TELEPORT_RADIUS
                ])
                continue
            
            # Attraction based on similarity
            similarity = similarity_matrix[i, j] - 0.5
            similarity *= 0.8
            if similarity > 0.3:
                # Only attract significantly similar words
                attraction = ATTRACTION_STRENGTH * similarity * distance
                forces[i] += direction * attraction
            elif similarity < 0:
                # Repulsion for dissimilar words
                repulsion = REPULSION_STRENGTH * (-similarity) / (distance ** 2)
                forces[i] -= direction * repulsion
            # Words with similarity between 0 and 0.3 have no force (neutral)

            # grr force: if two very similar nods are very far apart, they attract a bonus force
            if similarity_matrix[i, j] > 0.9 and distance > 100:
                bonus_attraction = ATTRACTION_STRENGTH * 5 * (distance - 200) / 200
                forces[i] += direction * bonus_attraction
            
            # Repulsion (inverse square law)
            repulsion = REPULSION_STRENGTH / (distance ** 2)
            forces[i] -= direction * repulsion
    
    # Update velocities and positions
    velocities += forces
    velocities *= DAMPING
    positions += velocities
    
    
    # Clear screen
    screen.fill(BG_COLOR)
    
    # Calculate bounding box of all positions
    min_x, min_y = np.min(positions, axis=0)
    max_x, max_y = np.max(positions, axis=0)
    
    # Add padding
    padding = 50
    range_x = max_x - min_x
    range_y = max_y - min_y
    
    # Scale to fit screen
    if range_x > 0 and range_y > 0:
        scale_x = (WIDTH - 2 * padding) / range_x
        scale_y = (HEIGHT - 2 * padding) / range_y
        scale = min(scale_x, scale_y)
        
        # Transform positions to screen coordinates
        screen_positions = (positions - [min_x, min_y]) * scale + [padding, padding]
    else:
        screen_positions = positions
    
    # Draw connections (optional - for highly similar pairs)
    for i in range(len(WORDS)):
        for j in range(i + 1, len(WORDS)):
            if similarity_matrix[i, j] > 0.9:
                color = (40, 40, 60, int(100 * similarity_matrix[i, j]))
                pygame.draw.line(screen, color[:3], screen_positions[i].astype(int), screen_positions[j].astype(int), 1)
    
    # Draw nodes and labels
    for i, pos in enumerate(screen_positions):
        pygame.draw.circle(screen, POINT_COLOR, pos.astype(int), 5)
        label = small_font.render(WORDS[i], True, TEXT_COLOR)
        screen.blit(label, (int(pos[0]) + 8, int(pos[1]) - 8))
    
    # Update display
    pygame.display.flip()
    clock.tick(FPS)

pygame.quit()