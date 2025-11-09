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

# Sample words to visualize
WORDS = [
    # Animals
    "cat", "dog", "kitten", "puppy", "lion", "tiger", "bear", "wolf",
    "rabbit", "fox", "deer", "elephant", "giraffe", "zebra",
    
    # Vehicles
    "car", "vehicle", "truck", "bicycle", "motorcycle", "bus", "train",
    "airplane", "boat", "ship", "helicopter", "scooter",
    
    # Fruits
    "apple fruit", "orange fruit", "banana", "grape", "strawberry", "watermelon",
    "pineapple", "mango", "peach", "cherry", "lemon", "lime",
    
    # Technology
    "computer", "laptop", "keyboard", "mouse", "phone", "tablet",
    "monitor", "printer", "router", "server", "software", "hardware", "Apple company"
    
    # Emotions
    "happy", "joyful", "sad", "angry", "excited", "calm", "worried",
    "surprised", "confused", "proud", "jealous", "grateful",
    
    # Actions
    "run", "walk", "sprint", "jog", "jump", "climb", "swim", "fly",
    "dance", "sing", "laugh", "cry", "sleep", "eat",
    
    # Geography
    "ocean", "sea", "water", "lake", "river", "stream", "pond",
    "mountain", "hill", "valley", "peak", "forest", "desert", "beach",
    
    # Weather
    "sunny", "rainy", "cloudy", "windy", "snowy", "foggy", "stormy",
    
    # Colors
    "red", "blue", "green", "yellow", "purple", "orange color", "black", "white"
]

# Physics constants
SPRING_STRENGTH = 0.02
REPULSION_STRENGTH = 800
DAMPING = 0.92
SIMILARITY_THRESHOLD = 0.85  # NOW this means 85% after rescaling (much stricter!)
MIN_DISTANCE = 40.0
MAX_VELOCITY = 15.0
IDEAL_DISTANCE = 120.0
GRAVITY_STRENGTH = 0.005  # Dark matter! Pulls everything toward center

# Simulated annealing
INITIAL_TEMPERATURE = 50.0
COOLING_RATE = 0.998  # Temperature multiplier per frame
temperature = INITIAL_TEMPERATURE

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

# RESCALE using the SAME formula as translator.py
# Map raw [0.6, 1.0] -> [0, 100] then apply stretching
rescaled = (similarity_matrix - 0.6) / 0.4 * 100  # [0.6, 1.0] -> [0, 100]
rescaled = (rescaled - 60) * 2.4 + 50  # Apply the stretch formula
similarity_matrix = np.clip(rescaled / 100, 0, 1)  # Back to [0, 1] range

# Initialize random positions and velocities
positions = np.random.rand(len(WORDS), 2) * [WIDTH - 200, HEIGHT - 200] + [100, 100]
velocities = np.zeros((len(WORDS), 2))

running = True
frame = 0
while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
    
    frame += 1
    # Cool down temperature
    temperature *= COOLING_RATE
    
    # Calculate forces
    forces = np.zeros_like(positions)
    
    # Calculate center of mass
    center = np.mean(positions, axis=0)
    
    for i in range(len(WORDS)):
        # Dark matter: weak pull toward center of mass
        to_center = center - positions[i]
        distance_to_center = np.linalg.norm(to_center)
        if distance_to_center > 0.1:
            gravity = GRAVITY_STRENGTH * distance_to_center
            forces[i] += (to_center / distance_to_center) * gravity
        
        for j in range(len(WORDS)):
            if i == j:
                continue
            
            diff = positions[j] - positions[i]
            distance = np.linalg.norm(diff)
            
            # Clamp distance
            distance = max(distance, MIN_DISTANCE)
            
            direction = diff / distance
            
            # Spring force with ideal distance (Hooke's law)
            if similarity_matrix[i, j] > SIMILARITY_THRESHOLD:
                displacement = distance - IDEAL_DISTANCE
                spring_force = SPRING_STRENGTH * displacement * similarity_matrix[i, j]
                forces[i] += direction * spring_force
            # Remove the medium similarity repulsion - it's creating spaghetti!
            
            # Repulsion (weakens as temperature drops)
            effective_repulsion = REPULSION_STRENGTH * (1 + temperature / INITIAL_TEMPERATURE)
            repulsion = effective_repulsion / (distance ** 2)
            forces[i] -= direction * repulsion
    
    # Add random "thermal" motion
    thermal_noise = np.random.randn(*forces.shape) * temperature * 0.1
    forces += thermal_noise
    
    # Update velocities and positions
    velocities += forces
    velocities *= DAMPING
    
    # Cap velocity
    velocity_mags = np.linalg.norm(velocities, axis=1, keepdims=True)
    velocity_mags = np.maximum(velocity_mags, 1e-10)
    velocities = np.where(
        velocity_mags > MAX_VELOCITY,
        velocities * MAX_VELOCITY / velocity_mags,
        velocities
    )
    
    positions += velocities
    
    # Clear screen
    screen.fill(BG_COLOR)
    
    # Calculate bounding box of all positions
    min_x, min_y = np.min(positions, axis=0)
    max_x, max_y = np.max(positions, axis=0)
    width = max_x - min_x
    height = max_y - min_y
    avg = (width + height) / 2
    if avg < 0.01:
        print("They imploded again")
    elif avg > 10000:
        print("They exploded again")
    
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
    
    # Draw connections for similar pairs
    for i in range(len(WORDS)):
        for j in range(i + 1, len(WORDS)):
            if similarity_matrix[i, j] > SIMILARITY_THRESHOLD:
                color = (40, 40, 60, int(100 * similarity_matrix[i, j]))
                pygame.draw.line(screen, color[:3], screen_positions[i].astype(int), screen_positions[j].astype(int), 1)
    
    # Draw nodes and labels
    for i, pos in enumerate(screen_positions):
        pygame.draw.circle(screen, POINT_COLOR, pos.astype(int), 5)
        label = small_font.render(WORDS[i], True, TEXT_COLOR)
        screen.blit(label, (int(pos[0]) + 8, int(pos[1]) - 8))
    
    # Draw temperature indicator
    temp_text = small_font.render(f"Temp: {temperature:.1f}", True, (255, 200, 100))
    screen.blit(temp_text, (10, 10))
    
    # Update display
    pygame.display.flip()
    clock.tick(FPS)

pygame.quit()
