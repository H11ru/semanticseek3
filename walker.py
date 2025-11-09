import numpy as np
import translator
import random
import time

# Sample words
WORDS = [
    # Animals
    "cat", "dog", "kitten", "puppy", "lion", "tiger", "bear", "wolf",
    "rabbit", "fox", "deer", "elephant", "giraffe", "zebra",
    
    # Vehicles
    "car", "vehicle", "truck", "bicycle", "motorcycle", "bus", "train",
    "airplane", "boat", "ship", "helicopter", "scooter",
    
    # Fruits
    "apple", "orange", "banana", "grape", "strawberry", "watermelon",
    "pineapple", "mango", "peach", "cherry", "lemon", "lime",
    
    # Technology
    "computer", "laptop", "keyboard", "mouse", "phone", "tablet",
    "monitor", "printer", "router", "server", "software", "hardware",
    
    # Emotions
    "happy", "joyful", "sad", "angry", "excited", "calm", "worried",
    "surprised", "confused", "proud", "jealous", "grateful",
    
    # Geography
    "ocean", "sea", "water", "lake", "river", "mountain", "hill", "valley",
]

# Walk parameters
WALK_DELAY = 0.8  # Seconds between steps
SIMILARITY_THRESHOLD = 0.85  # Only walk on strong connections
AVOID_RECENT = 5  # Don't revisit nodes from last N steps

# Load embeddings
print("Loading embeddings...")
embeddings_dict = translator.embed_words(WORDS)
embeddings = np.array([embeddings_dict[word] for word in WORDS])

# Normalize embeddings
embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)

# Calculate similarity matrix (rescaled like in translator.py)
similarity_matrix = np.dot(embeddings, embeddings.T)
rescaled = (similarity_matrix - 0.6) / 0.4 * 100
rescaled = (rescaled - 60) * 2.4 + 50
similarity_matrix = np.clip(rescaled / 100, 0, 1)

# Build adjacency graph (only strong connections)
neighbors = {}
for i in range(len(WORDS)):
    neighbors[i] = []
    for j in range(len(WORDS)):
        if i != j and similarity_matrix[i, j] > SIMILARITY_THRESHOLD:
            neighbors[i].append((j, similarity_matrix[i, j]))
    # Sort by similarity (strongest first)
    neighbors[i].sort(key=lambda x: x[1], reverse=True)

# Walk state
current_idx = random.randint(0, len(WORDS) - 1)
path_history = [current_idx]

print(f"\nStarting walk from: {WORDS[current_idx]}")
print(f"Neighbors: {[WORDS[n[0]] for n in neighbors[current_idx][:5]]}\n")

running = True
try:
    while running:
        time.sleep(WALK_DELAY)
        
        if neighbors[current_idx]:
            # Filter out recently visited nodes
            recent_nodes = set(path_history[-AVOID_RECENT:]) if len(path_history) >= AVOID_RECENT else set(path_history)
            available_choices = [(idx, sim) for idx, sim in neighbors[current_idx] if idx not in recent_nodes]
            
            # If all neighbors are recent, allow any neighbor
            if not available_choices:
                available_choices = neighbors[current_idx]
            
            # Probabilistic choice based on similarity
            weights = [sim ** 3 for _, sim in available_choices]
            next_idx = random.choices([idx for idx, _ in available_choices], weights=weights)[0]
            
            similarity = similarity_matrix[current_idx, next_idx]
            current_idx = next_idx
            path_history.append(current_idx)
            
            # Display current step
            recent_path = " → ".join([WORDS[i] for i in path_history[-10:]])
            print(f"\n[Step {len(path_history)}] {WORDS[current_idx]} (sim: {similarity:.2f})")
            print(f"Recent: {recent_path}")
            print(f"Options: {[(WORDS[n[0]], f'{n[1]:.2f}') for n in neighbors[current_idx][:5]]}")
        else:
            print(f"\nDead end at: {WORDS[current_idx]}")
            print(f"Final path ({len(path_history)} steps): {' → '.join([WORDS[i] for i in path_history])}")
            running = False
            
except KeyboardInterrupt:
    print(f"\n\nWalk stopped by user after {len(path_history)} steps")
    print(f"Full path: {' → '.join([WORDS[i] for i in path_history])}")