import numpy as np
import translator
import networkx as nx
from collections import defaultdict

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
    "monitor", "printer", "router", "server", "software", "hardware", "Apple",
    
    # Emotions
    "happy", "joyful", "sad", "angry", "excited", "calm", "worried",
    "surprised", "confused", "proud", "jealous", "grateful",
    
    # Geography
    "ocean", "sea", "water", "lake", "river", "mountain", "hill", "valley",
]

# Load embeddings
print("Loading embeddings...")
embeddings_dict = translator.embed_words(WORDS)
embeddings = np.array([embeddings_dict[word] for word in WORDS])

# Normalize embeddings
embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)

# Calculate similarity matrix (simple cosine similarity, no rescaling)
similarity_matrix = np.dot(embeddings, embeddings.T)

# Build weighted graph
G = nx.Graph()
for i in range(len(WORDS)):
    G.add_node(i, word=WORDS[i])
    for j in range(i + 1, len(WORDS)):
        # Add edge with similarity as weight
        weight = float(similarity_matrix[i, j])**19
        G.add_edge(i, j, weight=weight)

# Use Louvain algorithm (maximizes modularity automatically)
from networkx.algorithms import community
communities_generator = community.greedy_modularity_communities(G, weight='weight')
communities = [list(c) for c in communities_generator]

# Calculate modularity
modularity = community.modularity(G, communities_generator, weight='weight')

print(f"\n{'='*60}")
print(f"Found {len(communities)} communities")
print(f"Modularity: {modularity:.3f} (higher = better-defined groups)")
print(f"{'='*60}\n")

# Sort communities by size
communities.sort(key=len, reverse=True)

for idx, comm in enumerate(communities):
    words = [WORDS[i] for i in comm]
    
    # Calculate average internal similarity
    internal_sims = []
    for i in comm:
        for j in comm:
            if i < j and G.has_edge(i, j):
                internal_sims.append(G[i][j]['weight'])
    
    avg_internal = np.mean(internal_sims) if internal_sims else 0
    
    print(f"Community {idx + 1} ({len(comm)} nodes, avg similarity: {avg_internal:.2f}):")
    print(f"  {', '.join(words)}")
    print()