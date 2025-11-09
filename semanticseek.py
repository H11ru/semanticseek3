import sys
import random

import translator


banner = """
 _____                            _   _        _____           _
/  ___|                          | | (_)      /  ___|         | |
\ `--.  ___ _ __ ___   __ _ _ __ | |_ _  ___  \ `--.  ___  ___| | __
 `--. \/ _ \ '_ ` _ \ / _` | '_ \| __| |/ __|  `--. \/ _ \/ _ \ |/ /
/\__/ /  __/ | | | | | (_| | | | | |_| | (__  /\__/ /  __/  __/   <
\____/ \___|_| |_| |_|\__,_|_| |_|\__|_|\___| \____/ \___|\___|_|\_\\
_____________________________________________________________________"""

# Word list for the game
CATEGORIES = {
    "fruits": ["apple", "banana", "orange", "grape", "mango", "peach", "pear", "plum", "kiwi", "melon"],
    "animals": ["dog", "cat", "lion", "tiger", "elephant", "giraffe", "zebra", "bear", "wolf", "fox"],
    "countries": ["canada", "brazil", "france", "germany", "india", "japan", "mexico", "nigeria", "russia", "spain", "finland", "egypt"],
    "colors": ["red", "blue", "green", "yellow", "purple", "orange", "pink", "brown", "black", "white", "gold", "mint", "cyan", "silver", "lavender"],
    "objects": ["table", "chair", "computer", "phone", "book", "pen", "bottle", "glass", "lamp", "door"],
    "emotions": ["happy", "sad", "angry", "excited", "bored", "nervous", "calm", "confused", "surprised", "disgusted"],
    "vehicles": ["car", "bike", "bus", "train", "plane", "boat", "truck", "scooter", "subway", "helicopter"],
    "sports": ["soccer", "basketball", "tennis", "baseball", "golf", "swimming", "cycling", "running", "volleyball", "skiing"],
}

# command line arguments
def get_arg_value(*flags):
    """Helper to get value after a flag if it exists in sys.argv."""
    for flag in flags:
        if flag in sys.argv:
            idx = sys.argv.index(flag)
            return idx, sys.argv[idx + 1:] if idx + 1 < len(sys.argv) else []
    return None, []

if "--help" in sys.argv or "-h" in sys.argv:
    print("Semantic Seek - A word similarity guessing game")
    print("\nUsage: python semanticseek.py [OPTIONS]")
    print("\nOptions:")
    print("  --attempts N              Set number of attempts (default: 3 for synonym mode)")
    print("  --quickplay MODE          Skip menu and play MODE directly (approach|synonym)")
    print("  --categories yes|no       Show category hints in approach mode (default: no)")
    print("  --set-category NAME       Force specific category")
    print(f"\nAvailable categories: {', '.join(CATEGORIES.keys())}")
    print("\nExamples:")
    print("  python semanticseek.py --quickplay approach --categories yes")
    print("  python semanticseek.py --set-category fruits --attempts 5")
    print("  python semanticseek.py --quickplay synonym --attempts 10")
    sys.exit(0)

idx, after = get_arg_value("--attempts", "-a")
if idx is not None:
    try:
        custom_attempts = int(after[0])
        if custom_attempts <= 0:
            raise ValueError
    except (IndexError, ValueError):
        print("Usage: python semanticseek.py [--attempts N]")
        print("N must be a positive integer.")
        sys.exit(1)
else:
    custom_attempts = None

idx, after = get_arg_value("--quickplay", "-q")
if idx is not None:
    if not after or after[0] not in ("approach", "synonym"):
        print("Usage: python semanticseek.py --quickplay [approach|synonym]")
        sys.exit(1)
    quickplay_mode = after[0]
else:
    quickplay_mode = None

idx, after = get_arg_value("--categories", "-c")
if idx is not None:
    after = after[0].lower() if after else None
    if not after or after not in ["yes", "no", "y", "n", "true", "false", "on", "off"]:
        print("Usage: python semanticseek.py --categories [yes|no]")
        sys.exit(1)
    categories = after in ["yes", "y", "true", "on"]
else:
    categories = False

idx, after = get_arg_value("--set-category", "-s")
if idx is not None:
    if not after or after[0].lower() not in CATEGORIES:
        print("Usage: python semanticseek.py --set-category [category_name]")
        print(f"Available categories: {', '.join(CATEGORIES.keys())}")
        sys.exit(1)
    selected_category = after[0].lower()
    categories = True
else:
    selected_category = False

def synonym_mode():
    """Player tries to find the most similar word to the shown target."""
    print("\n=== SYNONYM MODE ===")
    print("Find a word as similar as possible to the target word.")
    print("You cannot use the exact same word!\n")
    
    target_word = random.choice([w for cat in CATEGORIES.values() for w in cat if not selected_category or w in CATEGORIES[selected_category]])
    print(f"Target word: {target_word.upper()}\n")
    
    best_score = 0
    attempts = custom_attempts if custom_attempts else 3
    i = 0
    
    while i < attempts:
        guess = input(f"Attempt {i+1}/{attempts}: ").strip().lower()
        
        if guess == 'quit':
            break

        # Check for empty input
        if not guess:
            print("❌ Please enter a word!\n")
            continue
        
        if guess == target_word:
            print("❌ You cannot use the same word!\n")
            continue  # Don't increment i, so they don't lose a turn
            
        try:
            score = translator.get_similarity(guess, target_word)
            print(f"Similarity: {score:.2f}")
            
            if score > best_score:
                best_score = score
                print(f"✨ New best score: {best_score:.2f}!\n")
            else:
                print()
            
            i += 1  # Only increment after a valid attempt
        except translator.TranslatorError as e:
            print(f"❌ Error: {e}\n")
            continue
    
    if best_score == 0:
        print("No valid attempts made. Better luck next time!")
    else:
        print(f"🏆 Final best score: {best_score:.2f}")

def approach_mode():
    """Player guesses words to find the hidden target word."""
    print("\n=== APPROACH MODE ===")
    print("Try to guess the hidden word by entering similar words.")
    print("You'll get a similarity score for each guess (0-100).\n")
    
    cat = random.choice(list(CATEGORIES.keys())) if categories else None
    if selected_category:
        cat = selected_category
    if not cat:
        raise RuntimeError("Category selection failed.")
    target_word = random.choice(CATEGORIES[cat])
    if categories:
        print(f"(HINT: category is {cat.upper()})\n")
    attempts = 0
    guesses = []
    
    while True:
        guess = input("Enter your guess (or 'quit' to give up): ").strip().lower()
        
        # Check for empty input
        if not guess:
            print("❌ Please enter a word!\n")
            continue
        
        if guess == 'quit':
            print(f"\nThe word was: {target_word}")
            break
            
        if guess == target_word:
            attempts += 1
            print(f"\n🎉 Correct! You found the word in {attempts} attempts!")
            break
            
        attempts += 1
        
        try:
            score = translator.get_similarity(guess, target_word)
            guesses.append((guess, score))
            
            print(f"Similarity: {score:.2f}")
            print(f"Previous guesses: {', '.join([f'{w}({s:.1f})' for w, s in guesses])}\n")
        except translator.TranslatorError as e:
            print(f"❌ Error: {e}\n")

print(banner)
print("\nWelcome to Semantic Seek!\n")

while True:
    print("\nChoose a game mode:")
    print("1. Approach - Find the hidden word")
    print("2. Synonym - Find the most similar word")
    print("3. Quit")
    
    if quickplay_mode == "approach":
        approach_mode()
        break
    if quickplay_mode == "synonym":
        synonym_mode()
        break
    choice = input("\nEnter your choice (1-3): ").strip()
    
    if choice == '1':
        approach_mode()
    elif choice == '2':
        synonym_mode()
    elif choice == '3':
        print("\nThanks for playing!")
        break
    else:
        print("Invalid choice. Please enter 1, 2, or 3.")