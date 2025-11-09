print("⚠️ Deleting all model caches to refresh them. Use command line argument --confirm to do this")
import sys
if "--confirm" not in sys.argv:
    print("Aborting. To confirm, run again with --confirm")
    sys.exit(1)
import shutil
from pathlib import Path

# Find the sentence-transformers cache
cache_dir = Path.home() / ".cache" / "huggingface" / "hub"
torch_cache = Path.home() / ".cache" / "torch" / "sentence_transformers"

print("=== Clearing Model Cache ===\n")

# Find the E5 model cache
e5_models = list(cache_dir.glob("*e5*"))
if e5_models:
    print(f"Found {len(e5_models)} E5 model cache(s):")
    for model_path in e5_models:
        print(f"  - {model_path.name}")
        try:
            shutil.rmtree(model_path)
            print(f"    ✓ Deleted")
        except Exception as e:
            print(f"    ✗ Error: {e}")
else:
    print("No E5 model cache found")

# Clear torch cache
if torch_cache.exists():
    print(f"\nClearing torch cache: {torch_cache}")
    try:
        shutil.rmtree(torch_cache)
        print("  ✓ Deleted")
    except Exception as e:
        print(f"  ✗ Error: {e}")

print("\n=== Cache Cleared! ===")
print("Run your test again - it will download a fresh model.")