
from pathlib import Path
import runpy
import sys

PROJECT_ROOT = Path(__file__).resolve().parent
LAB6_SCRIPT = PROJECT_ROOT / "src" / "register_model.py"


def main():
    print("=" * 55)
    print("       LAB 6: CACAO RATING MODEL REGISTRY")
    print("=" * 55)

    if not LAB6_SCRIPT.exists():
        print(f"[ERROR] Script not found: {LAB6_SCRIPT}")
        sys.exit(1)

    print("[INFO] Running Lab 6 registration...")
    runpy.run_path(str(LAB6_SCRIPT), run_name="__main__")


if __name__ == "__main__":
    main()
