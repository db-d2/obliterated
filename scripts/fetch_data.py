"""Download every prompt set into data/raw and print set sizes (never prompt text)."""

from obliterated.data import fetch_all

if __name__ == "__main__":
    for name, n in fetch_all().items():
        print(f"{name:>24}: {n}")
