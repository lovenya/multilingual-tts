#!/usr/bin/env python3
import argparse
from pathlib import Path

def compute_phoneme_vocab(dataset_dir: Path):
    """
    Computes the phoneme vocabulary from all phoneme sequence files in the dataset.
    Assumes that phoneme sequences are stored in the 'phonemes' subfolder of each language folder.
    Excludes the 'metadata' folder.
    
    Returns:
        vocab (set): The set of unique phoneme tokens.
    """
    vocab = set()
    for folder in dataset_dir.iterdir():
        # Skip if not a folder or if folder name is 'metadata'
        if not folder.is_dir() or folder.name.lower() == "metadata":
            continue
        phoneme_dir = folder / "phonemes"
        if not phoneme_dir.exists():
            print(f"Warning: {phoneme_dir} does not exist in {folder.name}.")
            continue
        # Read all .txt files in the phonemes folder.
        for file_path in phoneme_dir.glob("*.txt"):
            with open(file_path, "r", encoding="utf-8") as f:
                # Assuming tokens are space-separated.
                tokens = f.read().strip().split()
                vocab.update(tokens)
    return vocab

def main():
    parser = argparse.ArgumentParser(
        description="Compute the phoneme vocabulary from the dataset phoneme sequences."
    )
  
    parser.add_argument("--print_tokens", action="store_true",
                        help="Also print the full list of phoneme tokens.")
    args = parser.parse_args()
    
    dataset_path = Path("dataset")
    vocab = compute_phoneme_vocab(dataset_path)
    
    print("Phoneme vocabulary size:", len(vocab))
    if args.print_tokens:
        print("Phoneme tokens:", sorted(vocab))

if __name__ == "__main__":
    main()
