#!/usr/bin/env python3
import argparse
from pathlib import Path
import re
import unicodedata

def clean_phoneme_sequence(sequence):
    """
    Cleans the phoneme sequence by:
      - Removing stress markers (ˈ and ˌ)
      - Normalizing Unicode to NFD and removing combining diacritics
      - Removing extraneous punctuation (commas, periods, semicolons, colons, exclamation and question marks)
        while preserving parentheses (for language markers)
      - Normalizing whitespace
    Args:
        sequence (str): Raw phoneme sequence.
    Returns:
        str: Cleaned phoneme sequence.
    """
    # Remove stress markers
    sequence = sequence.replace("ˈ", "").replace("ˌ", "")
    # Normalize Unicode (NFD)
    sequence = unicodedata.normalize("NFD", sequence)
    # Remove combining diacritical marks
    sequence = ''.join(ch for ch in sequence if not unicodedata.combining(ch))
    # Remove extraneous punctuation (except parentheses)
    sequence = re.sub(r"[.,;:!?]", "", sequence)
    # Normalize whitespace
    sequence = re.sub(r"\s+", " ", sequence)
    return sequence.strip()

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
        if not folder.is_dir() or folder.name.lower() == "metadata":
            continue
        phonemes_dir = folder / "phonemes"
        if not phonemes_dir.exists():
            print(f"Warning: {phonemes_dir} does not exist in {folder.name}.")
            continue
        for file_path in phonemes_dir.glob("*.txt"):
            with open(file_path, "r", encoding="utf-8") as f:
                content = f.read().strip()
                cleaned = clean_phoneme_sequence(content)
                tokens = cleaned.split()  # Split on whitespace
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
    
    print("Cleaned phoneme vocabulary size:", len(vocab))
    if args.print_tokens:
        print("Phoneme tokens:", sorted(vocab))

if __name__ == "__main__":
    main()
