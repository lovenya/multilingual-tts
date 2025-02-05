#!/usr/bin/env python3
import argparse
from pathlib import Path

def update_english_phoneme_sequences(dataset_dir: Path):
    """
    Updates English phoneme sequence files by adding the prefix "(hi) ".
    Assumes that English phoneme sequences are stored in the 'phonemes' subfolder
    of folders named 'English_M' and 'English_F'.
    """
    # Define the folders that contain English phoneme sequences.
    english_folders = {"Bhojpuri_M", "Bhojpuri_F"}
    
    for folder in dataset_dir.iterdir():
        if folder.is_dir() and folder.name in english_folders:
            phoneme_dir = folder / "phonemes"
            if not phoneme_dir.exists():
                print(f"Warning: {phoneme_dir} does not exist.")
                continue
            # Process all text files in the phonemes folder.
            for file_path in phoneme_dir.glob("*.txt"):
                with open(file_path, "r", encoding="utf-8") as f:
                    content = f.read().strip()
                # If the file does not already start with (hi), add it.
                if not content.startswith("(hi)"):
                    new_content = "(hi) " + content
                    with open(file_path, "w", encoding="utf-8") as f:
                        f.write(new_content)
                    print(f"Updated {file_path} with (hi) prefix.")

def main():
    
    dataset_dir = "dataset"
    
    dataset_path = Path(dataset_dir)
    update_english_phoneme_sequences(dataset_path)
    print("Finished updating Bhojpuri phoneme sequences.")

if __name__ == "__main__":
    main()
