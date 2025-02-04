#!/usr/bin/env python3
import argparse
from pathlib import Path

def update_english_phoneme_sequences(dataset_dir: Path):
    """
    Updates English phoneme sequence files by adding the prefix "(en-us) ".
    Assumes that English phoneme sequences are stored in the 'phonemes' subfolder
    of folders named 'English_M' and 'English_F'.
    """
    # Define the folders that contain English phoneme sequences.
    english_folders = {"English_M", "English_F"}
    
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
                # If the file does not already start with (en-us), add it.
                if not content.startswith("(en-us)"):
                    new_content = "(en-us) " + content
                    with open(file_path, "w", encoding="utf-8") as f:
                        f.write(new_content)
                    print(f"Updated {file_path} with (en-us) prefix.")

def main():
    parser = argparse.ArgumentParser(
        description="Add (en-us) prefix to English phoneme sequences."
    )
    dataset_dir = "dataset"
    
    dataset_path = Path(dataset_dir)
    update_english_phoneme_sequences(dataset_path)
    print("Finished updating English phoneme sequences.")

if __name__ == "__main__":
    main()
