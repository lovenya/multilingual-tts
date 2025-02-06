# this script is meant to clean phoneme sequences
# this will also, after cleaning, extract phoneme sequences, will compare against our inventory
# "set" ensures no duplicates


import os
import re
import sys

# Add the path to 'util' to the system path to allow importing
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../util')))

# Import the get_fixed_inventory function
from generate_phoneme_inventory import get_fixed_inventory

def clean_phoneme_sequence(phoneme_text):
    """
    Clean the phoneme sequence by removing unwanted markers, stress, diacritics,
    and any non-phonemic symbols like punctuation or language tags.
    """
    # Remove any content within parentheses (e.g., language tags or extra markers)
    cleaned = re.sub(r'\([^)]*\)', '', phoneme_text)
    # Remove stress markers like ˈ, ˌ and other diacritical marks
    cleaned = re.sub(r'[ˈˌᵻʌːʰʳː]', '', cleaned)  # This removes common stress markers and diacritics
    # Remove punctuation (e.g., commas, periods)
    cleaned = re.sub(r'[.,;:!?]', '', cleaned)
    
    # Remove extra whitespace and newlines
    cleaned = cleaned.strip()
    return cleaned

def extract_phonemes_from_dataset(root_folder):
    """
    Extract unique phonemes from the phoneme files in the dataset,
    ensuring that duplicates are removed using a set.
    """
    extracted_phonemes = set()  # Using a set to automatically deduplicate phonemes

    # Iterate over the dataset folders (excluding 'metadata')
    for folder in os.listdir(root_folder):
        folder_path = os.path.join(root_folder, folder)
        if not os.path.isdir(folder_path):
            continue
        if folder.lower() == "metadata":
            continue

        phoneme_folder = os.path.join(folder_path, "phonemes")
        if not os.path.exists(phoneme_folder):
            continue

        # Process each phoneme text file in the phonemes folder
        for file in os.listdir(phoneme_folder):
            if file.endswith('.txt'):
                file_path = os.path.join(phoneme_folder, file)
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        content = f.read()
                    cleaned_content = clean_phoneme_sequence(content)
                    tokens = cleaned_content.split()
                    extracted_phonemes.update(tokens)  # Add tokens to set to ensure uniqueness
                except Exception as e:
                    print(f"Error processing {file_path}: {e}")
    return extracted_phonemes

def compare_with_inventory(extracted_phonemes, fixed_inventory):
    """
    Compare the phonemes from the dataset with the fixed inventory.
    """
    missing_phonemes = extracted_phonemes - set(fixed_inventory)
    extra_phonemes = set(fixed_inventory) - extracted_phonemes

    print(f"Phonemes in dataset but not in inventory ({len(missing_phonemes)}):")
    for phoneme in missing_phonemes:
        print(phoneme)

    print(f"\nPhonemes in inventory but not in dataset ({len(extra_phonemes)}):")
    for phoneme in extra_phonemes:
        print(phoneme)

if __name__ == "__main__":
    # Set the root folder for your dataset
    dataset_root = "dataset"  # Modify if necessary

    # Define your fixed phoneme inventory by calling the imported function
    fixed_inventory = get_fixed_inventory()  # This now calls the function from util/generate_phoneme_inventory.py
    
    # Extract phonemes from your dataset
    extracted_phonemes = extract_phonemes_from_dataset(dataset_root)

    # Compare the extracted phonemes with the fixed inventory
    compare_with_inventory(extracted_phonemes, fixed_inventory)
    
    # Print the count of unique phonemes
    print(f"\nTotal unique phonemes in dataset: {len(extracted_phonemes)}")
