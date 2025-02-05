import os
import re

def get_unified_prefix(folder_name):
    """
    Determines the unified prefix based on the folder's language.
    Expects folder names like 'English_M', 'Bhojpuri_F', etc.
    """
    mapping = {
        "english": "(en-us)",
        "bhojpuri": "(hi)",
        "gujarathi": "(gu)",
        "kannada": "(kn)",
    }
    # Extract language part (assumes the language name is the first part before '_')
    language = folder_name.split("_")[0].lower()
    return mapping.get(language, "")

def fix_phoneme_sequence(phoneme_text, unified_prefix):
    """
    Removes any existing tags and unwanted punctuation,
    then prepends the unified prefix.
    """
    # Remove any text inside parentheses (e.g., extra language tags)
    cleaned = re.sub(r'\([^)]*\)', '', phoneme_text)
    # Remove punctuation if necessary (adjust as needed)
    cleaned = re.sub(r'[.,;:!?]', '', cleaned)
    # Remove extra whitespace/newlines
    cleaned = cleaned.strip()
    # Prepend the unified prefix and a space (if desired)
    fixed_sequence = f"{unified_prefix} {cleaned}"
    return fixed_sequence

def build_inventory_from_dataset(root_folder):
    """
    Iterates over each dataset folder (excluding 'metadata'),
    fixes phoneme sequences by applying unified prefixes, and
    builds a unified inventory of phoneme symbols.
    """
    unified_inventory = set()

    # Process each folder in the root directory
    for folder in os.listdir(root_folder):
        folder_path = os.path.join(root_folder, folder)
        if not os.path.isdir(folder_path):
            continue
        # Skip metadata folder
        if folder.lower() == "metadata":
            continue

        # Determine the unified prefix based on the folder name
        unified_prefix = get_unified_prefix(folder)
        phoneme_folder = os.path.join(folder_path, "phonemes")
        if not os.path.exists(phoneme_folder):
            print(f"Phoneme folder not found in {folder_path}")
            continue

        # Process each phoneme text file
        for file in os.listdir(phoneme_folder):
            if file.endswith('.txt'):
                file_path = os.path.join(phoneme_folder, file)
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        content = f.read()
                    # Fix the phoneme sequence
                    fixed_sequence = fix_phoneme_sequence(content, unified_prefix)
                    # Optionally, you could write the fixed sequence back to the file here.
                    # For the inventory, we split the fixed sequence into tokens.
                    tokens = fixed_sequence.split()
                    unified_inventory.update(tokens)
                except Exception as e:
                    print(f"Error processing {file_path}: {e}")

    return unified_inventory

if __name__ == "__main__":
    # Set the root folder for your dataset
    dataset_root = "dataset"  # Modify if necessary
    inventory = build_inventory_from_dataset(dataset_root)
    
    output_file = "unified_inventory.txt"
    with open(output_file, 'w', encoding='utf-8') as f:
        for phoneme in sorted(inventory):
            f.write(phoneme + "\n")
    print(f"Unified inventory saved to {output_file}")

    
    
