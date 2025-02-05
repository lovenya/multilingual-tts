import os
import re

def clean_phoneme_sequence(phoneme_text):
    """
    Clean a phoneme sequence by removing any text within parentheses,
    unwanted punctuation, and extra whitespace.
    """
    # Remove any text within parentheses (e.g., (english), (kn), etc.)
    cleaned = re.sub(r'\([^)]*\)', '', phoneme_text)
    # Remove punctuation (adjust the pattern if needed)
    cleaned = re.sub(r'[.,;:!?]', '', cleaned)
    # Remove extra whitespace and newlines
    cleaned = cleaned.strip()
    return cleaned

def iterate_and_clean_phoneme_files(root_folder):
    """
    Iterate over each dataset folder (excluding the 'metadata' folder) and
    automatically process each file in its 'phonemes' subfolder.
    The file is read, cleaned using clean_phoneme_sequence, and then overwritten.
    """
    for folder in os.listdir(root_folder):
        folder_path = os.path.join(root_folder, folder)
        if not os.path.isdir(folder_path):
            continue
        # Skip the metadata folder
        if folder.lower() == "metadata":
            continue

        # Locate the phonemes subfolder inside this dataset folder
        phoneme_folder = os.path.join(folder_path, "phonemes")
        if not os.path.exists(phoneme_folder):
            print(f"Phoneme folder not found in {folder_path}")
            continue

        # Process each text file in the phonemes folder
        for file in os.listdir(phoneme_folder):
            if file.endswith('.txt'):
                file_path = os.path.join(phoneme_folder, file)
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        content = f.read()
                    cleaned_content = clean_phoneme_sequence(content)
                    # Overwrite the file with the cleaned content
                    with open(file_path, 'w', encoding='utf-8') as f:
                        f.write(cleaned_content)
                    print(f"Cleaned: {file_path}")
                except Exception as e:
                    print(f"Error processing {file_path}: {e}")

if __name__ == "__main__":
    # Set the root folder of your dataset (this is the file folder structure of your dataset for the multilingual TTS project)
    dataset_root = "dataset"  # Adjust this path if needed
    iterate_and_clean_phoneme_files(dataset_root)
