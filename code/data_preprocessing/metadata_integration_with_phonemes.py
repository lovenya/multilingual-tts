import os
import pandas as pd

# Path to the dataset
base_path = "dataset"
metadata_path = os.path.join(base_path, "metadata")

# Metadata files
metadata_files = ["train.csv", "validation.csv", "test.csv"]

# Function to read phoneme sequence correctly
def get_phoneme_sequence(audio_filepath):
    # Extract the relevant parts of the path to locate the phoneme file
    file_parts = audio_filepath.split("/")
    speaker_folder = file_parts[-3]  # E.g., Bhojpuri_F
    file_name = file_parts[-1].replace(".wav", ".txt")  # Replace .wav with .txt

    # Construct the phoneme file path
    phoneme_path = os.path.join(base_path, speaker_folder, "phonemes", file_name)

    try:
        # Read the phoneme file with correct encoding
        with open(phoneme_path, "r", encoding="utf-8") as f:
            return f.read().strip()
    except FileNotFoundError:
        return ""  # Return an empty string if the phoneme file is missing

# Process each metadata file
for metadata_file in metadata_files:
    file_path = os.path.join(metadata_path, metadata_file)
    
    # Read the metadata file with correct encoding
    df = pd.read_csv(file_path, encoding="utf-8")
    
    # Ensure no changes to existing columns
    if "phoneme_sequence" in df.columns:
        df = df.drop(columns=["phoneme_sequence"])  # Remove if already exists
    
    # Add the phoneme_sequence column
    df["phoneme_sequence"] = df["audio_filepath"].apply(get_phoneme_sequence)
    
    # Save the updated metadata file with correct encoding
    updated_file_path = os.path.join(metadata_path, f"updated_{metadata_file}")
    df.to_csv(updated_file_path, index=False, encoding="utf-8-sig")

print("Phoneme sequences successfully added to the metadata files.")
