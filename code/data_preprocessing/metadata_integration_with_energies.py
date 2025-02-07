import os
import pandas as pd

# Path to the dataset
base_path = "dataset"
metadata_path = os.path.join(base_path, "metadata")

# Metadata files
metadata_files = ["updated_train.csv", "updated_validation.csv", "updated_test.csv"]

def get_energy_filepath(audio_filepath):
    """
    Constructs the energy file path based on the audio_filepath.
    Assumes the speaker folder is the third-to-last part and replaces ".wav" with ".npy".
    """
    file_parts = audio_filepath.split("/")
    speaker_folder = file_parts[-3]  # e.g., Bhojpuri_F
    file_name = file_parts[-1].replace(".wav", ".npy")  # Replace .wav with .npy
    energy_path = os.path.join(base_path, speaker_folder, "energies", file_name)
    return energy_path

# Process each metadata file for energy integration
for metadata_file in metadata_files:
    file_path = os.path.join(metadata_path, metadata_file)
    
    # Read the metadata file with correct encoding
    df = pd.read_csv(file_path, encoding="utf-8")
    
    # Remove existing column if it exists
    if "energy_filepath" in df.columns:
        df = df.drop(columns=["energy_filepath"])
    
    # Add the energy_filepath column
    df["energy_filepath"] = df["audio_filepath"].apply(get_energy_filepath)
    
    # Save the updated metadata file
    updated_file_path = os.path.join(metadata_path, f"{metadata_file}")
    df.to_csv(updated_file_path, index=False, encoding="utf-8-sig")

print("Energy file paths successfully added to the metadata files.")
