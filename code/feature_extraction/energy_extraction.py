import argparse
from pathlib import Path
import numpy as np
import librosa

try:
    from tqdm import tqdm
except ModuleNotFoundError:
    tqdm = None

def nemo_extract_energy(file_path, sr=22050, frame_length=1024, hop_length=256):
    """
    Computes RMS energy frame-by-frame using librosa.
    
    Args:
        file_path (str): Path to the audio file.
        sr (int): Sampling rate (set to 22050 Hz).
        frame_length (int): Window length for RMS calculation.
        hop_length (int): Hop length for RMS calculation.
    
    Returns:
        np.ndarray: RMS energy per frame.
    """
    audio, _ = librosa.load(file_path, sr=sr)
    energy = librosa.feature.rms(y=audio, frame_length=frame_length, hop_length=hop_length)[0]
    return energy

def process_language_folder_energy(lang_folder: Path, sr=22050, frame_length=1024, hop_length=256):
    """
    Processes a single language/speaker folder by iterating over its WAV files,
    extracting energy, and saving them in a new subdirectory.
    
    Args:
        lang_folder (Path): Path to one language/speaker folder (e.g., dataset/English_M).
        sr (int): Sampling rate.
        frame_length (int): Frame length for energy extraction.
        hop_length (int): Hop length for energy extraction.
    """
    # Assume the audio files are in a subfolder named "wav".
    wav_dir = lang_folder / "wav"
    wav_files = list(wav_dir.glob("*.wav"))
    
    # Create target directory for energies.
    energies_dir = lang_folder / "energies"
    energies_dir.mkdir(exist_ok=True)
    
    if tqdm is not None:
        wav_files = tqdm(wav_files, desc=f"Processing {lang_folder.name} (Energy)")
    
    for wav_file in wav_files:
        basename = wav_file.stem
        energy_path = energies_dir / f"{basename}.npy"
        
        # Skip if energy is already extracted.
        if energy_path.exists():
            continue
        
        # Extract energy.
        energy = nemo_extract_energy(str(wav_file), sr=sr, frame_length=frame_length, hop_length=hop_length)
        
        np.save(energy_path, energy)
    
    print(f"Finished processing energy for folder: {lang_folder.name}")

def main():
    parser = argparse.ArgumentParser(
        description="Extract energy from all language folders in the dataset."
    )
    
    sr = 22050
    frame_length = 1024
    hop_length = 256
    
    dataset_path = Path("dataset")
    # Assume the "metadata" folder is in the dataset, so process all other folders.
    language_folders = [folder for folder in dataset_path.iterdir() 
                        if folder.is_dir() and folder.name.lower() != "metadata"]
    
    for lang_folder in language_folders:
        process_language_folder_energy(lang_folder, sr=sr, frame_length=frame_length, hop_length=hop_length)
    
    print("Finished processing all language folders for energy.")

if __name__ == "__main__":
    main()
