import argparse
from pathlib import Path
import numpy as np
import librosa
import torchaudio
import torchaudio.transforms as T
import torch
from concurrent.futures import ProcessPoolExecutor

try:
    from tqdm import tqdm
except ModuleNotFoundError:
    tqdm = None

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def torchaudio_extract_pitch(file_path, sr=16000, frame_period=5.0):
    """
    Extracts pitch using torchaudio's Crepe model, leveraging GPU if available.
    
    Args:
        file_path (str): Path to the audio file.
        sr (int): Sampling rate.
        frame_period (float): Frame period in ms.
    
    Returns:
        f0 (np.ndarray): Extracted fundamental frequency contour.
    """
    audio, _ = librosa.load(file_path, sr=sr)
    audio_tensor = torch.tensor(audio).unsqueeze(0).to(DEVICE)

    # Use torchaudio's crepe model (GPU acceleration)
    pitch_extractor = T.PitchShift(sr, n_steps=0).to(DEVICE)
    f0 = pitch_extractor(audio_tensor).cpu().numpy().squeeze()

    return f0

def process_language_folder_pitch(lang_folder: Path, sr=16000, frame_period=5.0):
    """
    Processes a single language/speaker folder by iterating over WAV files,
    extracting pitch using GPU acceleration, and saving results.
    """
    wav_dir = lang_folder / "wav"
    wav_files = list(wav_dir.glob("*.wav"))
    pitches_dir = lang_folder / "pitches"
    pitches_dir.mkdir(exist_ok=True)

    if tqdm is not None:
        wav_files = tqdm(wav_files, desc=f"Processing {lang_folder.name} (Pitch)")

    for wav_file in wav_files:
        basename = wav_file.stem
        pitch_path = pitches_dir / f"{basename}.npy"

        if pitch_path.exists():
            continue

        f0 = torchaudio_extract_pitch(str(wav_file), sr=sr, frame_period=frame_period)
        np.save(pitch_path, f0)

    print(f"Finished processing pitch for folder: {lang_folder.name}")

def main():
    parser = argparse.ArgumentParser(
        description="Extract pitch from all language folders in the dataset using GPU."
    )

    sr = 16000
    frame_period = 5.0
    dataset_path = Path("dataset")

    # Exclude metadata folder
    language_folders = [folder for folder in dataset_path.iterdir() 
                        if folder.is_dir() and folder.name.lower() != "metadata"]

    # Use multiprocessing to process multiple folders in parallel
    with ProcessPoolExecutor(max_workers=4) as executor:
        executor.map(process_language_folder_pitch, language_folders)

    print("Finished processing all language folders for pitch.")

if __name__ == "__main__":
    main()
