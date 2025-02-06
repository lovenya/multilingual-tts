import argparse
from pathlib import Path
import numpy as np
import librosa
import pyworld
from concurrent.futures import ProcessPoolExecutor
from tqdm import tqdm


def nemo_extract_pitch(file_path, sr=16000, frame_period=16.0):
    """
    Extracts pitch (F0) using pyworld.dio and pyworld.stonemask.
    
    Args:
        file_path (str): Path to the audio file.
        sr (int): Sampling rate (set to 16000 Hz).
        frame_period (float): Frame period in ms (default: 5.0 ms).
    
    Returns:
        f0 (np.ndarray): Extracted fundamental frequency contour.
        time_axis (np.ndarray): Time stamps for each frame.
    """
    audio, _ = librosa.load(file_path, sr=sr)
    audio = audio.astype(np.float64)
    f0, time_axis = pyworld.dio(audio, sr, frame_period=frame_period)
    f0 = pyworld.stonemask(audio, f0, time_axis, sr)
    return f0, time_axis

def process_language_folder(lang_folder: Path, sr=16000, frame_period=16.0):
    """
    Processes a single language/speaker folder by iterating over its WAV files,
    extracting pitch, and saving them in a new subdirectory.
    
    Args:
        lang_folder (Path): Path to one language/speaker folder (e.g., dataset/English_M).
        sr (int): Sampling rate.
        frame_period (float): Frame period in ms.
    """
    wav_dir = lang_folder / "wav"
    wav_files = list(wav_dir.glob("*.wav"))
    
    # Create target directory for pitch files.
    pitches_dir = lang_folder / "pitches"
    pitches_dir.mkdir(exist_ok=True)
    
    if tqdm is not None:
        wav_files = tqdm(wav_files, desc=f"Processing {lang_folder.name} (Pitch)")
    
    for wav_file in wav_files:
        basename = wav_file.stem
        pitch_path = pitches_dir / f"{basename}.npy"
        if pitch_path.exists():
            continue
        f0, _ = nemo_extract_pitch(str(wav_file), sr=sr, frame_period=frame_period)
        np.save(pitch_path, f0)
    
    print(f"Finished processing pitch for folder: {lang_folder.name}")

def main():
  
    sr = 16000
    hop_length = 256
    frame_period = 16.0
    # frame_period = hop_length / sr * 1000  # in ms, where hop_length is the number of samples and sr is the sample rate

    
    dataset_path = Path("dataset")
    # Process all subdirectories except for "metadata"
    language_folders = [folder for folder in dataset_path.iterdir() 
                        if folder.is_dir() and folder.name.lower() != "metadata"]
    
    # Use ProcessPoolExecutor to process language folders in parallel.
    with ProcessPoolExecutor(max_workers=len(language_folders)) as executor:
        futures = [executor.submit(process_language_folder, lang_folder,
                                   sr=sr, frame_period=frame_period)
                   for lang_folder in language_folders]
        # Optionally, wait for all to complete:
        for future in futures:
            future.result()
    
    print("Finished processing all language folders for pitch.")

if __name__ == "__main__":
    main()
