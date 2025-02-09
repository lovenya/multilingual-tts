import os
import numpy as np
import torchaudio
from torchaudio.transforms import MelSpectrogram
from tqdm import tqdm

def preprocess_data(dataset_dir, mel_dir, sample_rate=22050, n_mels=80, hop_length=256, win_length=1024):
    """
    Preprocess the audio files in all subfolders of the dataset and save mel spectrograms.

    Parameters:
    - dataset_dir: Path to the root dataset directory that contains language and speaker subfolders
    - mel_dir: Path to save the extracted mel spectrograms
    - sample_rate: The sample rate of the audio
    - n_mels: Number of mel bins
    - hop_length: Hop length for STFT
    - win_length: Window length for STFT
    """
    if not os.path.exists(mel_dir):
        os.makedirs(mel_dir)

    mel_transform = MelSpectrogram(sample_rate=sample_rate, n_mels=n_mels, hop_length=hop_length, win_length=win_length)

    # Walk through all subfolders in the dataset directory
    for root, dirs, files in os.walk(dataset_dir):
        # Process only the audio files (e.g., .wav)
        audio_files = [f for f in files if f.endswith('.wav')]
        
        for audio_file in tqdm(audio_files, desc=f"Processing {root}"):
            # Load audio
            audio_filepath = os.path.join(root, audio_file)
            waveform, sr = torchaudio.load(audio_filepath)

            # Check if sample rate matches
            assert sr == sample_rate, f"Sample rate mismatch: {sr} != {sample_rate}"

            # Apply MelSpectrogram transformation
            mel_spec = mel_transform(waveform)

            # Save the mel spectrogram as a .npy file
            mel_filepath = os.path.join(mel_dir, root.replace(dataset_dir, '').strip(os.sep).replace(os.sep, '_') + f'_{audio_file.replace(".wav", ".npy")}')
            os.makedirs(os.path.dirname(mel_filepath), exist_ok=True)
            np.save(mel_filepath, mel_spec.numpy())

if __name__ == "__main__":
    dataset_dir = 'dataset'  # Update this path
    mel_dir = 'dataset/mel_spectograms_for_hifi_gan'  # Update this path
    preprocess_data(dataset_dir, mel_dir)
