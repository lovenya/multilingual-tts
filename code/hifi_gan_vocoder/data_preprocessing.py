import os
import numpy as np
import torchaudio
from torchaudio.transforms import MelSpectrogram
from tqdm import tqdm

def preprocess_data(dataset_dir, mel_dir, sample_rate=22050, n_mels=80, n_fft=1024, hop_length=256, win_length=None):
    """
    Preprocess the audio files in all subfolders of the dataset and save mel spectrograms.

    Parameters:
    - dataset_dir: Path to the root dataset directory
    - mel_dir: Path to save the extracted mel spectrograms
    - sample_rate: The sample rate of the audio (default: 22050)
    - n_mels: Number of mel bins (default: 80)
    - n_fft: Size of FFT (default: 1024)
    - hop_length: Hop length for STFT (default: 256)
    - win_length: Window length for STFT (default: None, will use n_fft)
    """
    if not os.path.exists(mel_dir):
        os.makedirs(mel_dir)

    # Set win_length to n_fft if not specified
    if win_length is None:
        win_length = n_fft

    mel_transform = MelSpectrogram(
        sample_rate=sample_rate,
        n_mels=n_mels,
        n_fft=n_fft,
        hop_length=hop_length,
        win_length=win_length,
        power=1.0,  # Use amplitude instead of power
        normalized=True,  # Normalize by scale
        center=True,  # Pad for centering
        pad_mode='reflect'  # Use reflection padding
    )

    # Walk through all subfolders in the dataset directory
    for root, dirs, files in os.walk(dataset_dir):
        # Process only the audio files (e.g., .wav)
        audio_files = [f for f in files if f.endswith('.wav')]
        
        if not audio_files:
            continue
            
        # Create output directory structure
        rel_path = os.path.relpath(root, dataset_dir)
        out_dir = os.path.join(mel_dir, rel_path)
        os.makedirs(out_dir, exist_ok=True)
        
        for audio_file in tqdm(audio_files, desc=f"Processing {rel_path}"):
            try:
                # Load audio
                audio_filepath = os.path.join(root, audio_file)
                waveform, sr = torchaudio.load(audio_filepath)

                # Resample if needed
                if sr != sample_rate:
                    resampler = torchaudio.transforms.Resample(sr, sample_rate)
                    waveform = resampler(waveform)

                # Convert to mono if stereo
                if waveform.size(0) > 1:
                    waveform = waveform.mean(dim=0, keepdim=True)

                # Apply MelSpectrogram transformation
                mel_spec = mel_transform(waveform)

                # Save the mel spectrogram
                mel_filename = os.path.splitext(audio_file)[0] + '.npy'
                mel_filepath = os.path.join(out_dir, mel_filename)
                np.save(mel_filepath, mel_spec.numpy())

            except Exception as e:
                print(f"Error processing {audio_filepath}: {str(e)}")

if __name__ == "__main__":
    dataset_dir = 'dataset'
    mel_dir = 'dataset/mel_spectrograms_for_hifi_gan'  # Fixed typo in directory name
    
    preprocess_data(
        dataset_dir=dataset_dir,
        mel_dir=mel_dir,
        sample_rate=22050,
        n_mels=80,
        n_fft=1024,
        hop_length=256,
        win_length=1024
    )