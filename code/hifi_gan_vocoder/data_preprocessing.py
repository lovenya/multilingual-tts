import os
import numpy as np
import torchaudio
from torchaudio.transforms import MelSpectrogram
from tqdm import tqdm

def preprocess_data(dataset_dir, sample_rate=22050, n_mels=80, n_fft=1024, hop_length=256, win_length=1024):
    """
    Preprocess the audio files in all subfolders of the dataset and save mel spectrograms
    in corresponding mel_spectrograms folders within each speaker directory.

    Parameters:
    - dataset_dir: Path to the root dataset directory that contains speaker subfolders
    - sample_rate: The sample rate of the audio (default: 22050)
    - n_mels: Number of mel bins (default: 80)
    - n_fft: Size of FFT (default: 1024)
    - hop_length: Hop length for STFT (default: 256)
    - win_length: Window length for STFT (default: 1024)
    """
    # Create mel transform
    mel_transform = MelSpectrogram(
        sample_rate=sample_rate,
        n_mels=n_mels,
        n_fft=n_fft,
        hop_length=hop_length,
        win_length=win_length,
        power=1.0,
        normalized=True
    )

    # Process each speaker directory
    for speaker_dir in os.listdir(dataset_dir):
        speaker_path = os.path.join(dataset_dir, speaker_dir)
        
        # Skip if not a directory
        if not os.path.isdir(speaker_path):
            continue
            
        # Look for wav directory
        wav_dir = os.path.join(speaker_path, 'wav')
        if not os.path.exists(wav_dir):
            print(f"No wav directory found in {speaker_path}, skipping...")
            continue

        # Create mel_spectrograms directory within speaker directory
        mel_dir = os.path.join(speaker_path, 'mel_spectrograms')
        os.makedirs(mel_dir, exist_ok=True)

        # Get wav files
        wav_files = [f for f in os.listdir(wav_dir) if f.endswith('.wav')]
        
        # Process each wav file
        for wav_file in tqdm(wav_files, desc=f"Processing {speaker_dir}"):
            try:
                # Load audio
                wav_path = os.path.join(wav_dir, wav_file)
                waveform, sr = torchaudio.load(wav_path)

                # Resample if needed
                if sr != sample_rate:
                    resampler = torchaudio.transforms.Resample(sr, sample_rate)
                    waveform = resampler(waveform)

                # Convert to mono if stereo
                if waveform.size(0) > 1:
                    waveform = waveform.mean(dim=0, keepdim=True)

                # Generate mel spectrogram
                mel_spec = mel_transform(waveform)

                # Save mel spectrogram
                mel_filename = wav_file.replace('.wav', '.npy')
                mel_path = os.path.join(mel_dir, mel_filename)
                np.save(mel_path, mel_spec.numpy())

            except Exception as e:
                print(f"Error processing {wav_path}: {str(e)}")

if __name__ == "__main__":
    dataset_dir = 'dataset'  # Root folder containing speaker subfolders
    preprocess_data(dataset_dir)