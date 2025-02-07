import torch
import librosa
import numpy as np
import torchaudio
import matplotlib.pyplot as plt

def verify_alignment(audio_path, pitch_path, energy_path, sr=16000, n_fft=1024, hop_length=256, n_mels=80):
    """
    Computes the mel-spectrogram on the fly for a given audio file and compares its frame count
    with the lengths of the pitch and energy arrays.
    
    Args:
        audio_path (str): Path to the audio file.
        pitch_path (str): Path to the corresponding pitch .npy file.
        energy_path (str): Path to the corresponding energy .npy file.
        sr (int): Sampling rate.
        n_fft (int): FFT window size.
        hop_length (int): Hop length for mel computation.
        n_mels (int): Number of mel filter banks.
    """
    # Load audio
    audio, _ = librosa.load(audio_path, sr=sr)
    
    # Compute mel-spectrogram using torchaudio (can also use librosa if desired)
    mel_transform = torchaudio.transforms.MelSpectrogram(
        sample_rate=sr,
        n_fft=n_fft,
        hop_length=hop_length,
        n_mels=n_mels
    )
    # Convert to tensor and compute mel spectrogram
    mel_spec = mel_transform(torch.tensor(audio).unsqueeze(0)).squeeze(0).numpy()  # shape: [n_mels, T]
    
    # Load pitch and energy arrays
    pitch = np.load(pitch_path)
    energy = np.load(energy_path)
    
    print("Mel-spectrogram shape (n_mels, T):", mel_spec.shape)
    print("Pitch array shape:", pitch.shape)
    print("Energy array shape:", energy.shape)
    
    # Optionally, plot for visual inspection
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 3, 1)
    plt.imshow(mel_spec, aspect='auto', origin='lower')
    plt.title("Mel-Spectrogram")
    plt.subplot(1, 3, 2)
    plt.plot(pitch)
    plt.title("Pitch (F0)")
    plt.subplot(1, 3, 3)
    plt.plot(energy)
    plt.title("Energy")
    plt.tight_layout()
    plt.show()

verify_alignment("dataset/English_F/wav/En_F_00100.wav", "dataset/English_F/pitches/En_F_00100.npy", "dataset/English_F/energies/En_F_00100.npy")