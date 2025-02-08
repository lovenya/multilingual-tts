import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
import torchaudio
import pandas as pd
import torch.nn.utils.rnn as rnn_utils

from data_preprocessing.generate_phoneme_inventory import get_fixed_inventory

def compute_mel(wav_path, sr=16000, n_fft=1024, hop_length=256, n_mels=80):
    """
    Computes the mel-spectrogram on the fly using torchaudio.
    
    Args:
        wav_path (str): Path to the WAV file.
        sr (int): Target sampling rate.
        n_fft (int): FFT window size.
        hop_length (int): Hop length.
        n_mels (int): Number of mel channels.
    
    Returns:
        torch.Tensor: Mel-spectrogram tensor of shape (n_mels, T).
    """
    waveform, sample_rate = torchaudio.load(wav_path)
    if sample_rate != sr:
        resampler = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=sr)
        waveform = resampler(waveform)
    mel_transform = torchaudio.transforms.MelSpectrogram(
        sample_rate=sr,
        n_fft=n_fft,
        hop_length=hop_length,
        n_mels=n_mels
    )
    mel_spec = mel_transform(waveform)  # shape: (1, n_mels, T)
    return mel_spec.squeeze(0)  # (n_mels, T)

class TTSDataset(Dataset):
    def __init__(self, root_dir, metadata_csv, phoneme_vocab, language_map, speaker_map, sr=16000):
        """
        Args:
            root_dir (str): Root directory of the dataset.
            metadata_csv (str): Path to the CSV file containing metadata.
            phoneme_vocab (dict): Mapping from phoneme tokens to indices.
            language_map (dict): Mapping from language names to integer IDs.
            speaker_map (dict): Mapping from speaker IDs (e.g., "english_f", etc.) to integer IDs.
            sr (int): Sampling rate.
        """
        self.root_dir = root_dir
        self.metadata = pd.read_csv(metadata_csv, encoding="utf-8-sig")
        self.phoneme_vocab = phoneme_vocab
        self.language_map = language_map
        self.speaker_map = speaker_map
        self.sr = sr

    def __len__(self):
        return len(self.metadata)

    def __getitem__(self, idx):
        row = self.metadata.iloc[idx]
        file_id = row['file_id']   # Base filename (without extension)
        language = row['language'].lower()
        speaker = row['speaker_id'].lower()  # e.g., "english_f"
        folder = row['speaker_id']  # Speaker/language folder name
        
        # Construct file paths
        sample_folder = os.path.join(self.root_dir, folder)
        phoneme_path = os.path.join(sample_folder, "phonemes", f"{file_id}.txt")
        wav_path = os.path.join(sample_folder, "wav", f"{file_id}.wav")
        pitch_path = os.path.join(sample_folder, "pitches", f"{file_id}.npy")
        energy_path = os.path.join(sample_folder, "energies", f"{file_id}.npy")
        
        # Load phoneme sequence (assumes space-separated tokens)
        with open(phoneme_path, 'r', encoding='utf-8') as f:
            phoneme_seq_str = f.read().strip()
        phoneme_tokens = phoneme_seq_str.split()
        phoneme_ids = [self.phoneme_vocab.get(token, self.phoneme_vocab.get("<unk>", 0)) for token in phoneme_tokens]
        phoneme_ids = torch.tensor(phoneme_ids, dtype=torch.long)
        
        # Compute mel-spectrogram on the fly
        mel_spec = compute_mel(wav_path, sr=self.sr)
        
        # Load pitch and energy targets from .npy files
        pitch = torch.tensor(np.load(pitch_path), dtype=torch.float)
        energy = torch.tensor(np.load(energy_path), dtype=torch.float)
        
        # Get language and speaker IDs
        language_id = torch.tensor(self.language_map[language], dtype=torch.long)
        speaker_id = torch.tensor(self.speaker_map[speaker], dtype=torch.long)
        
        return phoneme_ids, mel_spec, pitch, energy, speaker_id, language_id

def dynamic_collate_fn(batch):
    """
    Custom collate function for dynamic batching of variable-length sequences.
    Pads phoneme sequences, mel-spectrograms, pitch, and energy to the maximum length in the batch.
    
    Each item in batch is:
      (phoneme_ids, mel_spec, pitch, energy, speaker_id, language_id)
    """
    # Sort batch by length of phoneme sequence (descending)
    batch.sort(key=lambda x: len(x[0]), reverse=True)
    phoneme_seqs, mel_specs, pitches, energies, speaker_ids, language_ids = zip(*batch)
    
    # Pad phoneme sequences
    phoneme_seqs_padded = rnn_utils.pad_sequence(phoneme_seqs, batch_first=True, padding_value=0)
    
    # Pad mel-spectrograms: assume mel_spec shape is (n_mels, T); pad along T.
    mel_specs_padded = rnn_utils.pad_sequence([spec.t() for spec in mel_specs], batch_first=True, padding_value=0)
    mel_specs_padded = mel_specs_padded.transpose(1, 2)  # (B, n_mels, T)
    
    # Pad pitch and energy arrays along time dimension.
    pitches_padded = rnn_utils.pad_sequence(pitches, batch_first=True, padding_value=0)
    energies_padded = rnn_utils.pad_sequence(energies, batch_first=True, padding_value=0)
    
    # Stack speaker and language IDs (scalars per sample)
    speaker_ids = torch.stack(speaker_ids)
    language_ids = torch.stack(language_ids)
    
    return phoneme_seqs_padded, mel_specs_padded, pitches_padded, energies_padded, speaker_ids, language_ids

def build_phoneme_vocab():
    fixed_inventory = get_fixed_inventory()  # Your function from the inventory file
    phoneme_vocab = {token: idx for idx, token in enumerate(fixed_inventory)}
    return phoneme_vocab

# Example usage:
if __name__ == '__main__':
    # Example mappings (replace with your actual mappings)
    phoneme_vocab = build_phoneme_vocab()  # Extend to your full phoneme vocabulary (from fixed inventory)
    language_map = {"english": 0, "gujarathi": 1, "bhojpuri": 2, "kannada": 3}
    speaker_map = {"english_f": 0, "english_m": 1, "bhojpuri_f": 2, "bhojpuri_m": 3,
                   "gujarathi_f": 4, "gujarathi_m": 5, "kannada_f": 6, "kannada_m": 7}
    
    # Path to metadata CSV (update path as needed)
    metadata_csv = "dataset/metadata/updated_train.csv"
    dataset = TTSDataset(root_dir="dataset", metadata_csv=metadata_csv,
                         phoneme_vocab=phoneme_vocab, language_map=language_map, speaker_map=speaker_map)
    
    # Create weighted sampling based on language.
    df = pd.read_csv(metadata_csv, encoding="utf-8-sig")
    weights = []
    for _, row in df.iterrows():
        lang = row['language'].lower()
        if lang in ["gujarathi", "bhojpuri"]:
            weights.append(2.0)
        else:
            weights.append(1.0)
    weights = torch.tensor(weights, dtype=torch.float)
    sampler = WeightedRandomSampler(weights, num_samples=len(weights), replacement=True)
    
    data_loader = DataLoader(dataset, batch_size=8, collate_fn=dynamic_collate_fn, sampler=sampler)
    
    for batch in data_loader:
        phoneme_seqs, mel_specs, pitches, energies, speaker_ids, language_ids = batch
        print("Phoneme batch shape:", phoneme_seqs.shape)
        print("Mel-spectrogram batch shape:", mel_specs.shape)
        break
