import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
import torchaudio
import pandas as pd
import torch.nn.utils.rnn as rnn_utils
# from data_preprocessing.generate_phoneme_inventory import get_fixed_inventory
# facing import issues, need to be rsolved, for now i am directly using the phoneme inventory, pasting the code here only

def get_fixed_inventory():
    """
    Returns a fixed, unified phoneme inventory for the four languages,
    with unified prefixes:
      - English: (en-us)
      - Bhojpuri: (hi)
      - Gujarati: (gu)
      - Kannada: (kn)
    
    Note: This is an example inventory. You may refine it based on your needs.
    """
    inventory = [
        # English (en-us)
        "(en-us) p", "(en-us) b", "(en-us) t", "(en-us) d", "(en-us) k", "(en-us) g",
        "(en-us) f", "(en-us) v", "(en-us) θ", "(en-us) ð", "(en-us) s", "(en-us) z",
        "(en-us) ʃ", "(en-us) ʒ", "(en-us) h", "(en-us) tʃ", "(en-us) dʒ", "(en-us) m",
        "(en-us) n", "(en-us) ŋ", "(en-us) l", "(en-us) r", "(en-us) j", "(en-us) w",
        "(en-us) i", "(en-us) ɪ", "(en-us) e", "(en-us) ɛ", "(en-us) æ", "(en-us) ʌ",
        "(en-us) ɑ", "(en-us) ɒ", "(en-us) ɔ", "(en-us) o", "(en-us) ʊ", "(en-us) u",
        "(en-us) aɪ", "(en-us) aʊ", "(en-us) ɔɪ", "(en-us) eɪ", "(en-us) oʊ",
        
        # Bhojpuri (hi) – using a Hindi-like inventory
        "(hi) p", "(hi) b", "(hi) t̪", "(hi) d̪", "(hi) ʈ", "(hi) ɖ", "(hi) k", "(hi) g",
        "(hi) tʃ", "(hi) dʒ", "(hi) f", "(hi) s", "(hi) h", "(hi) m", "(hi) n",
        "(hi) ɳ", "(hi) n̪", "(hi) l", "(hi) r", "(hi) j",
        "(hi) ə", "(hi) a", "(hi) ɪ", "(hi) i", "(hi) ʊ", "(hi) u",
        "(hi) e", "(hi) o", "(hi) ɛ", "(hi) ɔ", "(hi) ɒ",
        
        # Gujarati (gu) – similar to Hindi/Bhojpuri
        "(gu) p", "(gu) b", "(gu) t̪", "(gu) d̪", "(gu) ʈ", "(gu) ɖ", "(gu) k", "(gu) g",
        "(gu) tʃ", "(gu) dʒ", "(gu) f", "(gu) s", "(gu) h", "(gu) m", "(gu) n",
        "(gu) ɳ", "(gu) n̪", "(gu) l", "(gu) r", "(gu) j",
        "(gu) ə", "(gu) a", "(gu) ɪ", "(gu) i", "(gu) ʊ", "(gu) u",
        "(gu) e", "(gu) o", "(gu) ɛ", "(gu) ɔ", "(gu) ɒ",
        
        # Kannada (kn) – similar to the above but may have slight differences
        "(kn) p", "(kn) b", "(kn) t", "(kn) d", "(kn) ʈ", "(kn) ɖ", "(kn) k", "(kn) g",
        "(kn) tʃ", "(kn) dʒ", "(kn) f", "(kn) s", "(kn) h", "(kn) m", "(kn) n",
        "(kn) ɳ", "(kn) n̪", "(kn) l", "(kn) r", "(kn) j",
        "(kn) ə", "(kn) a", "(kn) ɪ", "(kn) i", "(kn) ʊ", "(kn) u",
        "(kn) e", "(kn) o", "(kn) ɛ", "(kn) ɔ", "(kn) ɒ",
    ]
    return inventory






def compute_mel(wav_path, sr=22050, n_fft=1024, hop_length=256, n_mels=80):
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
    def __init__(self, root_dir, metadata_csv, phoneme_vocab, language_map, speaker_map, sr=22050):
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
        # Get the row from metadata
        row = self.metadata.iloc[idx]
        
        # Get speaker and language IDs
        speaker = row['speaker_id'].lower()
        language = row['language'].lower()
        speaker_id = torch.tensor(self.speaker_map[speaker], dtype=torch.long)
        language_id = torch.tensor(self.language_map[language], dtype=torch.long)
        
        # Get phoneme sequence
        phoneme_sequence = row['phoneme_sequence']  # Assuming this column exists
        phoneme_ids = self.convert_phonemes_to_ids(phoneme_sequence)
        
        # Get mel spectrogram
        mel_path = os.path.join(self.root_dir, row['mel_path'])
        mel = torch.load(mel_path)
        
        # Get duration, pitch, energy features if available
        duration = torch.load(os.path.join(self.root_dir, row['duration_path']))
        pitch = torch.load(os.path.join(self.root_dir, row['pitch_path']))
        energy = torch.load(os.path.join(self.root_dir, row['energy_path']))
        
        return {
            "phoneme_ids": phoneme_ids,  # (T,)
            "speaker_id": speaker_id,    # (1,)
            "language_id": language_id,   # (1,)
            "mel": mel,                  # (80, T)
            "duration": duration,        # (T,)
            "pitch": pitch,              # (T,)
            "energy": energy,            # (T,)
            "phoneme_length": torch.tensor(len(phoneme_ids)),
            "mel_length": torch.tensor(mel.size(1))
        }

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
    
    # Generate mask for padded phoneme sequences
    phoneme_mask = phoneme_seqs_padded != 0  # True for non-padded tokens
    mel_mask = mel_specs_padded.sum(dim=1) != 0  # True for non-padded mel-spectrogram entries
    
    return phoneme_seqs_padded, mel_specs_padded, pitches_padded, energies_padded, speaker_ids, language_ids, phoneme_mask, mel_mask


def build_phoneme_vocab():
    fixed_inventory = get_fixed_inventory()  # Your function from the inventory file
    phoneme_vocab = {token: idx for idx, token in enumerate(fixed_inventory)}
    return phoneme_vocab


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
