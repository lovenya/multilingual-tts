import logging
import os
import pickle
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
    special_tokens = [
        "<pad>",  # Padding token
        "<unk>",  # Unknown phoneme token
        "<s>",    # Start of sequence token
        "</s>",   # End of sequence token
    ]
    
    
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
    return special_tokens + inventory


def convert_phonemes_to_ids(phoneme_sequence, phoneme_vocab):
    """Convert phoneme sequence to IDs with proper error handling."""
    if not isinstance(phoneme_sequence, str):
        raise ValueError(f"Expected string, got {type(phoneme_sequence)}")
    
    phonemes = phoneme_sequence.strip().split()
    phoneme_ids = []
    
    for phoneme in phonemes:
        if phoneme in phoneme_vocab:
            phoneme_ids.append(phoneme_vocab[phoneme])
        else:
            logging.warning(f"Unknown phoneme: {phoneme}")
            phoneme_ids.append(phoneme_vocab["<unk>"])
    
    # Add start and end tokens if sequence is not empty
    if phoneme_ids:
        phoneme_ids = [phoneme_vocab["<s>"]] + phoneme_ids + [phoneme_vocab["</s>"]]
    else:
        phoneme_ids = [phoneme_vocab["<s>"], phoneme_vocab["</s>"]]
    
    return torch.tensor(phoneme_ids, dtype=torch.long)


def build_phoneme_vocab():
    """Build vocabulary with special tokens and phonemes."""
    inventory = get_fixed_inventory()
    
    # Initialize with special tokens first to ensure consistent IDs
    phoneme_vocab = {}
    for idx, token in enumerate(inventory):
        phoneme_vocab[token] = idx
    
    # Log vocabulary information
    logging.info(f"Built vocabulary with {len(phoneme_vocab)} tokens")
    logging.info(f"Special tokens: {[k for k in phoneme_vocab.keys() if k.startswith('<')]}")
    
    if "<unk>" not in phoneme_vocab:
        raise ValueError("Vocabulary must contain <unk> token")
    if "<pad>" not in phoneme_vocab:
        raise ValueError("Vocabulary must contain <pad> token")
    
    return phoneme_vocab

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


def compute_duration(phoneme_sequence, alignment_path):
    """Compute phoneme durations from alignment."""
    alignment = torch.load(alignment_path)
    durations = []
    current_phoneme = 0
    duration_count = 0
    
    # Convert hard alignment to durations
    for frame in alignment:
        if frame == current_phoneme:
            duration_count += 1
        else:
            durations.append(duration_count)
            current_phoneme = frame
            duration_count = 1
    durations.append(duration_count)  # Add last duration
    
    return torch.tensor(durations, dtype=torch.long)


def estimate_durations(mel_length, num_phonemes):
    """
    Estimate durations by evenly distributing frames across phonemes.
    
    Args:
        mel_length (int): Length of mel spectrogram in frames
        num_phonemes (int): Number of phonemes in the sequence
        
    Returns:
        torch.Tensor: Estimated durations for each phoneme
    """
    # Base duration calculation
    base_duration = mel_length // num_phonemes
    remainder = mel_length % num_phonemes
    
    # Create durations tensor
    durations = torch.full((num_phonemes,), base_duration, dtype=torch.long)
    
    # Distribute remaining frames
    if remainder > 0:
        # Add extra frame to first 'remainder' phonemes
        durations[:remainder] += 1
    
    return durations

def safe_torch_load(filepath):
    """Safely load numpy or torch files."""
    try:
        # Try numpy loading first (for .npy files)
        if filepath.endswith('.npy'):
            return torch.from_numpy(np.load(filepath))
        # Try torch loading
        else:
            return torch.load(filepath, map_location='cpu')
    except Exception as e:
        logging.error(f"Failed to load file {filepath}: {str(e)}")
        return torch.ones(1)  # Return dummy tensor on failure   
    
class TTSDataset(Dataset):
    def __init__(self, root_dir, metadata_csv, phoneme_vocab, language_map, speaker_map, sr=22050):
        """Initialize the dataset."""
        self.root_dir = root_dir
        self.metadata = pd.read_csv(metadata_csv, encoding="utf-8-sig")
        self.phoneme_vocab = phoneme_vocab
        self.language_map = language_map
        self.speaker_map = speaker_map
        self.sr = sr
        
        # Audio processing parameters
        self.n_fft = 1024
        self.hop_length = 256
        self.n_mels = 80
        
        # Count unknown phonemes for logging
        self.unknown_phonemes = set()
        
        # Create mel transform
        self.mel_transform = torchaudio.transforms.MelSpectrogram(
            sample_rate=sr,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            n_mels=self.n_mels
        )
        
        # Validate required columns
        required_columns = ['speaker_id', 'language', 'phoneme_sequence',
                          'pitch_filepath', 'energy_filepath']
        missing_columns = [col for col in required_columns if col not in self.metadata.columns]
        if missing_columns:
            raise ValueError(f"Missing required columns: {missing_columns}")
        
        # Validate vocabulary
        if "<unk>" not in self.phoneme_vocab:
            raise ValueError("Phoneme vocabulary must contain <unk> token")
        if "<pad>" not in self.phoneme_vocab:
            raise ValueError("Phoneme vocabulary must contain <pad> token")
            
        self.pad_id = self.phoneme_vocab["<pad>"]
        self.unk_id = self.phoneme_vocab["<unk>"]
        
        logging.info(f"Initialized dataset with {len(self.metadata)} samples")
        logging.info(f"Vocabulary size: {len(self.phoneme_vocab)}")
        
        
        invalid_files = []
        for _, row in self.metadata.iterrows():
            wav_path = os.path.join(root_dir, row['audio_filepath'])
            pitch_path = os.path.join(root_dir, row['pitch_filepath'])
            energy_path = os.path.join(root_dir, row['energy_filepath'])
            
            if not os.path.exists(wav_path):
                invalid_files.append(wav_path)
            if not os.path.exists(pitch_path):
                invalid_files.append(pitch_path)
            if not os.path.exists(energy_path):
                invalid_files.append(energy_path)
        
        if invalid_files:
            logging.warning(f"Found {len(invalid_files)} invalid files:")
            for f in invalid_files[:10]:  # Show first 10 invalid files
                logging.warning(f"  {f}")
            if len(invalid_files) > 10:
                logging.warning(f"  ... and {len(invalid_files) - 10} more")
        
        
    
    def compute_mel(self, wav_path):
        """Compute mel-spectrogram on the fly."""
        waveform, sample_rate = torchaudio.load(wav_path)
        if sample_rate != self.sr:
            resampler = torchaudio.transforms.Resample(
                orig_freq=sample_rate, 
                new_freq=self.sr
            )
            waveform = resampler(waveform)
        
        # Convert to mono if stereo
        if waveform.size(0) > 1:
            waveform = waveform.mean(dim=0, keepdim=True)
            
        mel_spec = self.mel_transform(waveform)  # (1, n_mels, T)
        return mel_spec.squeeze(0)  # (n_mels, T)

    def convert_phonemes_to_ids(self, phoneme_sequence):
        """Convert phoneme sequence to IDs with better unknown phoneme handling."""
        if not isinstance(phoneme_sequence, str):
            raise ValueError(f"Expected string, got {type(phoneme_sequence)}")
        
        phonemes = phoneme_sequence.strip().split()
        phoneme_ids = []
        
        for phoneme in phonemes:
            if phoneme in self.phoneme_vocab:
                phoneme_ids.append(self.phoneme_vocab[phoneme])
            else:
                if phoneme not in self.unknown_phonemes:
                    logging.warning(f"Unknown phoneme: {phoneme}")
                    self.unknown_phonemes.add(phoneme)
                phoneme_ids.append(self.phoneme_vocab["<unk>"])
        
        # Add start and end tokens if needed
        if "<s>" in self.phoneme_vocab and "</s>" in self.phoneme_vocab:
            phoneme_ids = [self.phoneme_vocab["<s>"]] + phoneme_ids + [self.phoneme_vocab["</s>"]]
        
        return torch.tensor(phoneme_ids, dtype=torch.long)
    def __len__(self):
        return len(self.metadata)

    def __getitem__(self, idx):
        """Get a single training item."""
        try:
            # Get the row first
            row = self.metadata.iloc[idx]
            
            # Get speaker and language IDs
            speaker = row['speaker_id'].lower()
            language = row['language'].lower()
            
            speaker_id = torch.tensor(self.speaker_map[speaker], dtype=torch.long)
            language_id = torch.tensor(self.language_map[language], dtype=torch.long)
            
            # Get phoneme sequence
            phoneme_sequence = str(row['phoneme_sequence'])
            phoneme_ids = self.convert_phonemes_to_ids(phoneme_sequence)
            
            # Load pitch and energy features
            pitch_path = os.path.join(self.root_dir, row['pitch_filepath'])
            energy_path = os.path.join(self.root_dir, row['energy_filepath'])

            try:
                pitch = safe_torch_load(pitch_path)
                energy = safe_torch_load(energy_path)

                # Validate loaded tensors
                if pitch.numel() == 0 or energy.numel() == 0:
                    logging.warning(f"Empty tensor loaded for idx {idx}")
                    pitch = torch.ones(1)  # Default value
                    energy = torch.ones(1)  # Default value
            except Exception as e:
                logging.error(f"Error loading features for index {idx}: {str(e)}")
                pitch = torch.ones(1)  # Default value
                energy = torch.ones(1)  # Default value
                
            # Compute mel spectrogram on the fly
            wav_path = os.path.join(self.root_dir, row['audio_filepath'])
            
            # Debug logging
            logging.debug(f"Loading audio file: {wav_path}")
            if not os.path.exists(wav_path):
                raise FileNotFoundError(f"Audio file not found: {wav_path}")
                
            mel = self.compute_mel(wav_path)
            
            # Ensure all features have matching lengths
            min_length = min(mel.size(1), len(pitch), len(energy))
            mel = mel[:, :min_length]
            pitch = pitch[:min_length]
            energy = energy[:min_length]
            
            # Estimate durations
            duration = estimate_durations(mel.size(1), len(phoneme_ids))
            
            return {
                "phoneme_ids": phoneme_ids,
                "speaker_id": speaker_id,
                "language_id": language_id,
                "mel": mel,
                "duration": duration,
                "pitch": pitch,
                "energy": energy,
                "phoneme_length": torch.tensor(len(phoneme_ids)),
                "mel_length": torch.tensor(mel.size(1))
            }
                
        except Exception as e:
            logging.error(f"Error processing item {idx}:")
            logging.error(f"Error type: {type(e).__name__}")
            logging.error(f"Error message: {str(e)}")
            if 'row' in locals():
                logging.error(f"Row data: {row.to_dict()}")
            logging.error(f"Root dir: {self.root_dir}")
            raise  



def dynamic_collate_fn(batch):
    """Collate function for dynamic batch sizes with proper padding."""
    # Get max lengths
    max_phoneme_len = max(x["phoneme_length"] for x in batch)
    max_mel_len = max(x["mel_length"] for x in batch)
    
    # Initialize tensors
    batch_size = len(batch)
    phoneme_ids = torch.zeros(batch_size, max_phoneme_len, dtype=torch.long)
    speaker_ids = torch.zeros(batch_size, dtype=torch.long)
    language_ids = torch.zeros(batch_size, dtype=torch.long)
    mels = torch.zeros(batch_size, batch[0]["mel"].size(0), max_mel_len)
    durations = torch.zeros(batch_size, max_phoneme_len)
    pitch = torch.zeros(batch_size, max_mel_len)
    energy = torch.zeros(batch_size, max_mel_len)
    
    phoneme_lengths = []
    mel_lengths = []
    
    for i, item in enumerate(batch):
        phoneme_len = item["phoneme_length"]
        mel_len = item["mel_length"]
        
        # Store lengths
        phoneme_lengths.append(phoneme_len)
        mel_lengths.append(mel_len)
        
        # Fill tensors with data
        phoneme_ids[i, :phoneme_len] = item["phoneme_ids"]
        speaker_ids[i] = item["speaker_id"]
        language_ids[i] = item["language_id"]
        mels[i, :, :mel_len] = item["mel"]
        
        # Handle duration padding correctly
        dur_len = len(item["duration"])
        if dur_len > max_phoneme_len:
            # Truncate if longer
            durations[i, :] = item["duration"][:max_phoneme_len]
        else:
            # Pad if shorter
            durations[i, :dur_len] = item["duration"]
            
        pitch[i, :mel_len] = item["pitch"][:mel_len]
        energy[i, :mel_len] = item["energy"][:mel_len]
    
    return {
        "phoneme_ids": phoneme_ids,
        "speaker_ids": speaker_ids,
        "language_ids": language_ids,
        "mels": mels,
        "durations": durations,
        "pitch": pitch,
        "energy": energy,
        "phoneme_lengths": torch.tensor(phoneme_lengths),
        "mel_lengths": torch.tensor(mel_lengths)
    }
# def compute_mel(self, wav_path):
#     """Compute mel-spectrogram on the fly."""
#     waveform, sample_rate = torchaudio.load(wav_path)
#     if sample_rate != self.sr:
#         resampler = torchaudio.transforms.Resample(
#             orig_freq=sample_rate, 
#             new_freq=self.sr
#         )
#         waveform = resampler(waveform)
    
#     # Convert to mono if stereo
#     if waveform.size(0) > 1:
#         waveform = waveform.mean(dim=0, keepdim=True)
        
#     mel_spec = self.mel_transform(waveform)  # (1, n_mels, T)
#     return mel_spec.squeeze(0)  # (n_mels, T)


# def build_phoneme_vocab():
#     """Build vocabulary with special tokens and phonemes."""
#     inventory = get_fixed_inventory()
    
#     # Initialize with special tokens first to ensure consistent IDs
#     phoneme_vocab = {}
#     for idx, token in enumerate(inventory):
#         phoneme_vocab[token] = idx
    
#     # Log vocabulary information
#     logging.info(f"Built vocabulary with {len(phoneme_vocab)} tokens")
#     logging.info(f"Special tokens: {[k for k in phoneme_vocab.keys() if k.startswith('<')]}")
    
#     if "<unk>" not in phoneme_vocab:
#         raise ValueError("Vocabulary must contain <unk> token")
#     if "<pad>" not in phoneme_vocab:
#         raise ValueError("Vocabulary must contain <pad> token")
    
#     return phoneme_vocab

if __name__ == '__main__':
    # Example mappings (replace with your actual mappings)
    phoneme_vocab = build_phoneme_vocab()  # Extend to your full phoneme vocabulary (from fixed inventory)
    language_map = {"english": 0, "gujarathi": 1, "bhojpuri": 2, "kannada": 3}
    speaker_map = {"english_f": 0, "english_m": 1, "bhojpuri_f": 2, "bhojpuri_m": 3,
                   "gujarathi_f": 4, "gujarathi_m": 5, "kannada_f": 6, "kannada_m": 7}
    
    # Path to metadata CSV (update path as needed)
    metadata_csv = "dataset/metadata/updated_train.csv"
    dataset = TTSDataset(root_dir=".", metadata_csv=metadata_csv,
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


if __name__ == '__main__':
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(levelname)s: %(message)s'
    )

    # Initialize dataset components
    phoneme_vocab = build_phoneme_vocab()
    language_map = {"english": 0, "gujarathi": 1, "bhojpuri": 2, "kannada": 3}
    speaker_map = {
        "english_f": 0, "english_m": 1, "bhojpuri_f": 2, "bhojpuri_m": 3,
        "gujarathi_f": 4, "gujarathi_m": 5, "kannada_f": 6, "kannada_m": 7
    }
    
    # Create dataset
    metadata_csv = "dataset/metadata/updated_train.csv"
    dataset = TTSDataset(
        root_dir=".", 
        metadata_csv=metadata_csv,
        phoneme_vocab=phoneme_vocab, 
        language_map=language_map, 
        speaker_map=speaker_map
    )
    
    # Setup weighted sampling
    df = pd.read_csv(metadata_csv, encoding="utf-8-sig")
    weights = [2.0 if row['language'].lower() in ['gujarathi', 'bhojpuri'] else 1.0 
              for _, row in df.iterrows()]
    weights = torch.tensor(weights, dtype=torch.float)
    sampler = WeightedRandomSampler(weights, num_samples=len(weights), replacement=True)
    
    # Create dataloader
    data_loader = DataLoader(
        dataset, 
        batch_size=8, 
        collate_fn=dynamic_collate_fn,
        sampler=sampler,
        num_workers=0  # Set to 0 for debugging
    )
    
    print("\nTesting dataloader...")
    
    for batch_idx, batch in enumerate(data_loader):
        print(f"\nBatch {batch_idx + 1}:")
        print("Batch keys:", batch.keys())
        
        # Print shapes of all tensors in batch
        for key, value in batch.items():
            if isinstance(value, torch.Tensor):
                print(f"{key:<15} shape: {tuple(value.shape)}")
        
        # Verify tensor dimensions
        assert batch['phoneme_ids'].ndim == 2, "phoneme_ids should be 2D (batch_size, seq_len)"
        assert batch['mels'].ndim == 3, "mels should be 3D (batch_size, n_mels, time)"
        assert batch['pitch'].ndim == 2, "pitch should be 2D (batch_size, time)"
        assert batch['energy'].ndim == 2, "energy should be 2D (batch_size, time)"
        
        print("\nAll tensor shapes verified successfully!")
        break  # Test only first batch