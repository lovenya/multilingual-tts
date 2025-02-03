import torch
from pathlib import Path
import numpy as np
from tqdm import tqdm
import pandas as pd
from nemo.collections.tts.models import FastPitchModel
import librosa
import warnings
warnings.filterwarnings('ignore')

class MelSpectrogramGenerator:
    def __init__(self, model_name="tts_en_fastpitch"):
        """Initialize mel spectrogram generator using NeMo's FastPitch model."""
        # Load pretrained FastPitch model
        self.model = FastPitchModel.from_pretrained(model_name)
        self.model.eval()
        
        # Get parameters from model config
        self.sample_rate = self.model.cfg.preprocessor.sample_rate
        self.hop_length = self.model.cfg.preprocessor.hop_length
        
        # Setup device
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.model = self.model.to(self.device)
    
    def load_audio(self, file_path):
        """Load and normalize audio file."""
        try:
            wav, sr = librosa.load(file_path, sr=self.sample_rate)
            wav = wav / np.max(np.abs(wav))
            return wav
        except Exception as e:
            print(f"Error loading audio file {file_path}: {e}")
            return None

    def process_audio(self, wav_path):
        """Generate mel spectrogram from audio file."""
        # Load audio
        wav = self.load_audio(wav_path)
        if wav is None:
            return None

        # Convert to tensor
        wav_tensor = torch.FloatTensor(wav).unsqueeze(0).to(self.device)
        wav_length = torch.tensor([wav_tensor.shape[1]], device=self.device)

        # Generate mel spectrogram
        with torch.no_grad():
            mel, mel_length = self.model.preprocessor(
                input_signal=wav_tensor,
                length=wav_length
            )
            return mel[0].cpu().numpy()

def process_datasets(base_path, metadata_path, output_base_path):
    """Process all datasets (train, validation, test)."""
    print("Loading FastPitch model...")
    mel_gen = MelSpectrogramGenerator()
    
    output_base_path = Path(output_base_path)
    splits = ['train.csv', 'validation.csv', 'test.csv']
    
    for split in splits:
        print(f"\nProcessing {split}...")
        metadata_csv = Path(metadata_path) / split
        output_path = output_base_path / split.replace('.csv', '')
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Read metadata
        df = pd.read_csv(metadata_csv)
        print(f"Processing {len(df)} files...")
        
        for idx, row in tqdm(df.iterrows(), total=len(df)):
            wav_path = row['audio_filepath']
            mel_path = output_path / f"{Path(wav_path).stem}.npy"
            
            try:
                mel = mel_gen.process_audio(wav_path)
                if mel is not None:
                    np.save(str(mel_path), mel)
            except Exception as e:
                print(f"Error processing {wav_path}: {e}")

if __name__ == "__main__":
    base_path = "dataset"
    metadata_path = "dataset/metadata"
    output_path = "dataset/mel_spectrograms"
    
    process_datasets(base_path, metadata_path, output_path)