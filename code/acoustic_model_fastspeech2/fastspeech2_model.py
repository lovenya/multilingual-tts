import torch
import torch.nn as nn
# Import ESPnet2’s FastSpeech2 implementation
from espnet2.tts.fastspeech2.fastspeech2 import FastSpeech2 as ESPnetFastSpeech2
from embeddings import Embeddings
from predictors import DurationPredictor, PitchPredictor, EnergyPredictor

class FastSpeech2MultiLingual(nn.Module):
    def __init__(self, config):
        """
        A multilingual FastSpeech2 model that uses custom embeddings and predictors,
        while leveraging ESPnet2's pretrained FastSpeech2 encoder as the acoustic backbone.
        
        Args:
            config (dict): Configuration dictionary containing hyperparameters.
        """
        super(FastSpeech2MultiLingual, self).__init__()
        self.hidden_size = config["hidden_size"]
        
        # Custom embeddings for phonemes, languages, and speakers.
        self.embeddings = Embeddings(
            config["phoneme_vocab_size"],
            config["phoneme_embedding_dim"],
            config["language_vocab_size"],
            config["language_embedding_dim"],
            config["speaker_vocab_size"],
            config["speaker_embedding_dim"]
        )
        
        # Input projection: map the concatenated embeddings to the hidden size.
        combined_dim = (config["phoneme_embedding_dim"] +
                        config["language_embedding_dim"] +
                        config["speaker_embedding_dim"])
        self.input_projection = nn.Linear(combined_dim, config["hidden_size"])
        
        # Load ESPnet2's pretrained FastSpeech2 encoder.
        # This loads the encoder weights from the pretrained checkpoint "tts_en_fastspeech2".
        self.espnet_fastspeech2 = ESPnetFastSpeech2.from_pretrained("tts_en_fastspeech2")
        self.encoder = self.espnet_fastspeech2.encoder
        
        # Initialize predictors for prosodic features (duration, pitch, energy) from scratch.
        self.duration_predictor = DurationPredictor(config["hidden_size"], kernel_size=3, dropout=config["dropout"])
        self.pitch_predictor = PitchPredictor(config["hidden_size"], kernel_size=3, dropout=config["dropout"])
        self.energy_predictor = EnergyPredictor(config["hidden_size"], kernel_size=3, dropout=config["dropout"])
        
        # Final linear projection to produce mel-spectrogram.
        self.mel_linear = nn.Linear(config["hidden_size"], config["mel_dim"])
    
    def forward(self, phoneme_seq, language_ids, speaker_ids, durations=None, pitch=None, energy=None):
        """
        Args:
            phoneme_seq (LongTensor): Tensor of shape [B, T] containing phoneme indices.
            language_ids (LongTensor): Tensor of shape [B] or [B, T] containing language IDs.
            speaker_ids (LongTensor): Tensor of shape [B] containing speaker IDs.
            durations, pitch, energy: Optional targets (if provided for training).
        
        Returns:
            mel_out: Predicted mel-spectrogram [B, T, mel_dim].
            pred_duration: Predicted duration values [B, T].
            pred_pitch: Predicted pitch values [B, T].
            pred_energy: Predicted energy values [B, T].
        """
        # Obtain embeddings.
        embed = self.embeddings(phoneme_seq, language_ids, speaker_ids)  # (B, T, combined_dim)
        x = self.input_projection(embed)  # (B, T, hidden_size)
        
        # Transpose for the encoder (ESPnet2 expects input shape: [T, B, hidden_size])
        x = x.transpose(0, 1)
        encoder_output = self.encoder(x)
        encoder_output = encoder_output.transpose(0, 1)  # (B, T, hidden_size)
        
        # Predict prosodic features.
        pred_duration = self.duration_predictor(encoder_output)
        pred_pitch = self.pitch_predictor(encoder_output)
        pred_energy = self.energy_predictor(encoder_output)
        
        # Project to mel-spectrogram.
        mel_out = self.mel_linear(encoder_output)
        
        return mel_out, pred_duration, pred_pitch, pred_energy
