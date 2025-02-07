import torch
import torch.nn as nn
from embeddings import Embeddings
from predictors import DurationPredictor, PitchPredictor, EnergyPredictor

class FastSpeech2MultiLingual(nn.Module):
    def __init__(self, config):
        super(FastSpeech2MultiLingual, self).__init__()
        self.hidden_size = config["hidden_size"]
        
        # Embeddings: phoneme, language, and speaker embeddings.
        self.embeddings = Embeddings(
            config["phoneme_vocab_size"], config["phoneme_embedding_dim"],
            config["language_vocab_size"], config["language_embedding_dim"],
            config["speaker_vocab_size"], config["speaker_embedding_dim"]
        )
        
        # Input projection: combine embeddings to hidden size.
        combined_dim = config["phoneme_embedding_dim"] + config["language_embedding_dim"] + config["speaker_embedding_dim"]
        self.input_projection = nn.Linear(combined_dim, self.hidden_size)
        
        # Transformer Encoder as backbone.
        encoder_layer = nn.TransformerEncoderLayer(d_model=self.hidden_size, nhead=config["n_heads"], dropout=config["dropout"])
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=config["n_layers"])
        
        # Predictors.
        self.duration_predictor = DurationPredictor(self.hidden_size, kernel_size=3, dropout=config["dropout"])
        self.pitch_predictor = PitchPredictor(self.hidden_size, kernel_size=3, dropout=config["dropout"])
        self.energy_predictor = EnergyPredictor(self.hidden_size, kernel_size=3, dropout=config["dropout"])
        
        # Final projection: project encoder output to mel-spectrogram dimension.
        self.mel_linear = nn.Linear(self.hidden_size, config["mel_dim"])
    
    def forward(self, phoneme_seq, language_ids, speaker_ids, durations=None, pitch=None, energy=None):
        """
        Forward pass for the multilingual FastSpeech2 model.
        
        Args:
            phoneme_seq (LongTensor): [B, T] phoneme indices.
            language_ids (LongTensor): [B] or [B, T] language IDs.
            speaker_ids (LongTensor): [B] speaker IDs.
            durations, pitch, energy: Optional targets.
        
        Returns:
            mel_out: Predicted mel-spectrogram [B, T, mel_dim].
            pred_duration: Duration predictor output [B, T].
            pred_pitch: Pitch predictor output [B, T].
            pred_energy: Energy predictor output [B, T].
        """
        # Get embeddings.
        embed = self.embeddings(phoneme_seq, language_ids, speaker_ids)
        
        # Project to hidden dimension.
        x = self.input_projection(embed)  # (B, T, hidden_size)
        
        # Transformer encoder expects input in shape (T, B, hidden_size).
        x = x.transpose(0, 1)
        encoder_output = self.encoder(x)
        encoder_output = encoder_output.transpose(0, 1)  # (B, T, hidden_size)
        
        # Predict durations, pitch, and energy.
        pred_duration = self.duration_predictor(encoder_output)
        pred_pitch = self.pitch_predictor(encoder_output)
        pred_energy = self.energy_predictor(encoder_output)
        
        # Project encoder output to mel-spectrogram.
        mel_out = self.mel_linear(encoder_output)
        
        return mel_out, pred_duration, pred_pitch, pred_energy
