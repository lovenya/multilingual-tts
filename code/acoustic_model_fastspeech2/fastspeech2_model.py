import torch
import torch.nn as nn
# Import ESPnet2’s FastSpeech2 implementation.
# (Make sure ESPnet2 is installed in your environment.)
from espnet2.tts.fastspeech2.fastspeech2 import FastSpeech2 as ESPnetFastSpeech2

from embeddings import Embeddings
from predictors import DurationPredictor, PitchPredictor, EnergyPredictor

class FastSpeech2MultiLingual(nn.Module):
    def __init__(self, config):
        """
        A multilingual FastSpeech2 model that uses custom embeddings and predictors,
        while leveraging ESPnet2's FastSpeech2 encoder as the acoustic backbone.
        
        Args:
            config (dict): Configuration dictionary containing hyperparameters.
                Expected keys include:
                  - phoneme_vocab_size, phoneme_embedding_dim
                  - language_vocab_size, language_embedding_dim
                  - speaker_vocab_size, speaker_embedding_dim
                  - hidden_size, dropout, mel_dim
                  - adim, n_heads, n_layers, eunits, dunits, postnet_layers, 
                    postnet_chans, postnet_filts, encoder_type, decoder_type, etc.
        """
        super(FastSpeech2MultiLingual, self).__init__()
        self.hidden_size = config["hidden_size"]
        
        # Initialize custom embeddings for phonemes, languages, and speakers.
        self.embeddings = Embeddings(
            config["phoneme_vocab_size"],
            config["phoneme_embedding_dim"],
            config["language_vocab_size"],
            config["language_embedding_dim"],
            config["speaker_vocab_size"],
            config["speaker_embedding_dim"]
        )
        
        # Input projection: map the concatenated embedding to the model's hidden dimension.
        combined_dim = (config["phoneme_embedding_dim"] +
                        config["language_embedding_dim"] +
                        config["speaker_embedding_dim"])
        self.input_projection = nn.Linear(combined_dim, config["hidden_size"])
        
        # Instantiate ESPnet2's FastSpeech2 model with our desired parameters.
        # We only use the encoder part from this model.
        self.espnet_fastspeech2 = ESPnetFastSpeech2(
            idim=config["hidden_size"],  # Our input is the projected embedding
            odim=config["mel_dim"],
            adim=config.get("adim", 384),
            aheads=config.get("n_heads", 8),
            elayers=config.get("n_layers", 4),
            eunits=config.get("eunits", 1536),
            dlayers=config.get("n_layers", 4),
            dunits=config.get("dunits", 1536),
            postnet_layers=config.get("postnet_layers", 5),
            postnet_chans=config.get("postnet_chans", 512),
            postnet_filts=config.get("postnet_filts", 5),
            dropout=config.get("dropout", 0.1),
            encoder_type=config.get("encoder_type", "transformer"),
            decoder_type=config.get("decoder_type", "transformer")
            # Add any additional parameters from your config as needed.
        )
        # Extract the encoder (the acoustic feature extractor) from ESPnet2's model.
        self.encoder = self.espnet_fastspeech2.encoder
        
        # Initialize predictors for prosodic features to be trained from scratch.
        self.duration_predictor = DurationPredictor(config["hidden_size"], kernel_size=3, dropout=config["dropout"])
        self.pitch_predictor = PitchPredictor(config["hidden_size"], kernel_size=3, dropout=config["dropout"])
        self.energy_predictor = EnergyPredictor(config["hidden_size"], kernel_size=3, dropout=config["dropout"])
        
        # Final projection to map encoder output to the mel-spectrogram dimension.
        self.mel_linear = nn.Linear(config["hidden_size"], config["mel_dim"])
    
    def forward(self, phoneme_seq, language_ids, speaker_ids, durations=None, pitch=None, energy=None):
        """
        Forward pass for FastSpeech2MultiLingual.
        
        Args:
            phoneme_seq (LongTensor): (B, T) phoneme indices.
            language_ids (LongTensor): (B) or (B, T) language IDs.
            speaker_ids (LongTensor): (B) speaker IDs.
            durations, pitch, energy: Optional ground-truth targets.
        
        Returns:
            mel_out: Predicted mel-spectrogram [B, T, mel_dim].
            pred_duration: Predicted durations [B, T].
            pred_pitch: Predicted pitch [B, T].
            pred_energy: Predicted energy [B, T].
        """
        # Obtain embeddings for phonemes, languages, and speakers.
        embed = self.embeddings(phoneme_seq, language_ids, speaker_ids)  # (B, T, combined_dim)
        # Project embeddings to the hidden size.
        x = self.input_projection(embed)  # (B, T, hidden_size)
        # Transpose for the encoder (ESPnet2 encoder expects (T, B, hidden_size)).
        x = x.transpose(0, 1)
        encoder_output = self.encoder(x)  # (T, B, hidden_size)
        encoder_output = encoder_output.transpose(0, 1)  # (B, T, hidden_size)
        
        # Predict prosodic features.
        pred_duration = self.duration_predictor(encoder_output)
        pred_pitch = self.pitch_predictor(encoder_output)
        pred_energy = self.energy_predictor(encoder_output)
        
        # Project encoder output to mel-spectrogram.
        mel_out = self.mel_linear(encoder_output)
        
        return mel_out, pred_duration, pred_pitch, pred_energy
