import torch
import torch.nn as nn
from espnet2.tts.fastspeech2.fastspeech2 import FastSpeech2 as ESPnetFastSpeech2
from espnet2.train.abs_espnet_model import AbsESPnetModel  # Import the base model
from embeddings import Embeddings
from predictors import DurationPredictor, PitchPredictor, EnergyPredictor

class FastSpeech2MultiLingual(AbsESPnetModel):  # Inherit from AbsESPnetModel
    def __init__(self, config):
        super(FastSpeech2MultiLingual, self).__init__()

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

        # ESPnet2's FastSpeech2 model initialization
        self.espnet_fastspeech2 = ESPnetFastSpeech2(
            idim=config["phoneme_vocab_size"],    # 134 for the phoneme vocabulary size
            odim=config["mel_dim"],                # Mel-spectrogram output dimension
            adim=config["adim"],
            aheads=config["n_heads"],
            elayers=config["n_layers"],
            eunits=config["eunits"],
            dlayers=config["n_layers"],           # Not used in our forward pass
            dunits=config["dunits"],              # Not used in our forward pass
            postnet_layers=config["postnet_layers"],
            postnet_chans=config["postnet_chans"],
            postnet_filts=config["postnet_filts"]
            # Removed dropout parameter here
        )

        # Optionally load pretrained encoder weights if provided
        if "pretrained_encoder_path" in config and config["pretrained_encoder_path"]:
            print(f"Loading pretrained encoder weights from {config['pretrained_encoder_path']}")
            checkpoint = torch.load(config["pretrained_encoder_path"], map_location="cpu")
            self.espnet_fastspeech2.load_state_dict(checkpoint, strict=False)

        # Apply dropout inside layers manually where required
        self.dropout = nn.Dropout(config["dropout"])

        # Initialize predictors for prosodic features from scratch
        self.duration_predictor = DurationPredictor(config["hidden_size"], kernel_size=3, dropout=config["dropout"])
        self.pitch_predictor = PitchPredictor(config["hidden_size"], kernel_size=3, dropout=config["dropout"])
        self.energy_predictor = EnergyPredictor(config["hidden_size"], kernel_size=3, dropout=config["dropout"])

        # Final linear projection to produce the mel-spectrogram
        self.mel_linear = nn.Linear(config["hidden_size"], config["mel_dim"])

    def forward(self, phoneme_seq, language_ids, speaker_ids, durations=None, pitch=None, energy=None):
        """
        Args:
            phoneme_seq (LongTensor): [B, T] containing phoneme indices.
            language_ids (LongTensor): [B] or [B, T] containing language IDs.
            speaker_ids (LongTensor): [B] containing speaker IDs.
            durations, pitch, energy: Optional targets for training.

        Returns:
            mel_out: Predicted mel-spectrogram [B, T, mel_dim].
            pred_duration: Predicted duration values [B, T].
            pred_pitch: Predicted pitch values [B, T].
            pred_energy: Predicted energy values [B, T].
        """
        # Get combined embeddings.
        embed = self.embeddings(phoneme_seq, language_ids, speaker_ids)  # (B, T, combined_dim)
        x = self.input_projection(embed)  # (B, T, hidden_size)

        # Apply dropout to the input projection if needed
        x = self.dropout(x)

        # Transpose for the encoder (ESPnet2 expects [T, B, hidden_size]).
        x = x.transpose(0, 1)
        encoder_output = self.espnet_fastspeech2.encoder(x)
        encoder_output = encoder_output.transpose(0, 1)  # (B, T, hidden_size)

        # Apply dropout after the encoder layer if needed
        encoder_output = self.dropout(encoder_output)

        # Predict prosodic features
        pred_duration = self.duration_predictor(encoder_output)
        pred_pitch = self.pitch_predictor(encoder_output)
        pred_energy = self.energy_predictor(encoder_output)

        # Project encoder output to mel-spectrogram.
        mel_out = self.mel_linear(encoder_output)

        return mel_out, pred_duration, pred_pitch, pred_energy

    def compute_loss(self, mel_out, pred_duration, pred_pitch, pred_energy, mel, durations, pitch, energy):
        """
        Compute the loss for the model, which should be used in training.
        Args:
            mel_out: Predicted mel-spectrogram.
            pred_duration: Predicted duration values.
            pred_pitch: Predicted pitch values.
            pred_energy: Predicted energy values.
            mel: Ground-truth mel-spectrogram.
            durations: Ground-truth durations.
            pitch: Ground-truth pitch values.
            energy: Ground-truth energy values.

        Returns:
            loss: The total loss for the batch.
        """
        mel_loss = nn.MSELoss()(mel_out, mel)
        duration_loss = nn.MSELoss()(pred_duration, durations)
        pitch_loss = nn.MSELoss()(pred_pitch, pitch)
        energy_loss = nn.MSELoss()(pred_energy, energy)

        total_loss = mel_loss + duration_loss + pitch_loss + energy_loss
        return total_loss

    def compute_metrics(self, mel_out, pred_duration, pred_pitch, pred_energy, mel, durations, pitch, energy):
        """
        Compute additional metrics for evaluation, if necessary.
        """
        # For simplicity, we return None here; you can add more evaluation metrics as needed.
        return None
    
    def collect_feats(self, phoneme_seq, language_ids, speaker_ids, durations=None, pitch=None, energy=None):
        """
        Collect features from the input data that are passed to the model.
        Args:
            phoneme_seq: The phoneme sequence (B, T).
            language_ids: The language IDs (B, T) or (B).
            speaker_ids: The speaker IDs (B).
            durations: Ground-truth duration values (optional).
            pitch: Ground-truth pitch values (optional).
            energy: Ground-truth energy values (optional).

        Returns:
            A dictionary with the features to be passed to the model.
        """
        # Collect embeddings for phonemes, languages, and speakers.
        features = {
            "phoneme_seq": phoneme_seq,
            "language_ids": language_ids,
            "speaker_ids": speaker_ids
        }

        if durations is not None:
            features["durations"] = durations
        if pitch is not None:
            features["pitch"] = pitch
        if energy is not None:
            features["energy"] = energy

        return features
