import torch
import torch.nn as nn
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
                           Must include key "pretrained_encoder_path" for loading the
                           pretrained encoder checkpoint.
        """
        super(FastSpeech2MultiLingual, self).__init__()

        # Custom embeddings for phonemes, languages, and speakers.
        self.embeddings = Embeddings(
            config["phoneme_vocab_size"],       # Ensure this is 78 for the pretrained model
            config["phoneme_embedding_dim"],    # Set this to 78 for the pretrained model
            config["language_vocab_size"],      # For languages: {en, gu, bh, kn}
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
            idim=config["phoneme_vocab_size"],    # 78 for the pretrained model
            odim=config["mel_dim"],                # Mel-spectrogram output dimension
            adim=config["adim"],
            aheads=config["n_heads"],
            elayers=config["n_layers"],
            eunits=config["eunits"],
            dlayers=config["n_layers"],           # Not used in our forward pass
            dunits=config["dunits"],              # Not used in our forward pass
            postnet_layers=config["postnet_layers"],
            postnet_chans=config["postnet_chans"],
            postnet_filts=config["postnet_filts"],
            # dropout_rate=config["dropout"]
        )

        # Optionally load pretrained encoder weights if provided
        if "pretrained_encoder_path" in config and config["pretrained_encoder_path"]:
            print(f"Loading pretrained encoder weights from {config['pretrained_encoder_path']}")
            checkpoint = torch.load(config["pretrained_encoder_path"], map_location="cpu")
            print("Checkpoint Keys:", checkpoint.keys())  # To inspect the available keys

            # Modify checkpoint to match the model's keys
            state_dict = checkpoint.get('model', checkpoint)  # Adjust for different checkpoint formats
            state_dict = {k: v for k, v in state_dict.items() if k in self.espnet_fastspeech2.state_dict()}
            self.espnet_fastspeech2.load_state_dict(state_dict, strict=False)

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

        # Transpose for the encoder (ESPnet2 expects [T, B, hidden_size]).
        x = x.transpose(0, 1)
        encoder_output = self.espnet_fastspeech2.encoder(x)
        encoder_output = encoder_output.transpose(0, 1)  # (B, T, hidden_size)

        # Predict prosodic features
        pred_duration = self.duration_predictor(encoder_output)
        pred_pitch = self.pitch_predictor(encoder_output)
        pred_energy = self.energy_predictor(encoder_output)

        # Project encoder output to mel-spectrogram.
        mel_out = self.mel_linear(encoder_output)

        return mel_out, pred_duration, pred_pitch, pred_energy
