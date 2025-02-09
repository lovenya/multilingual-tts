import os
import nemo
import torch
from omegaconf import OmegaConf
from nemo.collections.asr import HIFIGANModel
from nemo.utils import logging
from torch.utils.data import DataLoader
from torch.optim import Adam
from tqdm import tqdm
import numpy as np
import torchaudio

class HiFiGANDataset(torch.utils.data.Dataset):
    def __init__(self, mel_dir, waveform_dir):
        """
        Dataset for HiFi-GAN training. Pairs mel-spectrograms with raw waveforms.

        Parameters:
        - mel_dir: Directory containing mel-spectrograms (.npy files)
        - waveform_dir: Directory containing raw waveforms (.wav files)
        """
        self.mel_files = [os.path.join(mel_dir, f) for f in os.listdir(mel_dir) if f.endswith('.npy')]
        self.wav_files = [os.path.join(waveform_dir, f.replace('.npy', '.wav')) for f in os.listdir(mel_dir) if f.endswith('.npy')]
        self.mel_dir = mel_dir
        self.wav_dir = waveform_dir

    def __len__(self):
        return len(self.mel_files)

    def __getitem__(self, idx):
        mel = np.load(self.mel_files[idx])
        mel = torch.tensor(mel, dtype=torch.float32)
        wav, _ = torchaudio.load(self.wav_files[idx])
        return mel, wav

def train_hifi_gan(model, train_loader, optimizer, num_epochs, log_interval=100, checkpoint_dir='./checkpoints'):
    """
    Fine-tune HiFi-GAN on the dataset.

    Parameters:
    - model: Pretrained HiFi-GAN model
    - train_loader: DataLoader for training data
    - optimizer: Optimizer for training
    - num_epochs: Number of epochs to train the model
    """
    model.train()
    criterion = torch.nn.MSELoss()

    # Training loop
    for epoch in range(num_epochs):
        running_loss = 0.0
        for i, (mel, waveform) in enumerate(tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}")):
            mel, waveform = mel.cuda(), waveform.cuda()

            # Forward pass through HiFi-GAN generator
            generated_waveform = model.generator(mel)

            # Calculate loss
            loss = criterion(generated_waveform, waveform)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            if i % log_interval == 0:
                logging.info(f"Epoch {epoch+1}, Iter {i}, Loss: {running_loss / (i+1)}")

        # Save checkpoint after each epoch
        if (epoch+1) % 10 == 0:
            checkpoint_path = os.path.join(checkpoint_dir, f"hifi_gan_epoch_{epoch+1}.ckpt")
            model.save_to(checkpoint_path)
            logging.info(f"Checkpoint saved to {checkpoint_path}")

def main():
    # Load HiFi-GAN configuration
    config_path = 'path/to/hifi_gan_config.yaml'
    config = OmegaConf.load(config_path)

    # Initialize HiFi-GAN model
    model = HIFIGANModel(cfg=config)
    model = model.cuda()

    # Load the pretrained HiFi-GAN model for fine-tuning (if available)
    checkpoint_path = 'path/to/pretrained_model.ckpt'
    if checkpoint_path:
        model.restore_from(restore_path=checkpoint_path)
        logging.info(f"Loaded pretrained model from {checkpoint_path}")
    else:
        logging.info("Initializing model from scratch")

    # Setup data loader
    mel_dir = 'path/to/mel/spectrograms'  # Directory with mel-spectrograms
    waveform_dir = 'path/to/wav/files'    # Directory with raw waveforms
    train_dataset = HiFiGANDataset(mel_dir, waveform_dir)
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)

    # Set optimizer (Adam)
    optimizer = Adam(model.parameters(), lr=1e-4, betas=(0.9, 0.999))

    # Training loop
    train_hifi_gan(model, train_loader, optimizer, num_epochs=200, checkpoint_dir='./checkpoints')

if __name__ == "__main__":
    main()
