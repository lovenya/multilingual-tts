import nemo
from nemo.collections.tts.models import HifiGanModel
from nemo.utils import logging
import torch
from torch.utils.data import DataLoader
import torchaudio
import os
import pandas as pd
import numpy as np
from tqdm import tqdm
import time

# DataLoader for training and validation
class HiFiGANDataset(torch.utils.data.Dataset):
    def __init__(self, csv_file, mel_dir, dataset_dir):
        """
        Dataset for HiFi-GAN training. Loads mel-spectrograms and raw waveforms
        based on the CSV metadata and directory structure.
        
        Parameters:
        - csv_file: Path to the CSV file with metadata (e.g., updated_train.csv)
        - mel_dir: Directory containing mel-spectrograms (.npy files)
        - dataset_dir: Root directory containing subfolders for each speaker
        """
        self.data = pd.read_csv(csv_file)
        self.mel_dir = mel_dir
        self.dataset_dir = dataset_dir

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        
        # Get the speaker folder and audio file path from the metadata
        speaker_folder = row['speaker_id']  # Assuming speaker_id is the folder name
        wav_path = os.path.join(self.dataset_dir, speaker_folder, 'wav', row['audio_filepath'])
        
        # Construct the mel spectrogram file path
        mel_path = os.path.join(self.dataset_dir, speaker_folder, 'mel_spectrograms', row['audio_filepath'].replace('.wav', '.npy'))

        # Load mel spectrogram and waveform
        mel = np.load(mel_path)
        mel = torch.tensor(mel, dtype=torch.float32)
        
        waveform, _ = torchaudio.load(wav_path)
        
        return mel, waveform


def fine_tune_hifi_gan(model, train_loader, val_loader, num_epochs=200, checkpoint_dir='./checkpoints'):
    """
    Fine-tune HiFi-GAN on the dataset, including validation.
    """
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4, betas=(0.9, 0.999))
    criterion = torch.nn.MSELoss()

    # Start timing the entire training process
    start_time = time.time()

    # Training loop
    for epoch in range(num_epochs):
        epoch_start_time = time.time()
        running_loss = 0.0

        # Create a progress bar for the training loop
        with tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}", unit="batch") as pbar:
            for i, (mel, waveform) in enumerate(pbar):
                mel, waveform = mel.cuda(), waveform.cuda()

                # Forward pass through HiFi-GAN generator
                generated_waveform = model.generator(mel)

                # Calculate loss
                loss = criterion(generated_waveform, waveform)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                running_loss += loss.item()

                # Update the progress bar with loss and remaining time
                pbar.set_postfix(loss=running_loss / (i + 1), batch_loss=loss.item())

        # Calculate and print time elapsed for the epoch
        epoch_time = time.time() - epoch_start_time
        logging.info(f"Epoch {epoch+1} finished in {epoch_time:.2f}s")

        # Evaluate on validation set after each epoch
        val_loss = 0.0
        model.eval()

        # Create a progress bar for validation
        with tqdm(val_loader, desc="Validation", unit="batch") as pbar_val:
            with torch.no_grad():
                for mel, waveform in pbar_val:
                    mel, waveform = mel.cuda(), waveform.cuda()
                    generated_waveform = model.generator(mel)
                    val_loss += criterion(generated_waveform, waveform).item()

        val_loss /= len(val_loader)
        logging.info(f"Epoch {epoch+1}, Validation Loss: {val_loss}")

        # Save checkpoint after each epoch
        if (epoch + 1) % 10 == 0:
            checkpoint_path = os.path.join(checkpoint_dir, f"hifi_gan_epoch_{epoch+1}.ckpt")
            model.save_to(checkpoint_path)
            logging.info(f"Checkpoint saved to {checkpoint_path}")

        # Print time elapsed after epoch
        total_time = time.time() - start_time
        logging.info(f"Total training time so far: {total_time / 60:.2f} minutes")

def main():
    # Paths
    mel_dir = 'dataset'  # Directory with mel-spectrograms
    dataset_dir = 'dataset'       # Root directory containing speaker subfolders
    train_csv = 'dataset/metadata/updated_train.csv'  # Path to the updated_train.csv file
    val_csv = 'dataset/metadata/updated_val.csv'    # Path to the updated_val.csv file

    # Load pretrained HiFi-GAN model
    model = HifiGanModel.from_pretrained("tts_en_hifigan")
    model = model.cuda()

    # Set up data loaders for training and validation
    train_dataset = HiFiGANDataset(train_csv, mel_dir, dataset_dir)
    val_dataset = HiFiGANDataset(val_csv, mel_dir, dataset_dir)
    
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)

    # Fine-tune the model
    fine_tune_hifi_gan(model, train_loader, val_loader)


if __name__ == "__main__":
    main()
