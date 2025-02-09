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
        Dataset for HiFi-GAN training.
        Args:
            csv_file (str): Path to CSV file with metadata
            mel_dir (str): Not used, kept for backward compatibility
            dataset_dir (str): Root directory containing all data
        """
        self.data = pd.read_csv(csv_file)
        self.dataset_dir = dataset_dir.rstrip('/')  # Remove trailing slash if present
        
        # Print debug info
        print("\nInitializing HiFiGANDataset:")
        print(f"Dataset directory: {self.dataset_dir}")
        print("First few audio_filepath entries:")
        print(self.data['audio_filepath'].head())

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        
        # Get the audio filepath from CSV
        audio_path = row['audio_filepath']
        
        # Construct wav path
        wav_path = os.path.join(self.dataset_dir, audio_path)
        
        # Construct mel path by:
        # 1. Split the path into parts
        path_parts = audio_path.split('/')  # ['dataset', 'Kannada_M', 'wav', 'Ka_M_04513.wav']
        
        # 2. Replace 'wav' directory with 'mel_spectrograms' and change extension
        mel_filename = os.path.splitext(path_parts[-1])[0] + '.npy'  # 'Ka_M_04513.npy'
        
        # 3. Construct mel path based on speaker folder
        mel_path = os.path.join(
            self.dataset_dir,  # dataset root directory
            path_parts[1],     # speaker directory (e.g., 'Kannada_M')
            'mel_spectrograms', # mel_spectrograms folder
            mel_filename       # mel spectrogram file
        )
        
        try:
            # Load mel spectrogram
            mel = np.load(mel_path)
            mel = torch.tensor(mel, dtype=torch.float32)
            
            # Load audio waveform
            waveform, _ = torchaudio.load(wav_path)
            
            return mel, waveform
        except Exception as e:
            print(f"\nError loading files for index {idx}:")
            print(f"Audio path from CSV: {audio_path}")
            print(f"Constructed wav path: {wav_path}")
            print(f"Constructed mel path: {mel_path}")
            raise
     
        
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
    # Use absolute path for dataset directory
    dataset_dir = 'dataset'
    
    # CSV paths
    train_csv = os.path.join(dataset_dir, 'metadata', 'updated_train.csv')
    val_csv = os.path.join(dataset_dir, 'metadata', 'updated_val.csv')
    
    print("\nInitializing training...")
    print(f"Dataset directory: {dataset_dir}")
    print(f"Training CSV: {train_csv}")
    print(f"Validation CSV: {val_csv}")
    
    # Load pretrained HiFi-GAN model
    model = HifiGanModel.from_pretrained("tts_en_hifigan")
    model = model.cuda()

    # Create datasets with smaller batch size initially for testing
    train_dataset = HiFiGANDataset(train_csv, None, dataset_dir)
    val_dataset = HiFiGANDataset(val_csv, None, dataset_dir)
    
    # Start with a smaller batch size to test
    batch_size = 8  # Reduced from 16 to help with debugging
    
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True,
        num_workers=2  # Reduced number of workers for testing
    )
    
    val_loader = DataLoader(
        val_dataset, 
        batch_size=batch_size, 
        shuffle=False,
        num_workers=2
    )

    # Fine-tune the model
    fine_tune_hifi_gan(model, train_loader, val_loader)

if __name__ == "__main__":
    main()
