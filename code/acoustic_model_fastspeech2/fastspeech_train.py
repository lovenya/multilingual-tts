import torch
import torch.nn as nn
import torch.optim as optim
import yaml
from torch.utils.data import DataLoader
from fastspeech2_model import FastSpeech2MultiLingual
from dataloader_for_acoustic_model import TTSDataset, dynamic_collate_fn
import os

def load_config(config_path):
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config

def main():
    # Load configuration.
    config = load_config("config/nemo_model_fastspeech2.yaml")
    
    # Define mapping dictionaries.
    phoneme_vocab = {}  # Fill with your actual phoneme-to-index mapping.
    language_map = {"english": 0, "gujarathi": 1, "bhojpuri": 2, "kannada": 3}
    speaker_map = {"english_f": 0, "english_m": 1, "bhojpuri_f": 2, "bhojpuri_m": 3,
                   "gujarathi_f": 4, "gujarathi_m": 5, "kannada_f": 6, "kannada_m": 7}
    
    # Create training dataset and DataLoader.
    train_metadata = "dataset/metadata/updated_train.csv"
    train_dataset = TTSDataset(root_dir="dataset", metadata_csv=train_metadata,
                               phoneme_vocab=phoneme_vocab, language_map=language_map, speaker_map=speaker_map)
    train_loader = DataLoader(train_dataset, batch_size=32, collate_fn=dynamic_collate_fn, shuffle=True)
    
    # (Optional) Create a validation DataLoader similarly.
    val_metadata = "dataset/metadata/updated_val.csv"
    val_dataset = TTSDataset(root_dir="dataset", metadata_csv=val_metadata,
                             phoneme_vocab=phoneme_vocab, language_map=language_map, speaker_map=speaker_map)
    val_loader = DataLoader(val_dataset, batch_size=32, collate_fn=dynamic_collate_fn, shuffle=False)
    
    # Initialize the model.
    model = FastSpeech2MultiLingual(config)
    model = model.cuda()
    
    # Set up optimizer with differential learning rates.
    optimizer = optim.Adam([
        {'params': model.embeddings.parameters(), 'lr': 1e-3},
        {'params': model.input_projection.parameters(), 'lr': 1e-4},
        {'params': model.encoder.parameters(), 'lr': 1e-4},
        {'params': model.duration_predictor.parameters(), 'lr': 1e-3},
        {'params': model.pitch_predictor.parameters(), 'lr': 1e-3},
        {'params': model.energy_predictor.parameters(), 'lr': 1e-3},
        {'params': model.mel_linear.parameters(), 'lr': 1e-4},
    ])
    
    # Define loss (L1 loss for mel reconstruction).
    criterion = nn.L1Loss()
    
    num_epochs = 50
    best_val_loss = float('inf')
    patience = 5
    patience_counter = 0
    
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        for batch in train_loader:
            phoneme_seqs, mel_specs, pitches, energies, speaker_ids, language_ids = batch
            phoneme_seqs = phoneme_seqs.cuda()
            mel_specs = mel_specs.cuda()
            speaker_ids = speaker_ids.cuda()
            language_ids = language_ids.cuda()
            
            optimizer.zero_grad()
            mel_out, pred_duration, pred_pitch, pred_energy = model(phoneme_seqs, language_ids, speaker_ids)
            # Compute loss on mel-spectrogram (adjust dimensions if necessary)
            loss = criterion(mel_out, mel_specs.transpose(1, 2))
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
        
        avg_train_loss = running_loss / len(train_loader)
        print(f"Epoch {epoch+1}/{num_epochs} - Training Loss: {avg_train_loss:.4f}")
        
        # Validation step (simulate if not implemented yet).
        model.eval()
        running_val_loss = 0.0
        with torch.no_grad():
            for batch in val_loader:
                phoneme_seqs, mel_specs, pitches, energies, speaker_ids, language_ids = batch
                phoneme_seqs = phoneme_seqs.cuda()
                mel_specs = mel_specs.cuda()
                speaker_ids = speaker_ids.cuda()
                language_ids = language_ids.cuda()
                
                mel_out, _, _, _ = model(phoneme_seqs, language_ids, speaker_ids)
                val_loss = criterion(mel_out, mel_specs.transpose(1, 2))
                running_val_loss += val_loss.item()
        
        avg_val_loss = running_val_loss / len(val_loader)
        print(f"Epoch {epoch+1}/{num_epochs} - Validation Loss: {avg_val_loss:.4f}")
        
        # Checkpoint saving and early stopping.
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            patience_counter = 0
            torch.save(model.state_dict(), f"checkpoint_epoch_{epoch+1}.pt")
            print("Saved new best checkpoint.")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print("Early stopping triggered.")
                break

if __name__ == "__main__":
    main()
