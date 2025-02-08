import os
import torch
import torch.nn as nn
import torch.optim as optim
import yaml
from torch.utils.data import DataLoader, WeightedRandomSampler
from tqdm import tqdm

from fastspeech2_model import FastSpeech2MultiLingual
from dataloader_for_acoustic_model import TTSDataset, dynamic_collate_fn
from data_preprocessing.generate_phoneme_inventory import get_fixed_inventory

def build_phoneme_vocab():
    fixed_inventory = get_fixed_inventory()  # Returns list of phoneme tokens.
    phoneme_vocab = {token: idx for idx, token in enumerate(fixed_inventory)}
    return phoneme_vocab

def load_config(config_path):
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config

def main():
    # Load configuration.
    config = load_config("config/espnet2_model_fastspeech2.yaml")  # You might want to rename this to reflect ESPnet2 usage.
    
    # Build phoneme vocabulary.
    phoneme_vocab = build_phoneme_vocab()
    
    # Define mapping dictionaries.
    language_map = {"english": 0, "gujarathi": 1, "bhojpuri": 2, "kannada": 3}
    speaker_map = {
        "english_f": 0, "english_m": 1, "bhojpuri_f": 2, "bhojpuri_m": 3,
        "gujarathi_f": 4, "gujarathi_m": 5, "kannada_f": 6, "kannada_m": 7
    }
    
    # Create training dataset and DataLoader.
    train_metadata = "dataset/metadata/updated_train.csv"
    train_dataset = TTSDataset(
        root_dir="dataset", 
        metadata_csv=train_metadata,
        phoneme_vocab=phoneme_vocab, 
        language_map=language_map, 
        speaker_map=speaker_map
    )
    
    # Create weighted sampling based on language frequency.
    import pandas as pd
    df_train = pd.read_csv(train_metadata, encoding="utf-8-sig")
    weights = []
    for _, row in df_train.iterrows():
        lang = row['language'].lower()
        weights.append(2.0 if lang in ["gujarathi", "bhojpuri"] else 1.0)
    weights = torch.tensor(weights, dtype=torch.float)
    sampler = WeightedRandomSampler(weights, num_samples=len(weights), replacement=True)
    
    train_loader = DataLoader(
        train_dataset, batch_size=32, collate_fn=dynamic_collate_fn, sampler=sampler
    )
    
    # Create validation DataLoader.
    val_metadata = "dataset/metadata/updated_val.csv"
    val_dataset = TTSDataset(
        root_dir="dataset", 
        metadata_csv=val_metadata,
        phoneme_vocab=phoneme_vocab, 
        language_map=language_map, 
        speaker_map=speaker_map
    )
    val_loader = DataLoader(val_dataset, batch_size=32, collate_fn=dynamic_collate_fn, shuffle=False)
    
    # Initialize the multilingual FastSpeech2 model.
    model = FastSpeech2MultiLingual(config)
    model = model.cuda()
    
    # Set up optimizer with differential learning rates:
    optimizer = optim.Adam([
        {'params': model.embeddings.parameters(), 'lr': 1e-3},
        {'params': model.input_projection.parameters(), 'lr': 1e-4},
        {'params': model.encoder.parameters(), 'lr': 1e-4},
        {'params': model.duration_predictor.parameters(), 'lr': 1e-3},
        {'params': model.pitch_predictor.parameters(), 'lr': 1e-3},
        {'params': model.energy_predictor.parameters(), 'lr': 1e-3},
        {'params': model.mel_linear.parameters(), 'lr': 1e-4},
    ])
    
    # Use L1 loss for mel-spectrogram reconstruction.
    criterion = nn.L1Loss()
    
    num_epochs = 50
    best_val_loss = float('inf')
    patience = 5
    patience_counter = 0
    
    # Training loop.
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        train_batches = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs} [Train]", leave=False)
        for batch in train_batches:
            phoneme_seqs, mel_specs, pitches, energies, speaker_ids, language_ids = batch
            phoneme_seqs = phoneme_seqs.cuda()
            mel_specs = mel_specs.cuda()
            speaker_ids = speaker_ids.cuda()
            language_ids = language_ids.cuda()
            
            optimizer.zero_grad()
            mel_out, pred_duration, pred_pitch, pred_energy = model(phoneme_seqs, language_ids, speaker_ids)
            # Adjust dimensions: model output [B, T, mel_dim]; target mel_specs is [B, n_mels, T].
            loss = criterion(mel_out, mel_specs.transpose(1, 2))
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            train_batches.set_postfix(loss=loss.item())
        
        avg_train_loss = running_loss / len(train_loader)
        print(f"Epoch {epoch+1}/{num_epochs} - Training Loss: {avg_train_loss:.4f}")
        
        # Validation step.
        model.eval()
        running_val_loss = 0.0
        val_batches = tqdm(val_loader, desc=f"Epoch {epoch+1}/{num_epochs} [Val]", leave=False)
        with torch.no_grad():
            for batch in val_batches:
                phoneme_seqs, mel_specs, pitches, energies, speaker_ids, language_ids = batch
                phoneme_seqs = phoneme_seqs.cuda()
                mel_specs = mel_specs.cuda()
                speaker_ids = speaker_ids.cuda()
                language_ids = language_ids.cuda()
                
                mel_out, _, _, _ = model(phoneme_seqs, language_ids, speaker_ids)
                val_loss = criterion(mel_out, mel_specs.transpose(1, 2))
                running_val_loss += val_loss.item()
                val_batches.set_postfix(loss=val_loss.item())
        
        avg_val_loss = running_val_loss / len(val_loader)
        print(f"Epoch {epoch+1}/{num_epochs} - Validation Loss: {avg_val_loss:.4f}")
        
        # Checkpointing and early stopping.
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            patience_counter = 0
            checkpoint_path = f"checkpoint_epoch_{epoch+1}.pt"
            torch.save(model.state_dict(), checkpoint_path)
            print(f"Saved new best checkpoint: {checkpoint_path}")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print("Early stopping triggered.")
                break

if __name__ == "__main__":
    main()
