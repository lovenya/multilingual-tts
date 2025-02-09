import pandas as pd
import torch
from torch.utils.data import DataLoader
from torch.optim import Adam
from torch.utils.data.sampler import WeightedRandomSampler
from espnet2.train.trainer import Trainer
from espnet2.tasks.tts import TTSTask
from fastspeech2_model import FastSpeech2MultiLingual
from dataloader_for_acoustic_model import TTSDataset, build_phoneme_vocab, dynamic_collate_fn
import yaml
import os

def load_config(config_path):
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def build_optimizer(model, config):
    optimizer = Adam([
        {'params': model.embeddings.parameters(), 'lr': 1e-3},
        {'params': model.input_projection.parameters(), 'lr': 1e-4},
        {'params': model.espnet_fastspeech2.parameters(), 'lr': 1e-4},
        {'params': model.duration_predictor.parameters(), 'lr': 1e-3},
        {'params': model.pitch_predictor.parameters(), 'lr': 1e-3},
        {'params': model.energy_predictor.parameters(), 'lr': 1e-3},
        {'params': model.mel_linear.parameters(), 'lr': 1e-4},
    ], betas=(0.9, 0.98), eps=1e-9)
    return optimizer

def main():
    config = load_config('config/espnet2_model_fastspeech2.yaml')
    phoneme_vocab = build_phoneme_vocab()  # Assumes this function is already defined

    language_map = {"english": 0, "gujarathi": 1, "bhojpuri": 2, "kannada": 3}
    speaker_map = {"english_f": 0, "english_m": 1, "bhojpuri_f": 2, "bhojpuri_m": 3,
                   "gujarathi_f": 4, "gujarathi_m": 5, "kannada_f": 6, "kannada_m": 7}

    # Create training and validation datasets
    train_metadata = "dataset/metadata/updated_train.csv"
    train_dataset = TTSDataset(
        root_dir="dataset",
        metadata_csv=train_metadata,
        phoneme_vocab=phoneme_vocab,
        language_map=language_map,
        speaker_map=speaker_map
    )

    # Use weighted sampling to handle imbalanced languages
    df_train = pd.read_csv(train_metadata, encoding="utf-8-sig")
    weights = [2.0 if row['language'].lower() in ['gujarathi', 'bhojpuri'] else 1.0 for _, row in df_train.iterrows()]
    weights = torch.tensor(weights, dtype=torch.float)
    sampler = WeightedRandomSampler(weights, num_samples=len(weights), replacement=True)

    train_loader = DataLoader(
        train_dataset, batch_size=config['train_config']['batch_size'], collate_fn=dynamic_collate_fn, sampler=sampler
    )

    # Validation DataLoader
    val_metadata = "dataset/metadata/updated_validation.csv"
    val_dataset = TTSDataset(
        root_dir="dataset",
        metadata_csv=val_metadata,
        phoneme_vocab=phoneme_vocab,
        language_map=language_map,
        speaker_map=speaker_map
    )
    val_loader = DataLoader(val_dataset, batch_size=config['train_config']['batch_size'], collate_fn=dynamic_collate_fn, shuffle=False)

    # Initialize the model
     # Initialize the model and optimizer
    model = FastSpeech2MultiLingual(config)
    model = model.cuda()
    optimizer = build_optimizer(model, config)

    # Initialize Trainer with the correct parameter names
    # Initialize Trainer with the correct parameter names
    # Initialize Trainer with the correct parameter names
    trainer = Trainer(
        iterators={
            "main": train_loader,
            "valid": val_loader
        },
        model=model,
        optimizers={"main": optimizer},
        schedulers={},  # Empty dict since we're not using schedulers
        max_epoch=config['train_config']['num_epochs'],
        reporter=None,
        scaler=None,
        options={
            "train": {
                "ngpu": 1,
                "grad_clip": 1.0,
                "accum_grad": 1,
                "log_interval": config['train_config'].get('log_interval', 50),
            },
            "save_interval_epochs": config['train_config'].get('checkpoint_save_interval', 5),
            "output_dir": os.path.join(config.get('checkpoint_dir', 'exp/train')),
            "device": "cuda",
            "best_model_criterion": [("valid/loss", "min")],
            "keep_nbest_models": 5,
            "early_stopping_criterion": None,
        },
        distributed_option=None
    )

    # Run the training process
    trainer.run()

if __name__ == '__main__':
    main()