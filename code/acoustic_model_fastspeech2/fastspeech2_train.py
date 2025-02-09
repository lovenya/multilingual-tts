#!/usr/bin/env python3

import argparse
import logging
import sys
from pathlib import Path
import pandas as pd
import torch
from torch.utils.data import DataLoader
from torch.optim import Adam
from torch.utils.data.sampler import WeightedRandomSampler
from espnet2.train.trainer import Trainer
from espnet2.torch_utils.initialize import initialize
from espnet2.utils.types import str2bool
from fastspeech2_model import FastSpeech2MultiLingual
from dataloader_for_acoustic_model import TTSDataset, build_phoneme_vocab, dynamic_collate_fn
import yaml
import os

class AttrDict(dict):
    def __init__(self, *args, **kwargs):
        super(AttrDict, self).__init__(*args, **kwargs)
        self.__dict__ = self

def get_parser():
    parser = argparse.ArgumentParser(
        description="Train FastSpeech2 model",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    
    parser.add_argument("--config", type=str, required=False,
                        help="Path to model config file")
    parser.add_argument("--gpu_id", type=int, default=0,
                        help="GPU ID for training")
    parser.add_argument("--resume", type=str2bool, default=False,
                        help="Resume training from checkpoint")
    
    return parser

def load_config(config_path):
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return AttrDict(config)

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

def setup_logging(config):
    log_dir = os.path.join(config.checkpoint_dir, "logs")
    os.makedirs(log_dir, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s (%(module)s:%(lineno)d) %(levelname)s: %(message)s",
        handlers=[
            logging.FileHandler(os.path.join(log_dir, "train.log")),
            logging.StreamHandler(sys.stdout)
        ]
    )

def build_dataloaders(config):
    phoneme_vocab = build_phoneme_vocab()
    language_map = {"english": 0, "gujarathi": 1, "bhojpuri": 2, "kannada": 3}
    speaker_map = {
        "english_f": 0, "english_m": 1, "bhojpuri_f": 2, "bhojpuri_m": 3,
        "gujarathi_f": 4, "gujarathi_m": 5, "kannada_f": 6, "kannada_m": 7
    }

    # Training dataset
    train_dataset = TTSDataset(
        root_dir="dataset",
        metadata_csv="dataset/metadata/updated_train.csv",
        phoneme_vocab=phoneme_vocab,
        language_map=language_map,
        speaker_map=speaker_map
    )

    # Handle imbalanced languages
    df_train = pd.read_csv("dataset/metadata/updated_train.csv", encoding="utf-8-sig")
    weights = [2.0 if row['language'].lower() in ['gujarathi', 'bhojpuri'] else 1.0 
              for _, row in df_train.iterrows()]
    weights = torch.tensor(weights, dtype=torch.float)
    sampler = WeightedRandomSampler(weights, num_samples=len(weights), replacement=True)

    train_loader = DataLoader(
        train_dataset,
        batch_size=16,
        collate_fn=dynamic_collate_fn,
        sampler=sampler
    )

    # Validation dataset
    val_dataset = TTSDataset(
        root_dir="dataset",
        metadata_csv="dataset/metadata/updated_val.csv",
        phoneme_vocab=phoneme_vocab,
        language_map=language_map,
        speaker_map=speaker_map
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=16,
        collate_fn=dynamic_collate_fn,
        shuffle=False
    )

    return train_loader, val_loader

def main(cmd=None):
    parser = get_parser()
    args = parser.parse_args(cmd)
    
    # Load configuration
    config = load_config("config/espnet2_model_fastspeech2.yaml")
    
    # Setup logging
    setup_logging(config)
    
    # Set random seed
    # initialize(seed=42)
    
    # Create dataloaders
    train_loader, val_loader = build_dataloaders(config)

    # Set device
    device = torch.device(f"cuda:{args.gpu_id}" if torch.cuda.is_available() else "cpu")
    
    # Create model
    model = FastSpeech2MultiLingual(config)
    model = model.to(device)
    
    # Create optimizer
    optimizer = build_optimizer(model, config)

    # Create trainer
    trainer = Trainer(
        model=model,
        optimizers=[optimizer],
        schedulers=[],
        max_epoch=50,
        train_dataloader=train_loader,
        valid_dataloader=val_loader,
        device=device,
        output_dir=Path(config.checkpoint_dir),
        resume=args.resume,
        log_interval=50,
        checkpoint_save_interval=5
    )

    # Start training
    try:
        trainer.run()
    except KeyboardInterrupt:
        logging.info("Training interrupted by user")
    except Exception as e:
        logging.error(f"Training failed with error: {str(e)}")
        raise
    finally:
        # Save the final model
        trainer.save_checkpoint("final")
        logging.info(f"Training finished. Model saved to {config.checkpoint_dir}")

if __name__ == "__main__":
    main()