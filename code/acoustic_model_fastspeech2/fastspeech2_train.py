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
import tensorboard



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
        root_dir=".",
        metadata_csv="dataset/metadata/updated_train.csv",
        phoneme_vocab=phoneme_vocab,
        language_map=language_map,
        speaker_map=speaker_map
    )

    # Handle imbalanced languages
    df_train = pd.read_csv("dataset/metadata/updated_train.csv", encoding="utf-8-sig")
    print("Sample audio filepath:", df_train['audio_filepath'].iloc[0])
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
        root_dir=".",
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
    
    # Create dataloaders
    train_loader, val_loader = build_dataloaders(config)

    # Set device
    device = torch.device(f"cuda:{args.gpu_id}" if torch.cuda.is_available() else "cpu")
    
    # Create model
    model = FastSpeech2MultiLingual(config)
    model = model.to(device)
    
    # Create optimizer
    optimizer = build_optimizer(model, config)

    trainer_options = {
    "max_epoch": config['train_config'].get('num_epochs', 50),
    "grad_clip": config['train_config'].get('grad_clip', 1.0),
    "accum_grad": config['train_config'].get('accum_grad', 1),
    "no_forward_run": False,
    "ngpu": 1 if torch.cuda.is_available() else 0,
    "use_amp": False,
    "output_dir": str(Path(config.checkpoint_dir)),
    "log_interval": config['train_config'].get('log_interval', 50),
    "resume": args.resume,
    # Add the missing required parameters
    "train_dtype": "float32",  # Added
    "grad_noise": False,       # Added
    "grad_clip_type": 2.0,    # Added
    # Rest of the parameters
    "use_matplotlib": False,
    "use_tensorboard": True,
    # "tensorboard_dir": str(Path(config.checkpoint_dir) / "tensorboard"),
    "use_wandb": False,
    "save_strategy": "epoch",
    "seed": 0,
    "patience": None,
    "keep_nbest_models": 5,
    "nbest_averaging_interval": 0,
    "early_stopping_criterion": [],
    "best_model_criterion": [],
    "val_scheduler_criterion": [],
    "unused_parameters": False,
    "wandb_model_log_interval": 0,
    "create_graph_in_tensorboard": False,
    "use_adapter": False,
    "adapter": "",
    "sharded_ddp": False
    }
    
    # os.makedirs(trainer_options["tensorboard_dir"], exist_ok=True)
    
    
    # Convert dictionary to TrainerOptions dataclass
    from espnet2.train.trainer import TrainerOptions
    trainer_options = TrainerOptions(**trainer_options)

    # Create distributed option (kept separate from trainer options)
    from espnet2.train.distributed_utils import DistributedOption
    distributed_option = DistributedOption(
        distributed=False,
        ngpu=1 if torch.cuda.is_available() else 0,
        dist_rank=0,
        dist_world_size=1,
        dist_master_addr="localhost",
        dist_master_port=None,
        dist_launcher="none",
        multiprocessing_distributed=False,
    )

    # Create iterators
    from espnet2.iterators.sequence_iter_factory import SequenceIterFactory
    
    batch_size = config['train_config'].get('batch_size', 16)
    train_batches = []
    for i in range(0, len(train_loader.dataset), batch_size):
        batch_indices = list(range(i, min(i + batch_size, len(train_loader.dataset))))
        train_batches.append(batch_indices)

    # Create batches for validation data
    val_batches = []
    for i in range(0, len(val_loader.dataset), batch_size):
        batch_indices = list(range(i, min(i + batch_size, len(val_loader.dataset))))
        val_batches.append(batch_indices)

    # Create iterators with correct parameters
    train_iter_factory = SequenceIterFactory(
        dataset=train_loader.dataset,
        batches=train_batches,
        collate_fn=dynamic_collate_fn,
        num_workers=4,
        seed=0,
        shuffle=True,
        pin_memory=True
    )

    valid_iter_factory = SequenceIterFactory(
        dataset=val_loader.dataset,
        batches=val_batches,
        collate_fn=dynamic_collate_fn,
        num_workers=4,
        seed=0,
        shuffle=False,
        pin_memory=True
    )

    # Run training using class method
    Trainer.run(
        model=model,
        optimizers=[optimizer],
        schedulers=[None],
        train_iter_factory=train_iter_factory,
        valid_iter_factory=valid_iter_factory,
        plot_attention_iter_factory=None,
        trainer_options=trainer_options,
        distributed_option=distributed_option,
    )
    train_iter_factory = SequenceIterFactory(
        dataset=train_loader.dataset,
        batch_size=config['train_config'].get('batch_size', 16),
        collate_fn=train_loader.collate_fn,
        sampler=train_loader.sampler,
    )
    valid_iter_factory = SequenceIterFactory(
        dataset=val_loader.dataset,
        batch_size=config['train_config'].get('batch_size', 16),
        collate_fn=val_loader.collate_fn,
        shuffle=False,
    )

    # Run training using class method
    Trainer.run(
        model=model,
        optimizers=[optimizer],
        schedulers=[None],
        train_iter_factory=train_iter_factory,
        valid_iter_factory=valid_iter_factory,
        plot_attention_iter_factory=None,
        trainer_options=trainer_options,
        distributed_option=distributed_option,
    )

if __name__ == "__main__":
    main()