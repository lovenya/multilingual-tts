import argparse
import os
import yaml
import torch
from torch.utils.data import DataLoader, WeightedRandomSampler
import pandas as pd

# Import ESPnet2's Trainer and TTSTask interfaces
from espnet2.train.trainer import Trainer
from espnet2.tasks.tts import TTSTask

from dataloader_for_acoustic_model import TTSDataset, dynamic_collate_fn
from fastspeech2_model import FastSpeech2MultiLingual


# For now, due to import issues, we include get_fixed_inventory() here directly.
def get_fixed_inventory():
    inventory = [
        # English (en-us)
        "(en-us) p", "(en-us) b", "(en-us) t", "(en-us) d", "(en-us) k", "(en-us) g",
        "(en-us) f", "(en-us) v", "(en-us) θ", "(en-us) ð", "(en-us) s", "(en-us) z",
        "(en-us) ʃ", "(en-us) ʒ", "(en-us) h", "(en-us) tʃ", "(en-us) dʒ", "(en-us) m",
        "(en-us) n", "(en-us) ŋ", "(en-us) l", "(en-us) r", "(en-us) j", "(en-us) w",
        "(en-us) i", "(en-us) ɪ", "(en-us) e", "(en-us) ɛ", "(en-us) æ", "(en-us) ʌ",
        "(en-us) ɑ", "(en-us) ɒ", "(en-us) ɔ", "(en-us) o", "(en-us) ʊ", "(en-us) u",
        "(en-us) aɪ", "(en-us) aʊ", "(en-us) ɔɪ", "(en-us) eɪ", "(en-us) oʊ",
        # Bhojpuri (hi)
        "(hi) p", "(hi) b", "(hi) t̪", "(hi) d̪", "(hi) ʈ", "(hi) ɖ", "(hi) k", "(hi) g",
        "(hi) tʃ", "(hi) dʒ", "(hi) f", "(hi) s", "(hi) h", "(hi) m", "(hi) n",
        "(hi) ɳ", "(hi) n̪", "(hi) l", "(hi) r", "(hi) j",
        "(hi) ə", "(hi) a", "(hi) ɪ", "(hi) i", "(hi) ʊ", "(hi) u",
        "(hi) e", "(hi) o", "(hi) ɛ", "(hi) ɔ", "(hi) ɒ",
        # Gujarati (gu)
        "(gu) p", "(gu) b", "(gu) t̪", "(gu) d̪", "(gu) ʈ", "(gu) ɖ", "(gu) k", "(gu) g",
        "(gu) tʃ", "(gu) dʒ", "(gu) f", "(gu) s", "(gu) h", "(gu) m", "(gu) n",
        "(gu) ɳ", "(gu) n̪", "(gu) l", "(gu) r", "(gu) j",
        "(gu) ə", "(gu) a", "(gu) ɪ", "(gu) i", "(gu) ʊ", "(gu) u",
        "(gu) e", "(gu) o", "(gu) ɛ", "(gu) ɔ", "(gu) ɒ",
        # Kannada (kn)
        "(kn) p", "(kn) b", "(kn) t", "(kn) d", "(kn) ʈ", "(kn) ɖ", "(kn) k", "(kn) g",
        "(kn) tʃ", "(kn) dʒ", "(kn) f", "(kn) s", "(kn) h", "(kn) m", "(kn) n",
        "(kn) ɳ", "(kn) n̪", "(kn) l", "(kn) r", "(kn) j",
        "(kn) ə", "(kn) a", "(kn) ɪ", "(kn) i", "(kn) ʊ", "(kn) u",
        "(kn) e", "(kn) o", "(kn) ɛ", "(kn) ɔ", "(kn) ɒ",
    ]
    return inventory


def build_phoneme_vocab():
    fixed_inventory = get_fixed_inventory()
    phoneme_vocab = {token: idx for idx, token in enumerate(fixed_inventory)}
    return phoneme_vocab


def load_config(config_path):
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


from espnet2.tasks.tts import TTSTask



class CustomTTSTask(TTSTask):
    @classmethod
    def build_model(cls, args):
        # This method should create and return the model.
        model = args.model
        return model

    @classmethod
    def build_optimizers(cls, model):
        # Create optimizers for the model
        optimizer = torch.optim.Adam([
            {'params': model.embeddings.parameters(), 'lr': 1e-3},
            {'params': model.input_projection.parameters(), 'lr': 1e-4},
            {'params': model.encoder.parameters(), 'lr': 1e-4},
            {'params': model.duration_predictor.parameters(), 'lr': 1e-3},
            {'params': model.pitch_predictor.parameters(), 'lr': 1e-3},
            {'params': model.energy_predictor.parameters(), 'lr': 1e-3},
            {'params': model.mel_linear.parameters(), 'lr': 1e-4},
        ])
        return optimizer

    @classmethod
    def build(cls, args):
        """
        Custom build method to return model, optimizer, and schedulers.
        This method is used by the Trainer class.
        """
        model = cls.build_model(args)
        optimizers = cls.build_optimizers(model)
        schedulers = None  # If you have schedulers, create and add them here
        return model, optimizers, schedulers

    @classmethod
    def add_task_arguments(cls, parser):
        """
        Add arguments specific to the task.
        This method is where you can add any task-specific arguments to the parser.
        Ensure that 'required' is initialized as a list before adding to it.
        """
        # Ensure 'required' is initialized as an empty list
        if not hasattr(cls, 'required'):
            cls.required = []
        
        parser = super().add_task_arguments(parser)  # Call the parent class method
        cls.required += ["token_list"]  # Add the argument to the list
        return parser


def main():
    # Load configuration
    config = load_config("config/espnet2_model_fastspeech2.yaml")
        # Example mappings (replace with your actual mappings)
    phoneme_vocab = build_phoneme_vocab()  # Extend to your full phoneme vocabulary (from fixed inventory)
    language_map = {"english": 0, "gujarathi": 1, "bhojpuri": 2, "kannada": 3}
    speaker_map = {"english_f": 0, "english_m": 1, "bhojpuri_f": 2, "bhojpuri_m": 3,
                   "gujarathi_f": 4, "gujarathi_m": 5, "kannada_f": 6, "kannada_m": 7}
    
    # Create argument parser and add task-specific arguments
    import argparse
    parser = argparse.ArgumentParser()
    parser = CustomTTSTask.add_task_arguments(parser)
    args = parser.parse_args()
    
    # Add model and other necessary attributes to args
    args.model = FastSpeech2MultiLingual(config).cuda()
    args.max_epoch = 50
    args.ngpu = 1
    args.train_dtype = "float32"
    args.grad_clip = 1.0
    args.accum_grad = 1
    args.batch_bins = 1000000  # Adjust as needed
    args.iterator_type = "sequence"
    
    # Calculate weights for the sampler
    train_metadata = "dataset/metadata/updated_train.csv"
    df_train = pd.read_csv(train_metadata, encoding="utf-8-sig")
    weights = []
    for _, row in df_train.iterrows():
        lang = row['language'].lower()
        weights.append(2.0 if lang in ["gujarathi", "bhojpuri"] else 1.0)
    weights = torch.tensor(weights, dtype=torch.float)

    # Create datasets
    train_dataset = TTSDataset(
        root_dir="dataset",
        metadata_csv=train_metadata,
        phoneme_vocab=phoneme_vocab,
        language_map=language_map,
        speaker_map=speaker_map
    )
    
    val_dataset = TTSDataset(
        root_dir="dataset",
        metadata_csv="dataset/metadata/updated_validation.csv",
        phoneme_vocab=phoneme_vocab,
        language_map=language_map,
        speaker_map=speaker_map
    )

    # Create dataloaders
    train_loader = DataLoader(
        train_dataset, 
        batch_size=32, 
        collate_fn=dynamic_collate_fn,
        sampler=WeightedRandomSampler(weights, num_samples=len(weights), replacement=True)
    )
    
    val_loader = DataLoader(
        val_dataset, 
        batch_size=32, 
        collate_fn=dynamic_collate_fn,
        shuffle=False
    )

    # Build task
    task = CustomTTSTask.build(args)

    # Create trainer
# Pass the task directly as a class reference
    trainer = Trainer(
        task=CustomTTSTask,  # Pass the class reference here
        train_loader=train_loader,
        valid_loader=val_loader,
        max_epoch=50,
        expdir="exp/tts_train",
        resume=False,
        log_interval=50,
        verbose=True
    )


    # Run training
    trainer.run()

if __name__ == "__main__":
    main()