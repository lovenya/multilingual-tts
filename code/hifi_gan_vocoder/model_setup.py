import nemo
import nemo.collections.asr as nemo_asr
from omegaconf import OmegaConf
from nemo.utils import logging
import torch

def setup_hifi_gan_model(config_path, checkpoint_path=None):
    """
    Set up the HiFi-GAN V1 model using NeMo.
    
    Parameters:
    - config_path: Path to the HiFi-GAN configuration YAML file.
    - checkpoint_path: Path to the pretrained HiFi-GAN model checkpoint (if fine-tuning).
    """
    # Load the HiFi-GAN configuration
    config = OmegaConf.load(config_path)
    
    # Initialize the HiFi-GAN model
    model = nemo_asr.models.HIFIGANModel(cfg=config)

    if checkpoint_path:
        # Load pretrained model weights for fine-tuning
        model.restore_from(restore_path=checkpoint_path)
        logging.info(f"Loaded checkpoint from {checkpoint_path}")
    else:
        logging.info("Initializing HiFi-GAN from scratch")

    return model

if __name__ == "__main__":
    # Path to the HiFi-GAN config YAML file
    config_path = 'path/to/hifi_gan_config.yaml'  # Update this path

    # Optionally provide a pretrained model checkpoint for fine-tuning
    checkpoint_path = 'path/to/pretrained_model.ckpt'  # Update this path (optional)

    # Set up the HiFi-GAN model
    model = setup_hifi_gan_model(config_path, checkpoint_path)

    # Check if CUDA is available and move the model to GPU if needed
    if torch.cuda.is_available():
        model = model.cuda()

    logging.info("HiFi-GAN model setup complete.")
