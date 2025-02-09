import torch

from fastspeech2_train import load_config

config = load_config("config/espnet2_model_fastspeech2.yaml")
checkpoint = torch.load(config["pretrained_encoder_path"], map_location="cpu")
print("Checkpoint keys:", checkpoint.keys())
