import nemo
from nemo.collections.tts import HIFIGANModel
import torchaudio
import torch
import os

def generate_waveform(model, mel_spectrogram):
    """
    Generate waveform from a mel-spectrogram.

    Parameters:
    - model: The HiFi-GAN model
    - mel_spectrogram: The input mel-spectrogram

    Returns:
    - waveform: The generated waveform
    """
    model.eval()
    with torch.no_grad():
        waveform = model.generator(mel_spectrogram)
    return waveform

def main():
    # Load the pretrained or fine-tuned HiFi-GAN model
    model = HIFIGANModel.from_pretrained("tts_en_hifigan")
    model = model.cuda()

    # Example input mel-spectrogram (replace this with actual data)
    mel_spectrogram = torch.randn(1, 80, 100)  # Example size (1, num_mels, time_steps)
    
    # Generate waveform
    waveform = generate_waveform(model, mel_spectrogram)
    
    # Save the generated waveform
    output_file = 'generated_audio.wav'
    torchaudio.save(output_file, waveform, 22050)
    print(f"Generated waveform saved to {output_file}")

if __name__ == "__main__":
    main()
