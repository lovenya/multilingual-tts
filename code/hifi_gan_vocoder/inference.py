import nemo
from nemo.collections.asr import HIFIGANModel
import torchaudio
import torch

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

if __name__ == "__main__":
    # Load the pretrained model
    model = HIFIGANModel.from_pretrained('path/to/hifi_gan_checkpoint.ckpt')
    model = model.cuda()
    
    # Example input mel-spectrogram
    mel_spectrogram = torch.randn(1, 80, 100)  # Example size (1, num_mels, time_steps)
    
    # Generate waveform
    waveform = generate_waveform(model, mel_spectrogram)
    
    # Save the generated waveform
    torchaudio.save('generated_audio.wav', waveform, 16000)
    print("Generated waveform saved to 'generated_audio.wav'")
