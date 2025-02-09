import torchaudio
import numpy as np
from pesq import pesq
import librosa

def evaluate_pesq(original_wav, generated_wav, sample_rate=22050):
    """
    Evaluate the perceptual quality of the generated audio using PESQ.
    """
    return pesq(sample_rate, original_wav, generated_wav)

def evaluate_mcd(original_wav, generated_wav):
    """
    Calculate Mel-Cepstral Distortion (MCD) between the original and generated waveforms.
    """
    orig_mel = librosa.feature.melspectrogram(y=original_wav, sr=22050)
    gen_mel = librosa.feature.melspectrogram(y=generated_wav, sr=22050)
    mcd = np.mean(np.square(orig_mel - gen_mel))
    return mcd

def main():
    original_wav_path = 'path/to/original.wav'
    generated_wav_path = 'path/to/generated.wav'

    original_wav, _ = torchaudio.load(original_wav_path)
    generated_wav, _ = torchaudio.load(generated_wav_path)

    # Evaluate PESQ and MCD
    pesq_score = evaluate_pesq(original_wav.numpy(), generated_wav.numpy())
    mcd_score = evaluate_mcd(original_wav.numpy(), generated_wav.numpy())

    print(f"PESQ Score: {pesq_score}")
    print(f"MCD Score: {mcd_score}")

if __name__ == "__main__":
    main()
