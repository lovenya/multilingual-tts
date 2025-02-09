import torchaudio
import numpy as np
from pystoi import stoi  # For Speech Quality Metric (STOI)

def evaluate_pesq(original_wav, generated_wav, sample_rate=16000):
    """
    Evaluate the perceptual quality of the generated audio using PESQ.
    """
    # Note: You'll need to install the PESQ library, e.g., with `pip install pesq`
    from pesq import pesq
    return pesq(sample_rate, original_wav, generated_wav)

def evaluate_mcd(original_wav, generated_wav):
    """
    Calculate Mel-Cepstral Distortion (MCD).
    """
    # Example MCD computation using librosa
    import librosa
    orig_mel = librosa.feature.melspectrogram(y=original_wav, sr=16000)
    gen_mel = librosa.feature.melspectrogram(y=generated_wav, sr=16000)
    mcd = np.mean(np.square(orig_mel - gen_mel))
    return mcd

if __name__ == "__main__":
    original_wav_path = 'path/to/original.wav'
    generated_wav_path = 'path/to/generated.wav'

    original_wav, _ = torchaudio.load(original_wav_path)
    generated_wav, _ = torchaudio.load(generated_wav_path)

    pesq_score = evaluate_pesq(original_wav.numpy(), generated_wav.numpy())
    mcd_score = evaluate_mcd(original_wav.numpy(), generated_wav.numpy())

    print(f"PESQ Score: {pesq_score}")
    print(f"MCD Score: {mcd_score}")
