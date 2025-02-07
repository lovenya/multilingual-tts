import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d

def interpolate_array(arr, target_length):
    """
    Interpolates a 1D numpy array to a new target length using linear interpolation.
    
    Args:
        arr (np.ndarray): Input array.
        target_length (int): Desired output length.
    
    Returns:
        np.ndarray: Interpolated array.
    """
    original_length = len(arr)
    if original_length == target_length:
        return arr
    interp_func = interp1d(np.linspace(0, 1, original_length), arr, kind='linear')
    return interp_func(np.linspace(0, 1, target_length))

def plot_alignment(mel_spec, pitch, energy):
    """
    Plots the mel-spectrogram along with pitch and energy contours.
    
    Args:
        mel_spec (np.ndarray): Mel-spectrogram, shape (n_mels, T).
        pitch (np.ndarray): Pitch contour.
        energy (np.ndarray): Energy contour.
    """
    plt.figure(figsize=(16, 6))
    
    plt.subplot(3, 1, 1)
    plt.imshow(mel_spec, aspect='auto', origin='lower')
    plt.title("Mel-Spectrogram")
    plt.colorbar()
    
    plt.subplot(3, 1, 2)
    plt.plot(pitch)
    plt.title("Pitch")
    
    plt.subplot(3, 1, 3)
    plt.plot(energy)
    plt.title("Energy")
    
    plt.tight_layout()
    plt.show()
