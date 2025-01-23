import os
from pydub import AudioSegment
from tqdm import tqdm


def normalize_audio(base_path, target_sample_rate=16000):
    """
    Normalize all .wav files in the dataset to a target sample rate.
    """
    # Get all language folders
    languages = [
        lang
        for lang in os.listdir(base_path)
        if os.path.isdir(os.path.join(base_path, lang))
    ]

    for lang in tqdm(languages, desc="Processing Languages"):
        wav_folder = os.path.join(base_path, lang, "wav")

        if os.path.exists(wav_folder):
            wav_files = [f for f in os.listdir(wav_folder) if f.endswith(".wav")]

            for wav_file in tqdm(wav_files, desc=f"Normalizing {lang}", leave=False):
                wav_path = os.path.join(wav_folder, wav_file)

                # Load the audio file
                audio = AudioSegment.from_file(wav_path)

                # Check and normalize sample rate
                if audio.frame_rate != target_sample_rate:
                    normalized_audio = audio.set_frame_rate(target_sample_rate)

                    # Export the normalized audio
                    normalized_audio.export(wav_path, format="wav")


if __name__ == "__main__":
    # Set the path to the dataset directory
    dataset_path = "dataset"  # Replace with your dataset path

    # Normalize all audio files to 16kHz
    normalize_audio(dataset_path, target_sample_rate=16000)
