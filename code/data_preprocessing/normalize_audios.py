import os
from pydub import AudioSegment
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor


def normalize_audio_file(file_info):
    """
    Normalize a single audio file to the target sample rate.
    """
    wav_path, target_sample_rate = file_info
    try:
        audio = AudioSegment.from_file(wav_path)
        if audio.frame_rate != target_sample_rate:
            normalized_audio = audio.set_frame_rate(target_sample_rate)
            normalized_audio.export(wav_path, format="wav")
    except Exception as e:
        print(f"Error processing {wav_path}: {e}")


def normalize_audio(base_path, target_sample_rate=16000):
    """
    Normalize all .wav files in the dataset to a target sample rate using multiprocessing.
    """
    # Collect all .wav files from the dataset
    file_list = []
    languages = [
        lang
        for lang in os.listdir(base_path)
        if os.path.isdir(os.path.join(base_path, lang))
    ]

    for lang in languages:
        wav_folder = os.path.join(base_path, lang, "wav")
        if os.path.exists(wav_folder):
            wav_files = [
                os.path.join(wav_folder, f)
                for f in os.listdir(wav_folder)
                if f.endswith(".wav")
            ]
            file_list.extend([(wav_file, target_sample_rate) for wav_file in wav_files])

    # Process files in parallel using ProcessPoolExecutor
    with ProcessPoolExecutor() as executor:
        list(
            tqdm(
                executor.map(normalize_audio_file, file_list),
                total=len(file_list),
                desc="Normalizing Audio Files",
            )
        )


if __name__ == "__main__":
    # Set the path to the dataset directory
    dataset_path = "dataset"  # Replace with your dataset path

    # Normalize all audio files to 16kHz
    normalize_audio(dataset_path, target_sample_rate=16000)
