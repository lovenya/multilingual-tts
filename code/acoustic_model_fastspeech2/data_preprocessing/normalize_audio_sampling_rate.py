import os
import subprocess
import json
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor

def get_sample_rate(wav_path):
    """Use ffprobe to quickly get the sample rate of an audio file."""
    try:
        result = subprocess.run(
            [
                'ffprobe', '-v', 'error',
                '-select_streams', 'a:0',
                '-show_entries', 'stream=sample_rate',
                '-of', 'json', wav_path
            ],
            capture_output=True,
            text=True,
            check=True
        )
        info = json.loads(result.stdout)
        return int(info['streams'][0]['sample_rate'])
    except Exception as e:
        print(f"Error reading sample rate for {wav_path}: {e}")
        return None

def convert_with_ffmpeg(wav_path, target_sample_rate):
    """Convert the audio file to the target sample rate using ffmpeg."""
    tmp_path = wav_path + ".tmp.wav"
    cmd = [
        'ffmpeg', '-y',        # Overwrite output file without asking
        '-i', wav_path,        # Input file
        '-ar', str(target_sample_rate),  # Set target sample rate
        tmp_path              # Output temporary file
    ]
    # (Optional) If your ffmpeg is built with multi-threading or GPU support,
    # you could add flags here. For example, you can force ffmpeg to use multiple threads:
    # cmd.extend(['-threads', '4'])
    # Note: GPU acceleration for audio is uncommon.
    
    try:
        subprocess.run(cmd, check=True, capture_output=True)
        os.replace(tmp_path, wav_path)
    except Exception as e:
        print(f"Error converting {wav_path}: {e}")
        if os.path.exists(tmp_path):
            os.remove(tmp_path)

def normalize_audio_file(file_info):
    """
    Normalize a single audio file to the target sample rate.
    Checks the sample rate using ffprobe and uses ffmpeg to convert if needed.
    """
    wav_path, target_sample_rate = file_info
    sample_rate = get_sample_rate(wav_path)
    if sample_rate is None:
        return  # Error already printed in get_sample_rate

    if sample_rate != target_sample_rate:
        convert_with_ffmpeg(wav_path, target_sample_rate)
    # Else, the file already has the desired sample rate

def normalize_audio(base_path, target_sample_rate=22050):
    """
    Normalize all .wav files in the dataset to a target sample rate using multiprocessing.
    """
    # Collect all .wav files from the dataset
    file_list = []
    languages = [
        lang for lang in os.listdir(base_path)
        if os.path.isdir(os.path.join(base_path, lang))
    ]

    for lang in languages:
        wav_folder = os.path.join(base_path, lang, "wav")
        if os.path.exists(wav_folder):
            wav_files = [
                os.path.join(wav_folder, f)
                for f in os.listdir(wav_folder)
                if f.lower().endswith(".wav")
            ]
            file_list.extend([(wav_file, target_sample_rate) for wav_file in wav_files])

    # Process files in parallel using ProcessPoolExecutor.
    # You can set max_workers if you want to fine-tune how many processes run concurrently.
    with ProcessPoolExecutor(max_workers=32) as executor:
        list(tqdm(
            executor.map(normalize_audio_file, file_list),
            total=len(file_list),
            desc="Normalizing Audio Files",
        ))

if __name__ == "__main__":
    # Set the path to the dataset directory
    dataset_path = "dataset"  # Replace with your dataset path

    # Normalize all audio files to 22050 Hz
    normalize_audio(dataset_path, target_sample_rate=22050)
