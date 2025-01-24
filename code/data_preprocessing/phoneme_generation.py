from tqdm import tqdm
import os
from phonemizer.phonemize import phonemize
from phonemizer.backend.espeak.wrapper import EspeakWrapper
from concurrent.futures import ProcessPoolExecutor

# Environment setup for eSpeak-NG
os.environ["PATH"] += r";C:\Program Files\eSpeak NG"
os.environ["ESPEAK_DATA_PATH"] = r"C:\Program Files\eSpeak NG\espeak-ng-data"
EspeakWrapper.set_library("C:\\Program Files\\eSpeak NG\\libespeak-ng.dll")

# Language code mapping for Phonemizer
LANGUAGE_MAPPING = {
    "en": "en-us",  # English (US)
    "gu": "gu",  # Gujarati
    "kn": "kn",  # Kannada
    "bh": "hi",  # Bhojpuri (fallback to Hindi)
}


def phonemize_text(text, language_code):
    """
    Convert text to phonemes using the Phonemizer library.
    """
    phonemizer_lang = LANGUAGE_MAPPING.get(language_code, "en-us")
    try:
        phonemes = phonemize(
            text,
            language=phonemizer_lang,
            backend="espeak",
            strip=True,
            preserve_punctuation=True,
            with_stress=True,
        )
        return phonemes
    except Exception as e:
        print(f"Error phonemizing text for language '{language_code}': {e}")
        return ""


def process_file(args):
    """
    Process a single file: read text, phonemize, and save phonemes.
    :param args: Tuple of (txt_path, phoneme_path, language_code)
    """
    txt_path, phoneme_path, language_code = args
    with open(txt_path, "r", encoding="utf-8") as f:
        text = f.read().strip()

    # Generate phonemes
    phonemes = phonemize_text(text, language_code)

    # Save phonemes to a new file
    with open(phoneme_path, "w", encoding="utf-8") as f:
        f.write(phonemes)


def phonemize_transcripts(base_path):
    """
    Phonemize all transcripts in the dataset using multiprocessing.
    """
    languages = [
        lang
        for lang in os.listdir(base_path)
        if os.path.isdir(os.path.join(base_path, lang))
    ]

    for lang in languages:
        txt_folder = os.path.join(base_path, lang, "txt")
        phoneme_folder = os.path.join(base_path, lang, "phonemes")
        os.makedirs(phoneme_folder, exist_ok=True)

        if os.path.exists(txt_folder):
            txt_files = sorted(
                [f for f in os.listdir(txt_folder) if f.endswith(".txt")]
            )
            language_code = lang.split("_")[0]

            # Prepare arguments for multiprocessing
            args_list = [
                (
                    os.path.join(txt_folder, txt_file),
                    os.path.join(phoneme_folder, txt_file),
                    language_code,
                )
                for txt_file in txt_files
            ]

            # Use ProcessPoolExecutor for multiprocessing
            with ProcessPoolExecutor(max_workers=6) as executor:
                list(
                    tqdm(
                        executor.map(process_file, args_list),
                        total=len(args_list),
                        desc=f"Processing {lang}",
                    )
                )


if __name__ == "__main__":
    # Set the path to the dataset directory
    dataset_path = "dataset"  # Replace with your dataset path

    # Phonemize all transcripts
    phonemize_transcripts(dataset_path)
