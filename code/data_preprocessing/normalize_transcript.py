import os
import re
import inflect
from tqdm import tqdm
from indicnlp.transliterate.unicode_transliterate import ItransTransliterator

# Initialize the inflect engine for English number-to-text
inflect_engine = inflect.engine()


def convert_numbers_to_text(text, language="en"):
    """
    Convert numeric expressions to written-out text for a given language.
    """
    if language == "en":
        # Convert numbers to text using inflect
        return re.sub(r"\d+", lambda x: inflect_engine.number_to_words(x.group()), text)
    elif language in {"gu", "bh", "kn"}:
        # Use Indic NLP transliteration for non-English numbers
        number_map = (
            {
                "0": "૦",
                "1": "૧",
                "2": "૨",
                "3": "૩",
                "4": "૪",  # Gujarati
                "5": "૫",
                "6": "૬",
                "7": "૭",
                "8": "૮",
                "9": "૯",
            }
            if language == "gu"
            else (
                {
                    "0": "೦",
                    "1": "೧",
                    "2": "೨",
                    "3": "೩",
                    "4": "೪",  # Kannada
                    "5": "೫",
                    "6": "೬",
                    "7": "೭",
                    "8": "೮",
                    "9": "೯",
                }
                if language == "kn"
                else {
                    "0": "०",
                    "1": "१",
                    "2": "२",
                    "3": "३",
                    "4": "४",  # Bhojpuri (Devanagari)
                    "5": "५",
                    "6": "६",
                    "7": "७",
                    "8": "८",
                    "9": "९",
                }
            )
        )
        # Replace each digit with its corresponding language-specific character
        for digit, written_form in number_map.items():
            text = text.replace(digit, written_form)
        return text
    else:
        return text  # Default: no conversion


def normalize_text(text, language="en"):
    """
    Normalize a given transcript:
    - Convert numbers to text.
    - Retain punctuation and remove only illegal characters.
    """
    # Convert numbers to text
    text = convert_numbers_to_text(text, language=language)

    # Remove illegal characters but retain punctuation
    text = re.sub(
        r"[^\w\s.,?!]", "", text
    )  # Retain letters, digits, spaces, and punctuation (.,?!)

    # Remove extra whitespace
    text = re.sub(r"\s+", " ", text).strip()

    return text


def normalize_transcripts(base_path):
    """
    Normalize all transcripts in the dataset.
    """
    # Traverse the dataset folders
    languages = [
        lang
        for lang in os.listdir(base_path)
        if os.path.isdir(os.path.join(base_path, lang))
    ]

    for lang in tqdm(languages, desc="Processing Languages"):
        txt_folder = os.path.join(base_path, lang, "txt")

        if os.path.exists(txt_folder):
            txt_files = sorted(
                [f for f in os.listdir(txt_folder) if f.endswith(".txt")]
            )
            language_code = lang.split("_")[
                0
            ]  # Extract language code (e.g., "en", "gu", "kn", "bh")

            for txt_file in tqdm(txt_files, desc=f"Normalizing {lang}", leave=False):
                txt_path = os.path.join(txt_folder, txt_file)

                # Read and normalize the transcript
                with open(txt_path, "r", encoding="utf-8") as f:
                    original_text = f.read().strip()

                # Normalize text
                normalized_text = normalize_text(original_text, language=language_code)

                # Overwrite the file with normalized text
                with open(txt_path, "w", encoding="utf-8") as f:
                    f.write(normalized_text)


if __name__ == "__main__":
    # Set the path to the dataset directory
    dataset_path = "dataset"  # Replace with your dataset path

    # Normalize all transcripts
    normalize_transcripts(dataset_path)
