import os
import logging
import traceback
import psutil
from tqdm import tqdm
from phonemizer.phonemize import phonemize
from phonemizer.backend.espeak.wrapper import EspeakWrapper
from concurrent.futures import ProcessPoolExecutor, as_completed
import logging

# Configure logging
logging.basicConfig(
    level=logging.WARNING,
    format="%(asctime)s - %(levelname)s: %(message)s",
    handlers=[
        logging.FileHandler("phonemizer.log"),
    ],
)
logger = logging.getLogger(__name__)

# Language code mapping for Phonemizer
LANGUAGE_MAPPING = {
    "en": "en-us",  # English (US)
    "gu": "gu",     # Gujarati
    "bh": "hi",     # Bhojpuri (fallback to Hindi)
    "bn": "bn",     # Bengali
}

def get_available_memory():
    """Check available system memory."""
    return psutil.virtual_memory().available / (1024 * 1024)

def phonemize_text(text, language_code, chunk_size=5000):
    """Convert text to phonemes using the Phonemizer library with chunking."""
    phonemizer_lang = LANGUAGE_MAPPING.get(language_code, "en-us")
    
    try:
        # Split text into chunks to manage memory
        chunks = [text[i:i + chunk_size] for i in range(0, len(text), chunk_size)]
        phonemized_chunks = []

        for chunk in chunks:
            try:
                # Add language tag before phonemization
                phonemized_chunk = f"({language_code})" + phonemize(
                    chunk,
                    language=phonemizer_lang,
                    backend="espeak",
                    strip=True,
                    preserve_punctuation=True,
                    with_stress=True,
                )
                phonemized_chunks.append(phonemized_chunk)
            except Exception as chunk_error:
                logger.warning(f"Error phonemizing chunk for language '{language_code}': {chunk_error}")
                phonemized_chunks.append("")

        return " ".join(phonemized_chunks)

    except Exception as e:
        logger.error(f"Unexpected error phonemizing text for language '{language_code}': {e}")
        return ""

def process_file(args):
    """Process a single file: read text, phonemize, and save phonemes."""
    try:
        txt_path, phoneme_path, language_code = args

        # Read file with robust encoding handling
        for encoding in ["utf-8", "utf-8-sig", "latin1", "iso-8859-1"]:
            try:
                with open(txt_path, "r", encoding=encoding) as f:
                    text = f.read().strip()
                break
            except UnicodeDecodeError:
                continue
        else:
            logger.error(f"Could not decode file: {txt_path}")
            return False

        # Generate phonemes
        phonemes = phonemize_text(text, language_code)

        # Save phonemes
        with open(phoneme_path, "w", encoding="utf-8") as f:
            f.write(phonemes)

        logger.info(f"Successfully processed: {txt_path}")
        return True

    except Exception as e:
        logger.error(f"Error processing {txt_path}: {traceback.format_exc()}")
        return False

def phonemize_transcripts(base_path, max_memory_threshold=75):
    """Phonemize all transcripts in the dataset using adaptive multiprocessing."""
    languages = [lang for lang in os.listdir(base_path) 
                if os.path.isdir(os.path.join(base_path, lang))]

    for lang in languages:
        txt_folder = os.path.join(base_path, lang, "txt")
        phoneme_folder = os.path.join(base_path, lang, "phonemes")
        os.makedirs(phoneme_folder, exist_ok=True)

        if os.path.exists(txt_folder):
            txt_files = sorted([f for f in os.listdir(txt_folder) if f.endswith(".txt")])
            language_code = lang.split("_")[0].lower()

            args_list = [
                (os.path.join(txt_folder, txt_file),
                 os.path.join(phoneme_folder, txt_file),
                 language_code)
                for txt_file in txt_files
            ]

            # Dynamically adjust workers based on memory
            available_memory = get_available_memory()
            memory_usage = psutil.virtual_memory().percent
            max_workers = max(1, min(os.cpu_count() - 1, int(len(args_list) / 10)))

            if memory_usage > max_memory_threshold:
                max_workers = max(1, max_workers // 2)
                logger.warning(f"High memory usage detected. Reducing workers to {max_workers}")

            successful_files = 0
            failed_files = 0

            with ProcessPoolExecutor(max_workers=max_workers) as executor:
                futures = [executor.submit(process_file, args) for args in args_list]

                for future in tqdm(as_completed(futures), 
                                 total=len(futures), 
                                 desc=f"Processing {lang}"):
                    try:
                        success = future.result()
                        if success:
                            successful_files += 1
                        else:
                            failed_files += 1
                    except Exception:
                        failed_files += 1

            logger.info(f"Language {lang} processing summary:")
            logger.info(f"Total files: {len(args_list)}")
            logger.info(f"Successful files: {successful_files}")
            logger.info(f"Failed files: {failed_files}")

def main():
    """Main execution function."""
    try:
        dataset_path = "dataset"
        if not os.path.exists(dataset_path):
            logger.error(f"Dataset path does not exist: {dataset_path}")
            return

        logger.info("Starting transcript phonemization...")
        phonemize_transcripts(dataset_path)
        logger.info("Phonemization completed successfully!")

    except Exception as e:
        logger.error(f"Unexpected error in main execution: {traceback.format_exc()}")

if __name__ == "__main__":
    main()