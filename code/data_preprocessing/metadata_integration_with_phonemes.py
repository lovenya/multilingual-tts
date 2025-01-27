# this integrates the CSVs in metadata/ folder with another column of phoneme sequences
# which it takes from the phonemes folders present in the folders of all the 8 speakers.
# the code is to be tested of course since i was running into RAM memory issues,
# and thus the phoneme sequences haven't ben generate for like >90% dataset. 

import os
import pandas as pd
from phonemizer import phonemize
from phonemizer.separator import Separator

def generate_phoneme_sequences(csv_path, language_code_map, output_path):
    """
    Adds a 'phoneme_sequence' column to the metadata CSV by generating phonemes for each transcript.

    Args:
        csv_path (str): Path to the input CSV file.
        language_code_map (dict): Mapping of speaker prefixes to phonemizer language codes (e.g., {'en': 'en-us'}).
        output_path (str): Path to save the updated CSV.
    """
    # Load the CSV
    df = pd.read_csv(csv_path)

    # Ensure the expected columns exist
    if 'transcript' not in df.columns or 'speaker_id' not in df.columns:
        raise ValueError("The CSV must contain 'transcript' and 'speaker_id' columns.")

    # Initialize phoneme_sequence column
    df['phoneme_sequence'] = ""

    for index, row in df.iterrows():
        transcript = row['transcript']
        speaker_id = row['speaker_id']

        # Determine language based on speaker ID prefix (e.g., en_f, gu_m)
        lang_prefix = speaker_id.split('_')[0]  # Extract language prefix
        language_code = language_code_map.get(lang_prefix)

        if language_code is None:
            raise ValueError(f"No language mapping found for speaker ID prefix: {lang_prefix}")

        # Generate phoneme sequence
        phoneme_sequence = phonemize(
            transcript,
            language=language_code,
            backend='espeak',
            separator=Separator(word="|", syllable=" ", phone=""),
        )

        # Store in the DataFrame
        df.at[index, 'phoneme_sequence'] = phoneme_sequence

    # Save updated CSV
    df.to_csv(output_path, index=False)

# Mapping speaker ID prefixes to phonemizer language codes
language_code_map = {
    'en': 'en-us',
    'gu': 'gu',
    'kn': 'kn',
    'bh': 'hi',  # Bhojpuri fallback to Hindi
}

# Paths to metadata CSVs
metadata_folder = "dataset/metadata"

for split in ['train', 'test', 'validation']:
    input_csv = os.path.join(metadata_folder, f"{split}.csv")
    output_csv = os.path.join(metadata_folder, f"{split}_updated.csv")

    # Generate metadata with phoneme sequences
    generate_phoneme_sequences(input_csv, language_code_map, output_csv)
