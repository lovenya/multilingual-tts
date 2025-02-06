import os
import re

# Minimal normalization: we won't use an extensive manual mapping here.
# This code simply cleans the sequence (removes stress markers and punctuation)
# and ensures each token is prefixed with its language tag.
# It also handles code switching by segmenting the sequence based on language tags.

def segment_by_language(phoneme_sequence):
    """
    Splits the phoneme sequence into segments based on language tags.
    Returns a list of tuples: (language_tag, text).
    """
    # This regex splits the string and keeps language tags (e.g., "(en-us)", "(hi)")
    segments = re.split(r'(\([^)]*\))', phoneme_sequence)
    combined_segments = []
    current_tag = None
    for seg in segments:
        seg = seg.strip()
        if not seg:
            continue
        # If the segment is a language tag, save it.
        if re.match(r'\([^)]*\)', seg):
            current_tag = seg
        else:
            # If there's no tag preceding this text, we'll assign a default later.
            combined_segments.append((current_tag, seg))
            current_tag = None
    return combined_segments

def minimal_normalize_segment(language_tag, segment_text):
    """
    Normalize a single segment:
      - Remove stress markers and punctuation.
      - Remove any embedded language tags from tokens.
      - Prepend the given language tag to every token.
    """
    # Remove common stress markers and punctuation
    cleaned = re.sub(r'[ˈˌ.,;:!?]', '', segment_text)
    tokens = cleaned.split()
    normalized_tokens = []
    for token in tokens:
        # Remove any language tag embedded in the token (if any)
        token = re.sub(r'\([^)]*\)', '', token)
        normalized_tokens.append(f"{language_tag} {token}")
    return " ".join(normalized_tokens)

def minimal_normalize_phoneme_sequence(phoneme_sequence, default_language_prefix):
    """
    Process the full phoneme sequence by:
      - Segmenting it based on language tags.
      - Normalizing each segment individually.
      - Reassembling the segments.
    If a segment lacks a language tag, the default is used.
    """
    segments = segment_by_language(phoneme_sequence)
    normalized_segments = []
    for lang_tag, seg in segments:
        if not lang_tag:
            lang_tag = default_language_prefix
        norm_seg = minimal_normalize_segment(lang_tag, seg)
        normalized_segments.append(norm_seg)
    return " ".join(normalized_segments)

def process_sample_files(input_folder, output_folder, default_language_prefix, num_files=10):
    """
    Processes the first num_files in the input_folder (e.g., English_F/phonemes),
    normalizes the phoneme sequences (handling code switching),
    and writes the normalized outputs to output_folder.
    """
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    # Get a sorted list of .txt files and process only the first num_files.
    files = sorted([f for f in os.listdir(input_folder) if f.endswith('.txt')])[:num_files]
    for file in files:
        input_path = os.path.join(input_folder, file)
        output_path = os.path.join(output_folder, file)
        with open(input_path, 'r', encoding='utf-8') as f:
            content = f.read()
        normalized_content = minimal_normalize_phoneme_sequence(content, default_language_prefix)
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(normalized_content)
        print(f"Processed {file}:")
        print(normalized_content)
        print("--------")

if __name__ == "__main__":
    # Set the input folder for English_F phoneme sequences.
    input_folder = os.path.join("dataset", "English_F", "phonemes")
    # Set the output folder for sample normalized phoneme outputs.
    output_folder = "sample_normalised_phoneme_output"
    # Default language prefix for files in English_F.
    default_language_prefix = "(en-us)"
    # Process the first 10 files.
    process_sample_files(input_folder, output_folder, default_language_prefix, num_files=10)
