import os
import re

# ---------------------------
# Provided Cleaning Function
# ---------------------------
def clean_phoneme_sequence(phoneme_text):
    """
    Clean the phoneme sequence by removing unwanted markers, stress, diacritics,
    and any non-phonemic symbols like punctuation.
    """
    # Remove any content within parentheses (e.g., language tags or extra markers)
    cleaned = re.sub(r'\([^)]*\)', '', phoneme_text)
    # Remove stress markers like ˈ, ˌ and other diacritical marks
    cleaned = re.sub(r'[ˈˌᵻʌːʰʳː]', '', cleaned)
    # Remove punctuation (e.g., commas, periods)
    cleaned = re.sub(r'[.,;:!?]', '', cleaned)
    # Remove extra whitespace and newlines
    cleaned = cleaned.strip()
    return cleaned

# ---------------------------
# Segmentation & Normalization Functions
# ---------------------------
def segment_by_language(phoneme_sequence):
    """
    Split the phoneme sequence into segments based on language tags.
    Returns a list of tuples: (language_tag, text).
    """
    segments = re.split(r'(\([^)]*\))', phoneme_sequence)
    combined_segments = []
    current_tag = None
    for seg in segments:
        seg = seg.strip()
        if not seg:
            continue
        if re.match(r'\([^)]*\)', seg):
            current_tag = seg
        else:
            combined_segments.append((current_tag, seg))
            current_tag = None
    return combined_segments

def normalize_segment(language_tag, segment_text):
    """
    Normalize a single segment:
      - Clean the segment text using clean_phoneme_sequence.
      - Split into tokens.
      - Prepend the language tag to every token.
    """
    cleaned_text = clean_phoneme_sequence(segment_text)
    tokens = cleaned_text.split()
    normalized_tokens = []
    for token in tokens:
        token = re.sub(r'\([^)]*\)', '', token)
        normalized_tokens.append(f"{language_tag}{token}")
    return " ".join(normalized_tokens)

def normalize_phoneme_sequence(phoneme_sequence, default_language_prefix):
    """
    Process the full phoneme sequence:
      - Segment based on language tags.
      - Normalize each segment individually.
      - Reassemble the segments.
    If a segment lacks an explicit language tag, use the default.
    """
    segments = segment_by_language(phoneme_sequence)
    normalized_segments = []
    for lang_tag, seg in segments:
        if not lang_tag:
            lang_tag = default_language_prefix
        norm_seg = normalize_segment(lang_tag, seg)
        normalized_segments.append(norm_seg)
    return " ".join(normalized_segments)

# # (Optional) Automated mapping suggestion (for review)
# # ---------------------------
# def suggest_mapping(unique_phonemes, fixed_inventory_tokens):
#     """
#     For each unique phoneme from the dataset (after removing language prefixes),
#     suggest a mapping based on similarity to tokens in the fixed inventory.
#     """
#     suggestions = {}
#     for phoneme in unique_phonemes:
#         fixed_tokens = [token.split(") ")[1] for token in fixed_inventory_tokens if ") " in token]
#         match = difflib.get_close_matches(phoneme, fixed_tokens, n=1)
#         if match:
#             suggestions[phoneme] = match[0]
#     return suggestions

# (You can use the above function to help build a mapping dictionary,
# but here we are relying on minimal normalization.)



# ---------------------------
# Processing All Files in the Dataset
# ---------------------------
def process_all_phoneme_files(root_dataset, default_language_prefixes):
    """
    Iterate over all speaker folders in the dataset (excluding 'metadata') and process
    every .txt file in each folder's 'phonemes' subfolder.
    The files are overwritten in place with normalized phoneme sequences.
    Additionally, for each speaker, exactly two normalized samples are printed to the terminal.
    """
    for folder in os.listdir(root_dataset):
        folder_path = os.path.join(root_dataset, folder)
        if not os.path.isdir(folder_path) or folder.lower() == "metadata":
            continue

        default_prefix = default_language_prefixes.get(folder, "(en-us)")
        phoneme_folder = os.path.join(folder_path, "phonemes")
        if not os.path.exists(phoneme_folder):
            continue

        sample_printed = 0  # Count of printed samples per speaker
        files = sorted([f for f in os.listdir(phoneme_folder) if f.endswith('.txt')])
        for file in files:
            file_path = os.path.join(phoneme_folder, file)
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            normalized_content = normalize_phoneme_sequence(content, default_prefix)
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(normalized_content)
            # Print only two samples per speaker
            if sample_printed < 2:
                print(f"Folder: {folder} | File: {file}")
                print(normalized_content)
                print("--------")
                sample_printed += 1

if __name__ == "__main__":
    root_dataset = "dataset"
    default_language_prefixes = {
        "English_F": "(en-us)",
        "English_M": "(en-us)",
        "Bhojpuri_F": "(hi)",
        "Bhojpuri_M": "(hi)",
        "Gujarathi_F": "(gu)",
        "Gujarathi_M": "(gu)",
        "Kannada_F": "(kn)",
        "Kannada_M": "(kn)",
    }
    process_all_phoneme_files(root_dataset, default_language_prefixes)
