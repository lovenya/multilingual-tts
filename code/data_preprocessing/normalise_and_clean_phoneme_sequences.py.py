import os
import re

# ---------------------------
# Cleaning Function (provided/modified)
# ---------------------------
def clean_phoneme_sequence(phoneme_text):
    """
    Clean the phoneme sequence by removing unwanted markers, stress, diacritics,
    and any non-phonemic symbols like punctuation.
    (Assumes that it is applied to text that may contain embedded language tags.)
    """
    # Remove any content within parentheses (e.g., language tags or extra markers)
    cleaned = re.sub(r'\([^)]*\)', '', phoneme_text)
    # Remove stress markers like ˈ, ˌ, and other diacritical marks
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
    # Split the string while preserving language tags
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
    Normalize a single segment by:
      - Cleaning the segment text using clean_phoneme_sequence.
      - Splitting into tokens.
      - Prepending the language tag to every token.
    """
    cleaned_text = clean_phoneme_sequence(segment_text)
    tokens = cleaned_text.split()
    normalized_tokens = []
    for token in tokens:
        # Remove any stray embedded language tags from the token
        token = re.sub(r'\([^)]*\)', '', token)
        normalized_tokens.append(f"{language_tag} {token}")
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
    Iterate over all folders in the dataset (excluding 'metadata') and process
    all .txt files in their 'phonemes' subfolder. The original files are overwritten
    with the normalized phoneme sequences.
    
    Parameters:
      root_dataset: Path to the dataset folder.
      default_language_prefixes: A dictionary mapping folder names (or language keys)
                                 to their default language prefixes.
                                 For example:
                                   {"English_F": "(en-us)",
                                    "English_M": "(en-us)",
                                    "Bhojpuri_F": "(hi)",
                                    "Bhojpuri_M": "(hi)",
                                    "Gujarathi_F": "(gu)",
                                    "Gujarathi_M": "(gu)",
                                    "Kannada_F": "(kn)",
                                    "Kannada_M": "(kn)"}
    """
    for folder in os.listdir(root_dataset):
        folder_path = os.path.join(root_dataset, folder)
        if not os.path.isdir(folder_path) or folder.lower() == "metadata":
            continue

        # Determine default language prefix for this folder from the mapping dictionary.
        default_prefix = default_language_prefixes.get(folder, "(en-us)")  # fallback to en-us if not found
        
        phoneme_folder = os.path.join(folder_path, "phonemes")
        if not os.path.exists(phoneme_folder):
            continue
        
        # Process every .txt file in the phonemes subfolder
        for file in os.listdir(phoneme_folder):
            if file.endswith('.txt'):
                file_path = os.path.join(phoneme_folder, file)
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                normalized_content = normalize_phoneme_sequence(content, default_prefix)
                # Overwrite the original file with the normalized content
                with open(file_path, 'w', encoding='utf-8') as f:
                    f.write(normalized_content)

if __name__ == "__main__":
    # Root dataset folder (adjust if necessary)
    root_dataset = "dataset"
    # Mapping of folder names to default language prefixes
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
    # Process all phoneme sequence files across all 8 speaker folders
    process_all_phoneme_files(root_dataset, default_language_prefixes)
