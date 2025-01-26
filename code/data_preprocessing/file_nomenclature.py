import os
import time
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor


def rename_files(base_path):
    start_time = time.time()

    # Get all the subfolders
    languages = [
        lang
        for lang in os.listdir(base_path)
        if os.path.isdir(os.path.join(base_path, lang))
    ]

    def get_language_and_gender(folder_name):
        parts = folder_name.split("_")
        return parts[0][:2], parts[1][:1]  # Assume all folder names are valid

    def process_language(lang):
        lang_path = os.path.join(base_path, lang)
        txt_folder = os.path.join(lang_path, "txt")
        wav_folder = os.path.join(lang_path, "wav")

        if os.path.exists(txt_folder) and os.path.exists(wav_folder):
            txt_files = sorted(os.listdir(txt_folder))  # Sorting for consistency
            language, gender = get_language_and_gender(lang)
            unique_counter = 1

            for txt_file in tqdm(txt_files, desc=f"Processing {lang}", leave=False):
                base_name, ext = os.path.splitext(txt_file)
                new_name = f"{language}_{gender}_{unique_counter:05d}"
                unique_counter += 1

                # Rename txt file
                old_txt_path = os.path.join(txt_folder, txt_file)
                new_txt_path = os.path.join(txt_folder, new_name + ext)
                os.rename(old_txt_path, new_txt_path)

                # Rename corresponding wav file if it exists
                old_wav_path = os.path.join(wav_folder, base_name + ".wav")
                if os.path.exists(old_wav_path):
                    new_wav_path = os.path.join(wav_folder, new_name + ".wav")
                    os.rename(old_wav_path, new_wav_path)

    with ThreadPoolExecutor() as executor:
        list(
            tqdm(
                executor.map(process_language, languages),
                total=len(languages),
                desc="Processing Languages",
                leave=True,
            )
        )

    elapsed_time = time.time() - start_time
    return elapsed_time


# Main execution
if __name__ == "__main__":
    # Set the path to the dataset directory
    dataset_path = "dataset"

    # Call the renaming function
    total_time = rename_files(dataset_path)

    # Print total time taken
    print(f"Total time taken: {total_time:.2f} seconds")
