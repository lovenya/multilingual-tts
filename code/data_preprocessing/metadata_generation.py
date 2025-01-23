import os
import random
import csv


def generate_metadata(base_path, output_dir):
    """
    Generate metadata for a dataset and split it into train, validation, and test sets.
    """
    metadata = []

    # Traverse all subfolders in the dataset
    languages = [
        lang
        for lang in os.listdir(base_path)
        if os.path.isdir(os.path.join(base_path, lang))
    ]

    for lang in languages:
        wav_folder = os.path.join(base_path, lang, "wav")
        txt_folder = os.path.join(base_path, lang, "txt")

        if os.path.exists(wav_folder) and os.path.exists(txt_folder):
            wav_files = sorted(
                [f for f in os.listdir(wav_folder) if f.endswith(".wav")]
            )

            for wav_file in wav_files:
                base_name, _ = os.path.splitext(wav_file)
                txt_file = os.path.join(txt_folder, base_name + ".txt")
                wav_path = os.path.join(wav_folder, wav_file)

                # Extract language and speaker ID from the folder name
                lang_code, gender = lang.split("_")
                speaker_id = f"spk_{lang_code}_{gender}"

                # Read transcript from the corresponding .txt file
                with open(txt_file, "r", encoding="utf-8") as f:
                    transcript = f.read().strip()

                # Append metadata row
                metadata.append(
                    {
                        "audio_filepath": wav_path,
                        "transcript": transcript,
                        "language": lang_code,
                        "speaker_id": speaker_id,
                    }
                )

    # Shuffle the metadata randomly
    random.shuffle(metadata)

    # Split into train, validation, and test sets (80-10-10)
    total = len(metadata)
    train_split = int(0.8 * total)
    val_split = int(0.9 * total)

    train_data = metadata[:train_split]
    val_data = metadata[train_split:val_split]
    test_data = metadata[val_split:]

    # Write each set to a CSV file
    os.makedirs(output_dir, exist_ok=True)
    write_to_csv(train_data, os.path.join(output_dir, "train.csv"))
    write_to_csv(val_data, os.path.join(output_dir, "validation.csv"))
    write_to_csv(test_data, os.path.join(output_dir, "test.csv"))

    print(
        f"Metadata generated:\n  Train: {len(train_data)}\n  Validation: {len(val_data)}\n  Test: {len(test_data)}"
    )


def write_to_csv(data, output_file):
    """
    Write metadata rows to a CSV file with UTF-8 encoding for proper Unicode support.
    """
    with open(output_file, "w", newline="", encoding="utf-8-sig") as csvfile:
        writer = csv.DictWriter(
            csvfile,
            fieldnames=["audio_filepath", "transcript", "language", "speaker_id"],
        )
        writer.writeheader()
        writer.writerows(data)


if __name__ == "__main__":
    # Set paths
    dataset_path = "dataset"  # Replace with your dataset path
    output_metadata_dir = "dataset\metadata"  # Directory to save metadata CSVs

    # Generate metadata
    generate_metadata(dataset_path, output_metadata_dir)
