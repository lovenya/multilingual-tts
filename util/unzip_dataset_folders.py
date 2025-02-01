import os
import gzip
import shutil

def extract_gz_files(base_path):
    """
    Extract all .gz files in the dataset directory and its subdirectories.

    :param base_path: Path to the dataset directory
    """
    for root, dirs, files in os.walk(base_path):
        for file in files:
            if file.endswith(".gz"):
                gz_file_path = os.path.join(root, file)
                extracted_file_path = os.path.splitext(gz_file_path)[0]  # Remove .gz extension

                # Extract the .gz file
                with gzip.open(gz_file_path, 'rb') as gz_file:
                    with open(extracted_file_path, 'wb') as extracted_file:
                        shutil.copyfileobj(gz_file, extracted_file)

                print(f"Extracted: {gz_file_path} -> {extracted_file_path}")
                os.remove(gz_file_path)  # Optionally, delete the .gz file

if __name__ == "__main__":
    dataset_path = "dataset"  # Replace with your dataset path
    extract_gz_files(dataset_path)
    print("All .gz files have been extracted.")
