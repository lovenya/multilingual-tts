import os
import tarfile

def extract_tar_files(base_path):
    """
    Extract all .tar files in the dataset directory and its subdirectories.

    :param base_path: Path to the dataset directory
    """
    for root, dirs, files in os.walk(base_path):
        for file in files:
            if file.endswith(".tar"):
                tar_file_path = os.path.join(root, file)
                extract_dir = os.path.splitext(tar_file_path)[0]  # Directory to extract to

                # Extract the .tar file
                with tarfile.open(tar_file_path, 'r') as tar:
                    tar.extractall(path=extract_dir)

                print(f"Extracted: {tar_file_path} -> {extract_dir}")
                os.remove(tar_file_path)  # Optionally, delete the .tar file

if __name__ == "__main__":
    dataset_path = "dataset"  # Replace with your dataset path
    extract_tar_files(dataset_path)
    print("All .tar files have been extracted.")
