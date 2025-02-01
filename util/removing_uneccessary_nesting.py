import os
import shutil

def flatten_folders(base_path):
    """
    Remove unnecessary nesting in the folder structure.

    :param base_path: Path to the dataset directory
    """
    for root, dirs, files in os.walk(base_path):
        # Find folders with unnecessary nesting
        for dir_name in dirs:
            inner_folder = os.path.join(root, dir_name, dir_name)  # Nested folder
            if os.path.exists(inner_folder):
                print(f"Flattening: {inner_folder}")

                # Move all contents of the nested folder up one level
                for item in os.listdir(inner_folder):
                    source = os.path.join(inner_folder, item)
                    destination = os.path.join(root, dir_name, item)
                    shutil.move(source, destination)

                # Remove the now-empty nested folder
                shutil.rmtree(inner_folder)
                print(f"Removed: {inner_folder}")

if __name__ == "__main__":
    dataset_path = "dataset"  # Replace with your dataset path
    flatten_folders(dataset_path)
    print("Folder structure has been flattened.")
