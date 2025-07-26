from pathlib import Path
from google.colab import files, drive
import shutil

def choose_upload_method(base_project_dir):
    """
    Lets user choose between uploading TIFFs or using Google Drive.
    Returns the resolved raw_data directory for downstream use.
    """
    print("=" * 60)
    print("STEP 1: Choose how to load your TIFF data")
    print("=" * 60)
    print("1 - Upload one or more TIFFs directly")
    print("2 - Use a folder from your Google Drive\n")

    method = input("Enter 1 or 2: ").strip()

    if method == "1":
        return upload_multiple_tiffs(base_project_dir)

    elif method == "2":
        drive.mount("/content/drive")
        path = input("\nPaste the full path to your TIFF folder in Drive:\n(e.g. /content/drive/MyDrive/my_experiment)\n").strip()
        user_dir = Path(path)
        if not user_dir.exists():
            raise FileNotFoundError(f"No such folder: {user_dir}")
        print(f"\nUsing data from: {user_dir.resolve()}")
        return user_dir

    else:
        raise ValueError("Invalid option. Please enter 1 or 2.")

