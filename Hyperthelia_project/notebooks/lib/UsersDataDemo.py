from pathlib import Path
from google.colab import files, drive
import shutil

def upload_multiple_tiffs(base_project_dir):
    """
    Prompts user to upload one or more TIFF files.
    Saves them in raw_data/user_upload/.
    Returns: Path to raw_data directory.
    """
    border = "=" * 60
    print(f"\n{border}")
    print("STEP 1: Upload your TIFF files")
    print(border)
    print("Use Ctrl (or Cmd) or Shift to select multiple files.\n")

    uploaded = files.upload()

    experiment_name = "user_upload"
    raw_data_dir = base_project_dir / "raw_data"
    exp_dir = raw_data_dir / experiment_name
    exp_dir.mkdir(parents=True, exist_ok=True)

    print(f"\nSaving files to: {exp_dir.resolve()}\n")
    for filename in uploaded:
        src = Path(filename)
        dst = exp_dir / src.name
        shutil.move(str(src), dst)
        print(f"- {dst.name}")

    print(f"\n{border}")
    print("Upload complete. Ready to run segmentation.")
    print(border)

    return raw_data_dir


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

