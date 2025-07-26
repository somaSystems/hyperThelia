from pathlib import Path
from google.colab import files
import shutil

def upload_multiple_tiffs(base_project_dir):
    """
    Prompts user to upload one or more TIFF files.
    Saves them in raw_data/user_upload/.
    Returns: Path to raw_data directory.
    """
    print("Select one or more TIFF files to upload (e.g. time series).")
    print("Use Ctrl or Shift to select multiple.")

    uploaded = files.upload()

    experiment_name = "user_upload"
    raw_data_dir = base_project_dir / "raw_data"
    exp_dir = raw_data_dir / experiment_name
    exp_dir.mkdir(parents=True, exist_ok=True)

    for filename in uploaded:
        src = Path(filename)
        dst = exp_dir / src.name
        shutil.move(str(src), dst)
        print(f"Saved: {dst.name}")

    print(f"\nFiles saved in: {exp_dir}")
    return raw_data_dir

