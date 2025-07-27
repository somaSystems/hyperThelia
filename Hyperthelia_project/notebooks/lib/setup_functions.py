from pathlib import Path
from UsersDataDemo import choose_upload_method


def setup_hyperthelia_project(clone_dir: Path, raw_dir: Path = None, base_dir: Path = None):
    """
    Sets up the HyperThelia project by setting paths,
    optionally triggering the upload UI, and preparing output directories.

    Args:
        clone_dir (Path): Where the repo is located.
        raw_dir (Path or None): Path to raw TIFF data. If None, upload box is shown.
        base_dir (Path or None): Where to save outputs. Defaults to clone_dir/Hyperthelia_project.

    Returns:
        base_project_dir (Path): Path to project folder inside the repo
        raw_dir (Path): Final location of raw data
        outputs_dir (Path): Output directory
    """
    base_project_dir = base_dir if base_dir is not None else clone_dir / "Hyperthelia_project"

    # Choose or confirm raw_dir
    if raw_dir is None:
        raw_dir = choose_upload_method(base_project_dir)

    # Set up output directories
    raw_dir, outputs_dir = setup_project_io(base_project_dir, raw_dir=raw_dir)

    return base_project_dir, raw_dir, outputs_dir


def setup_project_io(base_project_dir: Path, raw_dir: Path):
    """
    Sets up output directory. raw_dir must be provided explicitly.
    """
    outputs_dir = base_project_dir / "outputs"
    outputs_dir.mkdir(exist_ok=True)

    if not raw_dir.exists():
        raise FileNotFoundError(f"Raw input folder not found: {raw_dir}")

    print(f"Project outputs will be saved to: {outputs_dir}")
    print(f"Looking for raw experiment folders in: {raw_dir}")
    return raw_dir, outputs_dir
