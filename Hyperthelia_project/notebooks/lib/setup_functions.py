from pathlib import Path
import sys

import sys
from pathlib import Path
from UsersDataDemo import choose_upload_method
from setup_functions import setup_project_io  # assumes this already exists

def setup_hyperthelia_project(clone_dir: Path, raw_dir: Path = None, base_dir: Path = None):
    """
    Sets up the HyperThelia project by cloning the repo (if needed), setting paths,
    optionally triggering the upload UI, and preparing output directories.

    Args:
        clone_dir (Path): Where to clone the repo.
        raw_dir (Path or None): Path to raw TIFF data. If None, upload box is shown.
        base_dir (Path or None): Where to save outputs. Defaults to clone_dir.

    Returns:
        base_project_dir (Path): Path to project folder inside the repo
        raw_dir (Path): Final location of raw data
        outputs_dir (Path): Output directory
    """
    REPO_URL = "https://github.com/somaSystems/HyperThelia.git"

    # Clone if needed
    if not clone_dir.exists():
        print(f"Cloning HyperThelia repo to {clone_dir}...")
        !git clone {REPO_URL} {clone_dir}
    else:
        print("HyperThelia repo already exists.")

    # Set base directory (inside clone or overridden)
    base_project_dir = base_dir if base_dir is not None else clone_dir / "Hyperthelia_project"

    # Add lib/ to path
    lib_dir = base_project_dir / "notebooks" / "lib"
    if str(lib_dir) not in sys.path:
        sys.path.insert(0, str(lib_dir))

    # Choose or confirm raw_dir
    if raw_dir is None:
        raw_dir = choose_upload_method(base_project_dir)

    # Set up output directories
    raw_dir, outputs_dir = setup_project_io(base_project_dir, raw_dir=raw_dir)

    return base_project_dir, raw_dir, outputs_dir

def clone_hyperthelia_repo(clone_parent_dir):
    REPO_NAME = "HyperThelia"
    PROJECT_SUBDIR = "Hyperthelia_project"
    REPO_URL = f"https://github.com/somaSystems/{REPO_NAME}.git"

    clone_dir = clone_parent_dir / REPO_NAME
    base_project_dir = clone_dir / PROJECT_SUBDIR

    if not clone_dir.exists():
        print("Cloning HyperThelia repo...")
        import subprocess
        subprocess.run(["git", "clone", REPO_URL, str(clone_dir)], check=True)
    else:
        print(f"Repo already exists at: {clone_dir}")

    # Add lib directory to sys.path
    lib_dir = base_project_dir / "notebooks" / "lib"
    if str(lib_dir) not in sys.path:
        sys.path.insert(0, str(lib_dir))
        print(f"Added to sys.path: {lib_dir}")

    print(f"BASE_PROJECT_DIR is set to: {base_project_dir}")
    return clone_dir, base_project_dir


def setup_project_io(base_project_dir, raw_dir):
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


