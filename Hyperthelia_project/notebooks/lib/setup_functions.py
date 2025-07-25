from pathlib import Path
import sys


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


def setup_project_io(base_project_dir, raw_dir=None):
    """
    Ensures output and raw input directories are set up correctly.
    If raw_dir is None, defaults to data_demo inside the base project.
    Returns: (raw_dir, outputs_dir)
    """
    outputs_dir = base_project_dir / "outputs"
    outputs_dir.mkdir(exist_ok=True)

    if raw_dir is None:
        raw_dir = base_project_dir / "data_demo"

    if not raw_dir.exists():
        raise FileNotFoundError(f"Raw input folder not found: {raw_dir}")

    print(f"Project outputs will be saved to: {outputs_dir}")
    print(f"Looking for raw experiment folders in: {raw_dir}")
    return raw_dir, outputs_dir

