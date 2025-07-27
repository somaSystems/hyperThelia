# measurevisualise.py

import numpy as np
import pandas as pd
import tifffile
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.colors import Normalize
from skimage.segmentation import find_boundaries
from pathlib import Path

import ipywidgets as widgets
from IPython.display import display, clear_output
from tifffile import imread
from typing import Union, List



# ===  LIST AVAILABLE MEASUREMENT CSVs ===
def list_available_measurement_csvs(base_dir: Path, return_first: bool = False) -> Union[Path, List[Path]]:
    pattern = "outputs/outputs_*/measured/regionprops_*_tracked.csv"
    matches = sorted(base_dir.glob(pattern))

    if not matches:
        raise FileNotFoundError("No tracked measurement CSVs found.")

    if return_first:
        return matches[0]

    print(f"Found {len(matches)} measurement CSV(s).")
    return matches




# ===  LOAD CSV-COUPLED DATA ===
def get_image_paths_from_csv_path(csv_path, base_dir):
    """
    Infers experiment folder and corresponding TIFF masks from the CSV filename.
    Works with experiment names containing underscores.
    """
    from pathlib import Path

    name = csv_path.stem  # e.g. regionprops_user_upload_tracked
    if name.startswith("regionprops_") and name.endswith("_tracked"):
        experiment_key = name.replace("regionprops_", "").replace("_tracked", "")
    elif name.startswith("regionprops_"):
        experiment_key = name.replace("regionprops_", "")
    else:
        raise ValueError(f"Cannot infer experiment name from: {name}")

    image_dir = base_dir / f"outputs/outputs_{experiment_key}" / "tracking" / "full_masks"
    tif_paths = sorted(image_dir.glob("*.tif"))

    if not tif_paths:
        raise FileNotFoundError(f"No TIFFs found for experiment {experiment_key} in {image_dir}")

    return experiment_key, image_dir, tif_paths

# ===  VIEW BY CSV ===
def view_by_csv(csv_path: Path, base_dir: Path, timepoint: int, z: int, value_column: str):
    experiment_key, image_dir, tif_paths = get_image_paths_from_csv_path(csv_path, base_dir)
    image_path = tif_paths[timepoint]
    df = pd.read_csv(csv_path)
    df_tp = df[df["timepoint"] == timepoint]

    print(f"\n Viewing experiment: {experiment_key}")
    print(f"   Timepoint: {timepoint}, Z-Slice: {z}")
    print(f"   Value to color by: {value_column}")
    print(f"   Image path: {image_path}")
    print(f"   CSV path: {csv_path}")

    volume = tifffile.imread(image_path)
    labels = volume[z]
    label_map = dict(zip(df_tp["label_id"], df_tp[value_column]))

    vmin, vmax = np.nanmin(list(label_map.values())), np.nanmax(list(label_map.values()))
    norm = Normalize(vmin=vmin, vmax=vmax)
    colormap = cm.get_cmap("viridis")

    colored_img = np.zeros((*labels.shape, 3), dtype=float)
    for lbl in np.unique(labels):
        if lbl == 0:
            continue
        val = label_map.get(lbl, vmin)
        color = colormap(norm(val))[:3]
        colored_img[labels == lbl] = color

    boundaries = find_boundaries(labels, mode='outer')

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.imshow(colored_img)
    ax.contour(boundaries, levels=[0.5], colors='white', linewidths=0.5)
    ax.set_title(f"TP {timepoint}, Z {z} — colored by {value_column}")
    ax.axis('off')
    fig.colorbar(cm.ScalarMappable(norm=norm, cmap=colormap), ax=ax, label=value_column)
    plt.show()

# ===  EXPORT MEASURE-LABELED TIFF ===
def export_measurement_values_as_tiff(
    csv_path: Path,
    base_dir: Path,
    timepoint: int,
    value_column: str,
    output_dir: Path = None,
    mode: str = "3d",
    z: int = None
):
    experiment_key, _, tif_paths = get_image_paths_from_csv_path(csv_path, base_dir)
    
    # Set default export folder to exports/exports_<experiment>/
    if output_dir is None:
        output_dir = base_dir / "exports" / f"exports_{experiment_key}"

    image_path = tif_paths[timepoint]
    df = pd.read_csv(csv_path)
    df_tp = df[df["timepoint"] == timepoint]

    label_map = dict(zip(df_tp["label_id"], df_tp[value_column]))
    volume = tifffile.imread(image_path)

    def map_values(label_slice):
        out = np.zeros_like(label_slice, dtype=np.float32)
        for lbl in np.unique(label_slice):
            if lbl == 0:
                continue
            out[label_slice == lbl] = label_map.get(lbl, 0)
        return out

    if mode == "2d":
        if z is None:
            raise ValueError("Z index must be provided for 2D export.")
        result = map_values(volume[z])
    else:
        result = np.stack([map_values(volume[i]) for i in range(volume.shape[0])], axis=0)

    output_dir.mkdir(parents=True, exist_ok=True)
    output_name = f"{experiment_key}_tp{timepoint}_{value_column}.tif"
    out_path = output_dir / output_name

    tifffile.imwrite(out_path, result)
    print(f" Saved measurement TIFF: {out_path}")

# File: notebooks/lib/measurevisualise.py


def interactive_segmentation_viewer(output_base_dir):
    """Display segmented TIFFs interactively with slice selection."""
    experiments = sorted([d for d in output_base_dir.glob("outputs_*") if d.is_dir()])
    if not experiments:
        print("❌ No experiments found in:", output_base_dir)
        return

    tiff_groups = {
        exp.name: sorted((exp / "raw_segmented_tiffs").glob("*.tif"))
        for exp in experiments
    }

    exp_input = widgets.BoundedIntText(
        value=0, min=0, max=len(experiments) - 1, description='Experiment:'
    )
    tiff_input = widgets.BoundedIntText(value=0, min=0, max=0, description='Timepoint:')
    z_input = widgets.BoundedIntText(value=0, min=0, max=0, description='Z-slice:')
    output_box = widgets.Output()

    def update_plot(*args):
        with output_box:
            clear_output(wait=True)
            try:
                exp = experiments[exp_input.value]
                tiffs = tiff_groups[exp.name]
                tiff_input.max = max(len(tiffs) - 1, 0)
                selected_tiff = tiffs[tiff_input.value]
                img = imread(selected_tiff)

                if img.ndim != 3:
                    print("❌ Not a 3D image.")
                    return

                z_input.max = img.shape[0] - 1
                if z_input.value >= img.shape[0]:
                    z_input.value = img.shape[0] // 2

                plt.figure(figsize=(5, 5))
                plt.imshow(img[z_input.value], cmap='nipy_spectral')
                plt.title(f"{selected_tiff.name} — Z={z_input.value}")
                plt.axis("off")
                plt.show()

            except Exception as e:
                print(f"⚠️ Error: {e}")

    # React to user input
    exp_input.observe(update_plot, names='value')
    tiff_input.observe(update_plot, names='value')
    z_input.observe(update_plot, names='value')

    # Display widgets and first view
    display(widgets.HBox([exp_input, tiff_input, z_input]), output_box)
    update_plot()

