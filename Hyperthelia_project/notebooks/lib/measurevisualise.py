# File: notebooks/lib/measurevisualise.py
# Purpose: View and export tracked segmentation TIFFs colored by measurement values (2D or 3D).
# Supports both interactive (widget-based) and scripted viewing and exporting.

import numpy as np
import pandas as pd
import tifffile
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.colors import Normalize, ListedColormap
from skimage.segmentation import find_boundaries
from pathlib import Path

import ipywidgets as widgets
from IPython.display import display, clear_output
from tifffile import imread
from typing import Union, List

# ===  LIST AVAILABLE MEASUREMENT CSVs ===
def list_available_measurement_csvs(base_dir, return_first=True, use_dropdown=False):
    matches = list(base_dir.rglob("outputs_*/measured/regionprops_*_tracked_*.csv"))
    matches = sorted(matches)
    if not matches:
        raise FileNotFoundError("No tracked measurement CSVs found.")

    if use_dropdown:
        options = [str(p) for p in matches]
        dropdown = widgets.Dropdown(options=options, description='Select CSV:')
        display(dropdown)
        def get_path():
            return Path(dropdown.value)
        return get_path

    if return_first:
        return matches[0]

    return matches


# ===  LOAD CSV-COUPLED DATA ===
def get_image_paths_from_csv_path(csv_path, base_dir):
    name = csv_path.stem.replace("regionprops_", "")
    for suffix in ["_tracked_2D", "_tracked_3D", "_tracked"]:
        if name.endswith(suffix):
            name = name[: -len(suffix)]
    experiment_key = name

    full_masks_dir = base_dir / f"outputs_{experiment_key}" / "tracking" / "full_masks"
    tif_paths = sorted(full_masks_dir.glob("propagated_t*.tif"))
    if not tif_paths:
        raise FileNotFoundError(
            f"No TIFFs found for experiment '{experiment_key}' in {full_masks_dir}"
        )
    return experiment_key, full_masks_dir, tif_paths


# ===  VIEW BY CSV ===
def view_by_csv(csv_path: Path, base_dir: Path, timepoint: int, z: int, value_column: str):
    experiment_key, image_dir, tif_paths = get_image_paths_from_csv_path(csv_path, base_dir)
    image_path = tif_paths[timepoint]
    df = pd.read_csv(csv_path)
    df_tp = df[df["timepoint"] == timepoint] if "timepoint" in df.columns else df.copy()

    print(f"\n Viewing experiment: {experiment_key}")
    print(f"   Timepoint: {timepoint}, Z-Slice: {z}")
    print(f"   Value to color by: {value_column}")
    print(f"   Image path: {image_path}")
    print(f"   CSV path: {csv_path}")

    volume = tifffile.imread(image_path)
    if z < 0 or z >= volume.shape[0]:
        print(f"⚠️ Z index {z} is out of bounds. Image has {volume.shape[0]} slices.")
        return

    labels = volume[z]
    label_map = dict(zip(df_tp["label_id"], df_tp[value_column]))

    colored_img = np.zeros((*labels.shape, 3), dtype=float)
    unique_labels = np.unique(labels)
    unique_labels = unique_labels[unique_labels != 0]

    values = [label_map.get(lbl, None) for lbl in unique_labels]

    if not values or all(v is None for v in values):
        print(f"⚠️ No values found to display for Z-slice {z}. Showing empty background.")
        fig, ax = plt.subplots(figsize=(6, 6))
        ax.imshow(np.zeros_like(labels), cmap="gray")
        ax.set_title(f"TP {timepoint}, Z {z} — no labels")
        ax.axis('off')
        plt.show()
        return

    if all(isinstance(v, (int, float, np.number)) for v in values if v is not None):
        vals = np.array([label_map.get(lbl, 0) for lbl in labels.flat]).reshape(labels.shape)
        vmin, vmax = np.nanmin(values), np.nanmax(values)
        norm = Normalize(vmin=vmin, vmax=vmax)
        colormap = cm.get_cmap("viridis")
        for lbl in unique_labels:
            val = label_map.get(lbl, vmin)
            color = colormap(norm(val))[:3]
            colored_img[labels == lbl] = color
        colorbar = cm.ScalarMappable(norm=norm, cmap=colormap)
    else:
        unique_vals = sorted(set(v for v in values if v is not None))
        cmap = plt.get_cmap("tab20", len(unique_vals))
        val_to_color = {v: cmap(i)[:3] for i, v in enumerate(unique_vals)}
        for lbl in unique_labels:
            val = label_map.get(lbl, None)
            color = val_to_color.get(val, (0, 0, 0))
            colored_img[labels == lbl] = color
        colorbar = None

    boundaries = find_boundaries(labels, mode='outer')

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.imshow(colored_img)
    ax.contour(boundaries, levels=[0.5], colors='white', linewidths=0.5)
    ax.set_title(f"TP {timepoint}, Z {z} — colored by {value_column}")
    ax.axis('off')
    if colorbar:
        fig.colorbar(colorbar, ax=ax, label=value_column)
    plt.show()


# === INTERACTIVE VIEWER ===
def interactive_measurement_viewer(
    output_base_dir: Path,
    csv_path: Path = None,
    timepoint: int = None,
    z: int = None,
    value_column: str = None
):
    if all(v is not None for v in [csv_path, timepoint, z, value_column]):
        if not isinstance(csv_path, Path):
            csv_path = Path(csv_path)
        if not csv_path.is_absolute():
            csv_path = output_base_dir / csv_path
        print(" Scripted mode: displaying specified measurement view")
        view_by_csv(
            csv_path=csv_path,
            base_dir=output_base_dir,
            timepoint=timepoint,
            z=z,
            value_column=value_column
        )
        return

    print(" Interactive mode: use dropdowns to explore measurements")
    csv_paths = list_available_measurement_csvs(output_base_dir, return_first=False)
    if not csv_paths:
        print("❌ No tracked measurement CSVs found.")
        return

    csv_dropdown = widgets.Dropdown(
        options=[str(p.relative_to(output_base_dir)) for p in csv_paths],
        description="CSV File:"
    )
    timepoint_selector = widgets.IntSlider(description="TimePoint", min=0, max=0)
    z_selector = widgets.IntSlider(description="Z-slice", min=0, max=0)
    measure_dropdown = widgets.Dropdown(description="Measurement:", disabled=True)
    output_box = widgets.Output()

    def update_fields(*args):
        with output_box:
            clear_output(wait=True)
            try:
                csv_path_val = output_base_dir / csv_dropdown.value
                df = pd.read_csv(csv_path_val)
                cols = df.columns.tolist()

                if "timepoint" in cols:
                    timepoints = sorted(df["timepoint"].unique())
                    timepoint_selector.max = max(timepoints)
                    timepoint_selector.disabled = False
                else:
                    timepoint_selector.max = 0
                    timepoint_selector.value = 0
                    timepoint_selector.disabled = False

                # Detect number of Z-slices from corresponding TIFF
                experiment_key, _, tif_paths = get_image_paths_from_csv_path(csv_path_val, output_base_dir)
                sample_stack = tifffile.imread(tif_paths[0])
                z_selector.disabled = False
                z_selector.max = sample_stack.shape[0] - 1
                z_selector.value = 0

                exclude_cols = {
                    "label_id", "timepoint", "Zslice", "CellName", "experiment",
                    "filename", "Series", "image_path", "track_id"
                }
                value_measures = [c for c in df.columns if c not in exclude_cols]
                measure_dropdown.options = value_measures
                measure_dropdown.disabled = False

            except Exception as e:
                print(f"⚠️ Failed to read CSV or extract valid measurement columns: {e}")

    def update_plot(*args):
        with output_box:
            clear_output(wait=True)
            try:
                selected_csv = output_base_dir / csv_dropdown.value
                view_by_csv(
                    csv_path=selected_csv,
                    base_dir=output_base_dir,
                    timepoint=timepoint_selector.value,
                    z=z_selector.value,
                    value_column=measure_dropdown.value
                )
            except Exception as e:
                print(f"⚠️ Failed to display image: {e}")

    csv_dropdown.observe(update_fields, names="value")
    timepoint_selector.observe(update_plot, names="value")
    z_selector.observe(update_plot, names="value")
    measure_dropdown.observe(update_plot, names="value")

    control_row = widgets.HBox([csv_dropdown])
    control_row2 = widgets.HBox([timepoint_selector, z_selector, measure_dropdown])
    display(widgets.VBox([control_row, control_row2, output_box]))
    update_fields()
    csv_dropdown.value = csv_dropdown.options[0]


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

    if output_dir is None:
        output_dir = base_dir / "exports" / f"exports_{experiment_key}"

    image_path = tif_paths[timepoint]
    df = pd.read_csv(csv_path)
    df_tp = df[df["timepoint"] == timepoint]

    volume = tifffile.imread(image_path)

    if mode == "2d":
        if z is None:
            raise ValueError("Z index must be provided for 2D export.")

        df_tp = df_tp[df_tp["Zslice"] == z]
        label_map = dict(zip(df_tp["label_id"], df_tp[value_column]))

        def map_slice(slice_labels):
            out = np.zeros_like(slice_labels, dtype=np.float32)
            for lbl in np.unique(slice_labels):
                if lbl == 0:
                    continue
                out[slice_labels == lbl] = label_map.get(lbl, 0)
            return out

        result = map_slice(volume[z])

    else:
        label_map = dict(zip(df_tp["label_id"], df_tp[value_column]))

        def map_slice(slice_labels):
            out = np.zeros_like(slice_labels, dtype=np.float32)
            for lbl in np.unique(slice_labels):
                if lbl == 0:
                    continue
                out[slice_labels == lbl] = label_map.get(lbl, 0)
            return out

        result = np.stack([map_slice(volume[z_idx]) for z_idx in range(volume.shape[0])])

    output_dir.mkdir(parents=True, exist_ok=True)
    output_name = f"{experiment_key}_tp{timepoint}_{value_column}.tif"
    out_path = output_dir / output_name

    tifffile.imwrite(out_path, result)
    print(f" Saved measurement TIFF: {out_path}")
    
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
