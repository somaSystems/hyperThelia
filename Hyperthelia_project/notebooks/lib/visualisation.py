# visualisation.py (updated with post-tracking tools)

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tifffile import imread
from pathlib import Path
from skimage.measure import regionprops
from skimage.segmentation import find_boundaries
from matplotlib import colormaps, cm, colors
from matplotlib.collections import LineCollection
from tracking import get_segmented_tiffs_by_experiment

# === VISUALISATION FUNCTIONS AVAILABLE ===
print(" visualisation.py loaded with functions:")
for name in dir():
    if callable(eval(name)) and not name.startswith("_") and name not in ['np', 'pd', 'plt', 'imread', 'Path', 'regionprops', 'find_boundaries', 'colormaps', 'cm', 'colors', 'LineCollection', 'get_segmented_tiffs_by_experiment']:
        print("  ", name)

# === PRE-TRACKING QC ===
def plot_volume_histogram_for_experiment(
    exp_index=0,
    experiments_dict=None,
    min_volume=500,
    max_volume=5000,
    bins=50
):
    if experiments_dict is None:
        print(" No experiment data provided.")
        return

    exp_names = list(experiments_dict.keys())
    if exp_index >= len(exp_names):
        print(" Invalid experiment index.")
        return

    exp_name = exp_names[exp_index]
    tiff_paths = experiments_dict[exp_name]
    print(f"\n Plotting volumes for experiment: {exp_name} ({len(tiff_paths)} TIFFs)")

    all_volumes = []
    for path in tiff_paths:
        label_img = imread(path)
        props = regionprops(label_img)
        all_volumes.extend([p.area for p in props])

    plt.figure(figsize=(8, 5))
    plt.hist(all_volumes, bins=bins, color='skyblue', edgecolor='black')
    plt.axvline(min_volume, color='red', linestyle='dotted', label=f"min_volume = {min_volume}")
    plt.axvline(max_volume, color='green', linestyle='dotted', label=f"max_volume = {max_volume}")
    plt.xlabel("Volume (voxels)")
    plt.ylabel("Object count")
    plt.title(f"3D Object Volume Histogram — {exp_name}")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

def view_segmentation_slice_with_boundaries(
    exp_index=0,
    experiments_dict=None,
    time_index=0,
    z_slice=18,
    edge_margin=1,
    cmap_name='viridis'
):
    if experiments_dict is None:
        print(" No experiment data provided.")
        return

    exp_names = list(experiments_dict.keys())
    if exp_index >= len(exp_names):
        print(" Invalid experiment index.")
        return

    exp_name = exp_names[exp_index]
    tiff_paths = experiments_dict[exp_name]
    print(f"\n🩻 Viewing experiment: {exp_name}, timepoint {time_index}, Z-slice {z_slice}")

    if time_index >= len(tiff_paths):
        print(" Invalid time index.")
        return

    label_stack = [imread(f) for f in tiff_paths]
    label_img = label_stack[time_index]
    if z_slice >= label_img.shape[0]:
        print(" Invalid Z-slice index.")
        return

    slice_labels = label_img[z_slice]
    shape_y, shape_x = slice_labels.shape
    boundary_ids = set()

    props = regionprops(label_img)
    for p in props:
        minz, miny, minx, maxz, maxy, maxx = (*p.bbox[:3], *p.bbox[3:])
        if (
            minz <= edge_margin or maxz >= label_img.shape[0] - edge_margin - 1 or
            miny <= edge_margin or maxy >= shape_y - edge_margin - 1 or
            minx <= edge_margin or maxx >= shape_x - edge_margin - 1
        ):
            boundary_ids.add(p.label)

    colored_img = np.zeros((*slice_labels.shape, 3), dtype=np.float32)
    label_ids = np.unique(slice_labels)
    label_ids = label_ids[label_ids != 0]

    if label_ids.size > 0:
        cmap = colormaps[cmap_name]
        for lbl in label_ids:
            mask = slice_labels == lbl
            color = cmap(lbl / label_ids.max())[:3]
            colored_img[mask] = color

    for lbl in boundary_ids:
        mask = (slice_labels == lbl)
        edges = find_boundaries(mask, mode='inner')
        colored_img[edges] = [1.0, 0.0, 0.0]

    plt.figure(figsize=(7, 7))
    plt.imshow(colored_img)
    plt.title(f"{exp_name} | Time {time_index} | Z {z_slice}")
    plt.axis('off')
    plt.tight_layout()
    plt.show()

# === POST-TRACKING QC ===
def plot_tracked_centroids_xy_by_index(exp_index=0, output_base_dir=None):
    experiments_dict = get_segmented_tiffs_by_experiment(output_base_dir)
    experiments = list(experiments_dict.keys())

    if len(experiments) == 0:
        print(" No experiments found.")
        return
    if exp_index < 0 or exp_index >= len(experiments):
        print(f" Invalid index {exp_index}. Must be between 0 and {len(experiments)-1}.")
        return

    exp_name = experiments[exp_index]
    print(f" Visualizing tracking for: {exp_name}")

    csv_path = output_base_dir / f"outputs_{exp_name}" / "tracking" / "centroids" / "centroids.csv"
    if not csv_path.exists():
        print(f" Centroids file not found: {csv_path}")
        return

    df = pd.read_csv(csv_path)
    if df.empty:
        print(f" No centroid data in {csv_path}")
        return

    fig, ax = plt.subplots(figsize=(8, 8))
    norm = colors.Normalize(vmin=df['z'].min(), vmax=df['z'].max())
    cmap = cm.get_cmap('viridis_r')

    for track_id, group in df.groupby('track_id'):
        group = group.sort_values('time')
        x, y, z = group['x'].values, group['y'].values, group['z'].values
        if len(x) < 2:
            continue

        points = np.array([x, y]).T.reshape(-1, 1, 2)
        segments = np.concatenate([points[:-1], points[1:]], axis=1)
        z_avg = (z[:-1] + z[1:]) / 2

        lc = LineCollection(segments, cmap=cmap, norm=norm)
        lc.set_array(z_avg)
        lc.set_linewidth(1.5)
        ax.add_collection(lc)
        ax.plot(x[0], y[0], 'o', color='green', markersize=5, label='Start' if track_id == 0 else "")
        ax.plot(x[-1], y[-1], 'o', color='black', markersize=5, label='End' if track_id == 0 else "")

    ax.invert_yaxis()
    sm = cm.ScalarMappable(norm=norm, cmap=cmap)
    sm.set_array([])
    plt.colorbar(sm, ax=ax, label='Z position (µm)')
    ax.set_xlabel("X (µm)")
    ax.set_ylabel("Y (µm)")
    ax.set_title(f"Tracked Centroids: {exp_name}")
    plt.axis('equal')
    plt.tight_layout()

    handles, labels = ax.get_legend_handles_labels()
    if handles:
        ax.legend(loc='upper right')

    plt.show()

def preview_propagated_labels_zslice(
    exp_index: int = 0,
    z_slice: int = 0,
    timepoints: list[int] | None = None,
    output_base_dir=None,
    cmap: str = "nipy_spectral",
):
    """
    Preview propagated label slices using ONLY lexicographic filename order.
    - Timepoint = index in the sorted list of files (0..N-1).
    - No parsing of numbers from filenames.
    - No assumed frame count; everything is derived from what's in full_masks/.
    - If `timepoints` is None, show all indices 0..N-1. If provided, out-of-range
      indices are ignored with a note (no 'Missing:' per-file spam).
    """
    from pathlib import Path
    import numpy as np
    import matplotlib.pyplot as plt
    import tifffile as tiff

    if output_base_dir is None:
        raise ValueError("output_base_dir is required")

    output_base_dir = Path(output_base_dir)

    # 1) Discover experiments (lexicographic by folder name)
    experiments = sorted([p for p in output_base_dir.glob("outputs_*") if p.is_dir()], key=lambda p: p.name)
    if not experiments:
        print(f"No experiment folders found under: {output_base_dir}")
        return

    if exp_index < 0 or exp_index >= len(experiments):
        raise IndexError(f"Experiment index {exp_index} is out of range (0..{len(experiments)-1}).")

    exp_dir = experiments[exp_index]
    exp_name = exp_dir.name.replace("outputs_", "")

    # 2) Locate propagated volumes
    full_mask_dir = exp_dir / "tracking" / "full_masks"
    if not full_mask_dir.exists():
        print(f"No full_masks directory found: {full_mask_dir}")
        return

    files = sorted(full_mask_dir.glob("propagated_*.tif"), key=lambda p: p.name)
    if not files:
        print(f"No propagated_*.tif files found in: {full_mask_dir}")
        return

    # 3) Decide which indices to show (purely by index in `files`)
    all_indices = list(range(len(files)))
    if timepoints is None:
        selected_indices = all_indices
    else:
        selected_indices = [i for i in timepoints if 0 <= i < len(files)]
        missing = [i for i in timepoints if i not in selected_indices]
        if missing:
            print(f"Note: ignoring out-of-range indices (0..{len(files)-1}): {sorted(set(missing))}")

    if not selected_indices:
        print("No matching frames to display after filtering.")
        return

    # 4) Load requested z-slices
    slices = []
    used_idx = []
    for idx in selected_indices:
        path = files[idx]
        vol = tiff.imread(str(path))
        if z_slice < 0 or z_slice >= vol.shape[0]:
            print(f"Z-slice {z_slice} out of bounds for {path.name} (depth {vol.shape[0]}). Skipping.")
            continue
        slices.append(vol[z_slice])
        used_idx.append(idx)

    if not slices:
        print("No valid slices loaded (all were out of bounds or unreadable).")
        return

    stack = np.stack(slices, axis=0)
    vmax = stack.max()

    # 5) Render
    fig_count = len(slices)
    fig, axes = plt.subplots(1, fig_count, figsize=(3 * fig_count, 4))
    if fig_count == 1:
        axes = [axes]

    for i, ax in enumerate(axes):
        ax.imshow(stack[i], cmap=cmap, interpolation="nearest", vmin=0, vmax=vmax)
        ax.set_title(f"t = {used_idx[i]}")  # strictly the lexicographic index
        ax.axis("off")

    plt.suptitle(f"Experiment: {exp_name} — Z = {z_slice}, Propagated Labels", fontsize=16)
    plt.tight_layout()
    plt.show()

