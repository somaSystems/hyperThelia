# tracking.py

import pickle
from pathlib import Path
from collections import defaultdict
import numpy as np
from tifffile import imread
from skimage.measure import regionprops
from scipy.spatial.distance import cdist
from scipy.optimize import linear_sum_assignment

# === DISCOVER SEGMENTED TIFFS ===
def get_segmented_tiffs_by_experiment(output_base_dir):
    experiment_groups = {}
    print(f" Looking for segmented TIFFs in: {output_base_dir}")

    for exp_dir in sorted(output_base_dir.glob("outputs_*")):
        experiment_name = exp_dir.name.replace("outputs_", "")
        tiff_dir = exp_dir / "raw_segmented_tiffs"

        print(f"\n Checking: {tiff_dir}")
        if not tiff_dir.exists():
            print(f" Skipping: {tiff_dir} (folder missing)")
            continue

        tiffs = sorted([f for f in tiff_dir.glob("*.tif") if f.is_file()])
        if not tiffs:
            print(f" No TIFFs found in {tiff_dir}")
            continue

        print(f" Found {len(tiffs)} TIFF(s) in {tiff_dir}")
        experiment_groups[experiment_name] = tiffs

        tracking_dir = exp_dir / "tracking"
        tracking_dir.mkdir(parents=True, exist_ok=True)
        log_path = tracking_dir / "timepoint_log.txt"
        with open(log_path, 'w') as log:
            for i, t in enumerate(tiffs):
                log.write(f"Timepoint {i+1}: {t.name}\n")
        print(f" Timepoint log saved to: {log_path}")

    if not experiment_groups:
        print(" No segmented TIFFs found in any experiment folders.")

    return experiment_groups

# === TRACKING FUNCTIONS ===
def get_3d_centroids(label_img, size_range=(10000, 150000), edge_margin=1):
    shape_z, shape_y, shape_x = label_img.shape
    props = regionprops(label_img)
    centroids = []
    for p in props:
        z, y, x = p.centroid
        minz, miny, minx, maxz, maxy, maxx = (*p.bbox[:3], *p.bbox[3:])
        volume = p.area
        if not (size_range[0] <= volume <= size_range[1]):
            continue
        if (
            minz <= edge_margin or maxz >= shape_z - edge_margin - 1 or
            miny <= edge_margin or maxy >= shape_y - edge_margin - 1 or
            minx <= edge_margin or maxx >= shape_x - edge_margin - 1
        ):
            continue
        centroids.append((z, y, x))
    return np.array(centroids)

def scale_centroids_physically(centroids, xy_um=0.325, z_um=1.0):
    scaled = centroids.astype(float).copy()
    if len(scaled) > 0:
        scaled[:, 0] *= z_um
        scaled[:, 1] *= xy_um
        scaled[:, 2] *= xy_um
    return scaled

def track_3d_centroids(label_stack,
                       size_range=(10000, 150000),
                       edge_margin=1,
                       xy_um=0.325,
                       z_um=1.0,
                       max_dist_um=30,
                       mode="nearest"):
    if mode != "nearest":
        raise NotImplementedError("Only 'nearest' mode is currently supported.")

    tracked = []
    id_counter = 1

    prev_centroids = get_3d_centroids(label_stack[0], size_range, edge_margin)
    prev_ids = list(range(id_counter, id_counter + len(prev_centroids)))
    id_counter += len(prev_centroids)
    tracked.append(list(zip(prev_ids, prev_centroids)))

    for t in range(1, len(label_stack)):
        curr_centroids = get_3d_centroids(label_stack[t], size_range, edge_margin)
        if len(prev_centroids) == 0 or len(curr_centroids) == 0:
            tracked.append([])
            prev_centroids = curr_centroids
            prev_ids = []
            continue
        scaled_prev = scale_centroids_physically(prev_centroids, xy_um, z_um)
        scaled_curr = scale_centroids_physically(curr_centroids, xy_um, z_um)
        cost = cdist(scaled_prev, scaled_curr)
        cost[cost > max_dist_um] = 1e6
        row_ind, col_ind = linear_sum_assignment(cost)
        matched_ids = {}
        for i, j in zip(row_ind, col_ind):
            if cost[i, j] < 1e6:
                matched_ids[j] = prev_ids[i]
        unmatched = [i for i in range(len(curr_centroids)) if i not in matched_ids]
        new_id_map = {i: id_counter + idx for idx, i in enumerate(unmatched)}
        id_counter += len(unmatched)
        curr_ids = []
        for i in range(len(curr_centroids)):
            id_val = matched_ids.get(i, new_id_map.get(i))
            curr_ids.append(id_val)
        tracked.append(list(zip(curr_ids, curr_centroids)))
        prev_centroids = curr_centroids
        prev_ids = curr_ids

    return tracked

# === SAVE AND LOAD ===
in_memory_tracking = {}

def save_tracking_results(exp_name, tracked, output_base_dir, save_pkl=True):
    in_memory_tracking[exp_name] = tracked
    if save_pkl:
        tracking_dir = output_base_dir / f"outputs_{exp_name}" / "tracking"
        tracking_dir.mkdir(parents=True, exist_ok=True)
        tracking_file = tracking_dir / "tracked_objects.pkl"
        with open(tracking_file, 'wb') as f:
            pickle.dump(tracked, f)
        print(f" Saved tracking results to: {tracking_file}")

def load_tracking_results(exp_name, output_base_dir):
    tracking_file = output_base_dir / f"outputs_{exp_name}" / "tracking" / "tracked_objects.pkl"
    if tracking_file.exists():
        with open(tracking_file, 'rb') as f:
            return pickle.load(f)
    elif exp_name in in_memory_tracking:
        print(f" Using in-memory tracking results for {exp_name}")
        return in_memory_tracking[exp_name]
    else:
        raise FileNotFoundError(f"Tracking results for '{exp_name}' not found in file or memory.")

# === PIPELINE ===
def run_tracking_pipeline(
    output_base_dir,
    xy_um=0.325,
    z_um=1.0,
    max_dist_um=30,
    min_volume=10000,
    max_volume=150000,
    edge_margin=1,
    tracking_mode="nearest",
    save_pickle=True
):
    segmented_groups = get_segmented_tiffs_by_experiment(output_base_dir)
    for exp_name, tiffs in segmented_groups.items():
        print(f"\n Tracking {len(tiffs)} TIFF(s) for experiment: {exp_name}")
        label_stack = [imread(f) for f in tiffs]

        tracked = track_3d_centroids(
            label_stack,
            size_range=(min_volume, max_volume),
            edge_margin=edge_margin,
            xy_um=xy_um,
            z_um=z_um,
            max_dist_um=max_dist_um,
            mode=tracking_mode
        )

        if save_pickle:
            save_tracking_results(exp_name, tracked, output_base_dir)

        print(f" {exp_name}: Tracked {sum(len(x) for x in tracked)} total objects across {len(tracked)} timepoints.")
