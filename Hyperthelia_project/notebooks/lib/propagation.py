# propagation.py

import pickle
from pathlib import Path
import numpy as np
import pandas as pd
from tifffile import imread, imwrite
from tracking import get_segmented_tiffs_by_experiment

# === LABEL PROPAGATION ===
def propagate_labels(tracked_results, label_stack, output_dir):
    output_dir = Path(output_dir)
    full_mask_dir = output_dir / "full_masks"
    centroid_mask_dir = output_dir / "centroid_labels"
    centroid_csv_dir = output_dir / "centroids"

    for d in [full_mask_dir, centroid_mask_dir, centroid_csv_dir]:
        d.mkdir(parents=True, exist_ok=True)

    centroid_records = []

    for t, (label_img, frame_tracks) in enumerate(zip(label_stack, tracked_results)):
        tracked_img = np.zeros_like(label_img, dtype=np.uint16)
        centroid_img = np.zeros_like(label_img, dtype=np.uint16)

        for track_id, centroid in frame_tracks:
            zc, yc, xc = np.round(centroid).astype(int)

            if not (
                0 <= zc < label_img.shape[0] and
                0 <= yc < label_img.shape[1] and
                0 <= xc < label_img.shape[2]
            ):
                continue

            orig_label = label_img[zc, yc, xc]
            if orig_label != 0:
                tracked_img[label_img == orig_label] = track_id
                centroid_img[zc, yc, xc] = track_id

                centroid_records.append({
                    "time": t,
                    "track_id": track_id,
                    "z": centroid[0],
                    "y": centroid[1],
                    "x": centroid[2]
                })

        imwrite(full_mask_dir / f"propagated_t{t:03d}.tif", tracked_img)
        imwrite(centroid_mask_dir / f"centroid_t{t:03d}.tif", centroid_img)

    df = pd.DataFrame(centroid_records)
    df.to_csv(centroid_csv_dir / "centroids.csv", index=False)

    print("\u2705 Propagation complete.")
    print(f"\u2192 Full masks:       {full_mask_dir}")
    print(f"\u2192 Centroid masks:   {centroid_mask_dir}")
    print(f"\u2192 Centroid CSV:     {centroid_csv_dir / 'centroids.csv'}")

# === PIPELINE ===
def run_propagation_pipeline(output_base_dir):
    segmented_groups = get_segmented_tiffs_by_experiment(output_base_dir)
    for exp_name, tiffs in segmented_groups.items():
        tracking_file = output_base_dir / f"outputs_{exp_name}" / "tracking" / "tracked_objects.pkl"
        if not tracking_file.exists():
            print(f" Skipping {exp_name}: No tracking file found.")
            continue

        print(f"\n Propagating labels for experiment: {exp_name}")
        with open(tracking_file, 'rb') as f:
            tracked = pickle.load(f)

        label_stack = [imread(f) for f in tiffs]
        tracking_dir = output_base_dir / f"outputs_{exp_name}" / "tracking"
        propagate_labels(tracked, label_stack, tracking_dir)
