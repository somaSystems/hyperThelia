# measurement.py

import numpy as np
import pandas as pd
import tifffile
from skimage.measure import regionprops, marching_cubes
from pathlib import Path
from tqdm import tqdm
import warnings

def discover_experiments(outputs_dir: Path, is_tracked: bool = True) -> dict:
    experiment_folders = sorted([p for p in outputs_dir.iterdir() if p.is_dir() and p.name.startswith("outputs_")])
    experiment_data = {}

    for exp_path in experiment_folders:
        experiment_name = exp_path.name.replace("outputs_", "")
        mask_dir = exp_path / ("tracking/full_masks" if is_tracked else "raw_segmented_tiffs")
        tif_files = sorted(mask_dir.glob("*.tif"))

        if mask_dir.exists() and tif_files:
            experiment_data[experiment_name] = {
                "mask_dir": mask_dir,
                "tif_paths": tif_files,
                "n_tiffs": len(tif_files),
                "exp_path": exp_path,
            }

    return experiment_data

def summarise_experiment_data(experiment_data: dict) -> pd.DataFrame:
    summary_df = pd.DataFrame([
        {"experiment": name, "n_tiffs": data["n_tiffs"]}
        for name, data in experiment_data.items()
    ]).sort_values("experiment").reset_index(drop=True)

    print(f"\n Summary of discovered experiments")
    print(f" Total experiments: {len(summary_df)}")
    print(f" Total TIFFs: {summary_df['n_tiffs'].sum()}\n")

    return summary_df


def measure_experiment(experiment_name: str, data: dict, is_tracked: bool, compute_surface: bool = True, intensity_dict: dict = None, force: bool = False) -> pd.DataFrame:
    results = []
    tif_paths = data["tif_paths"]
    n_tiffs = len(tif_paths)

    # Setup intensity per-channel tiff paths
    if intensity_dict:
        for channel, paths in intensity_dict.items():
            if len(paths) != n_tiffs:
                msg = f"Channel '{channel}' in experiment '{experiment_name}' has {len(paths)} intensity TIFFs but {n_tiffs} segmentation TIFFs."
                if not force:
                    raise ValueError(f" TIFF count mismatch. {msg}")
                else:
                    warnings.warn(f" {msg} Using only first {min(len(paths), n_tiffs)} frames.")

    for t_idx, path in enumerate(tqdm(tif_paths, desc=f"   Timepoints for {experiment_name}")):
        if intensity_dict:
            intensity_frames = {
                ch: tifffile.imread(paths[t_idx]) if t_idx < len(paths) else None
                for ch, paths in intensity_dict.items()
            }
        else:
            intensity_frames = {}

        volume = tifffile.imread(path)
        labels = np.unique(volume)
        labels = labels[labels != 0]

        for label_id in tqdm(labels, leave=False, desc=f"   TP {t_idx}", mininterval=1.0, miniters=20):
            mask = (volume == label_id).astype(np.uint8)
            props = regionprops(mask)

            if not props:
                continue

            obj = props[0]

            try:
                eigvals = np.sort(obj.inertia_tensor_eigvals)
                elongation = np.sqrt(1 - eigvals[0] / eigvals[2]) if eigvals[2] > 0 else np.nan
            except:
                eigvals = [np.nan, np.nan, np.nan]
                elongation = np.nan

            try:
                minor = obj.minor_axis_length
                aspect_ratio = obj.major_axis_length / minor if minor > 0 else np.nan
            except:
                minor = np.nan
                aspect_ratio = np.nan

            if compute_surface:
                try:
                    verts, faces, _, _ = marching_cubes(mask, level=0)
                    surface_area = sum(
                        np.sqrt(max(0, s * (s - a) * (s - b) * (s - c)))
                        for tri in faces
                        for p0, p1, p2 in [(verts[tri[0]], verts[tri[1]], verts[tri[2]])]
                        for a, b, c in [(np.linalg.norm(p1 - p0), np.linalg.norm(p2 - p1), np.linalg.norm(p0 - p2))]
                        for s in [(a + b + c) / 2]
                    )
                except:
                    surface_area = np.nan
            else:
                surface_area = np.nan

            volume_voxels = obj.area if hasattr(obj, 'area') else np.nan
            sphericity = (np.pi ** (1/3)) * ((6 * volume_voxels) ** (2/3)) / surface_area if surface_area > 0 else np.nan
            centroid_z, centroid_y, centroid_x = obj.centroid if hasattr(obj, 'centroid') else (np.nan, np.nan, np.nan)
            bbox = obj.bbox if hasattr(obj, 'bbox') else [np.nan]*6
            major_axis_length = obj.major_axis_length if hasattr(obj, 'major_axis_length') else np.nan

            row = {
                'experiment': experiment_name,
                'label_id': label_id,
                'timepoint': t_idx,
                'filename': path.name,
                'source': str(data["mask_dir"].relative_to(data["exp_path"])),
                'is_tracked': is_tracked,
                'area_voxels': volume_voxels,
                'centroid_z': centroid_z,
                'centroid_y': centroid_y,
                'centroid_x': centroid_x,
                'bbox_zmin': bbox[0], 'bbox_ymin': bbox[1], 'bbox_xmin': bbox[2],
                'bbox_zmax': bbox[3], 'bbox_ymax': bbox[4], 'bbox_xmax': bbox[5],
                'major_axis_length': major_axis_length,
                'minor_axis_length': minor,
                'eigval1': eigvals[0], 'eigval2': eigvals[1], 'eigval3': eigvals[2],
                'elongation': elongation,
                'aspect_ratio': aspect_ratio,
                'surface_area': surface_area,
                'sphericity': sphericity,
                'valid_geometry': not any(np.isnan([centroid_z, centroid_y, centroid_x, volume_voxels, minor, aspect_ratio, *eigvals, elongation, sphericity, surface_area]))
            }

            # Optional: add intensity stats
            for ch, img in intensity_frames.items():
                if img is not None:
                    intensities = img[mask == 1]
                    row[f'intensity_mean_{ch}'] = np.mean(intensities)
                    row[f'intensity_max_{ch}'] = np.max(intensities)
                    row[f'intensity_min_{ch}'] = np.min(intensities)
                    row[f'intensity_std_{ch}'] = np.std(intensities)

            results.append(row)

    return pd.DataFrame(results)

def save_measurements(df: pd.DataFrame, exp_path: Path, experiment_name: str, is_tracked: bool) -> Path:
    measured_dir = exp_path / "measured"
    measured_dir.mkdir(exist_ok=True)
    csv_path = measured_dir / f"regionprops_{experiment_name}_{'tracked' if is_tracked else 'untracked'}.csv"
    df.to_csv(csv_path, index=False)
    print(f" Saved: {csv_path}")
    return csv_path


def run_all_measurements(
    experiment_data: dict,
    is_tracked: bool,
    compute_surface: bool = True,
    enable_intensity_measurement: bool = False,
    intensity_dir: Path = None,
    force: bool = False
):
    """
    Loop over all experiments, optionally loading intensity TIFFs per channel from folders.
    """
    intensity_dict = None
    if enable_intensity_measurement:
        if intensity_dir is None:
            raise ValueError("enable_intensity_measurement=True but intensity_dir was not provided.")

        intensity_dict = {}
        for channel_folder in sorted(intensity_dir.glob("*/")):
            channel_name = channel_folder.name
            tiff_paths = sorted(channel_folder.glob("*.tif"))
            if not tiff_paths:
                print(f" Skipping empty channel: {channel_name}")
                continue
            intensity_dict[channel_name] = tiff_paths

        print(f" Found intensity channels: {list(intensity_dict.keys())}")

    for experiment_name, data in experiment_data.items():
        print(f"\n Measuring experiment: {experiment_name}")
        df = measure_experiment(
            experiment_name,
            data,
            is_tracked,
            compute_surface=compute_surface,
            intensity_dict=intensity_dict,
            force=force
        )
        save_measurements(df, data["exp_path"], experiment_name, is_tracked)
