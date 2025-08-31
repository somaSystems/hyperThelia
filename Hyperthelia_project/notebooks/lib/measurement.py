# measurement.py

import numpy as np
import pandas as pd
import tifffile
from skimage.measure import regionprops, marching_cubes
from pathlib import Path
from tqdm import tqdm
import warnings
import subprocess
from datetime import datetime

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

def measure_experiment(
    experiment_name: str,
    data: dict,
    is_tracked: bool,
    compute_surface: bool = True,
    intensity_dict: dict = None,
    force: bool = False,
    mode: str = "all"
) -> tuple[pd.DataFrame, pd.DataFrame]:
    results_3D = []
    results_2D = []
    tif_paths = data["tif_paths"]
    n_tiffs = len(tif_paths)

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

        if mode in ["3D", "all"]:
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

                volume_voxels = obj.area
                sphericity = (np.pi ** (1/3)) * ((6 * volume_voxels) ** (2/3)) / surface_area if surface_area > 0 else np.nan
                centroid_z, centroid_y, centroid_x = obj.centroid
                bbox = obj.bbox
                major_axis_length = obj.major_axis_length

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

                for ch, img in intensity_frames.items():
                    if img is not None:
                        intensities = img[mask == 1]
                        row[f'intensity_mean_{ch}'] = np.mean(intensities)
                        row[f'intensity_max_{ch}'] = np.max(intensities)
                        row[f'intensity_min_{ch}'] = np.min(intensities)
                        row[f'intensity_std_{ch}'] = np.std(intensities)

                results_3D.append(row)

        if mode in ["2D", "all"]:
            for z in range(volume.shape[0]):
                slice_mask = volume[z]
                props2D = regionprops(slice_mask)

                for obj in props2D:
                    # base fields (defensive reads)
                    label_id   = getattr(obj, "label", np.nan)
                    area_2D    = float(getattr(obj, "area", np.nan))
                    solidity_2D= float(getattr(obj, "solidity", np.nan))
                    cy, cx     = getattr(obj, "centroid", (np.nan, np.nan))
                    by0, bx0, by1, bx1 = getattr(obj, "bbox", (np.nan, np.nan, np.nan, np.nan))

                    # extra geometry (defensive reads)
                    eccentricity        = float(getattr(obj, "eccentricity", np.nan))
                    orientation         = float(getattr(obj, "orientation", np.nan))  # radians
                    perimeter           = float(getattr(obj, "perimeter", np.nan))
                    perimeter_crofton   = getattr(obj, "perimeter_crofton", np.nan)
                    perimeter_crofton   = float(perimeter_crofton) if perimeter_crofton is not np.nan else np.nan
                    major_axis_length   = float(getattr(obj, "major_axis_length", np.nan))
                    minor_axis_length   = float(getattr(obj, "minor_axis_length", np.nan))
                    convex_area         = float(getattr(obj, "convex_area", np.nan))
                    extent              = float(getattr(obj, "extent", np.nan))
                    filled_area         = float(getattr(obj, "filled_area", np.nan))

                    # derived metrics
                    equivalent_diameter = 2.0 * np.sqrt(area_2D / np.pi) if np.isfinite(area_2D) and area_2D > 0 else np.nan
                    _P = perimeter_crofton if np.isfinite(perimeter_crofton) else perimeter
                    circularity = (4.0 * np.pi * (area_2D / (_P * _P))) if (np.isfinite(area_2D) and np.isfinite(_P) and _P > 0) else np.nan
                    roundness   = ((4.0 * area_2D) / (np.pi * (major_axis_length ** 2))) if (np.isfinite(area_2D) and np.isfinite(major_axis_length) and major_axis_length > 0) else np.nan

                    row2D = {
                        'experiment': experiment_name,
                        'label_id': label_id,
                        'timepoint': t_idx,
                        'Zslice': z,
                        'filename': path.name,
                        'source': str(data["mask_dir"].relative_to(data["exp_path"])),
                        'is_tracked': is_tracked,

                        # original basics
                        'area_2D': area_2D,
                        'solidity_2D': solidity_2D,
                        'centroid_y': cy,
                        'centroid_x': cx,
                        'bbox_ymin': by0, 'bbox_xmin': bx0,
                        'bbox_ymax': by1, 'bbox_xmax': bx1,

                        # added geometry
                        'eccentricity': eccentricity,
                        'orientation': orientation,
                        'perimeter': perimeter,
                        'perimeter_crofton': _P if np.isfinite(perimeter_crofton) else np.nan,
                        'major_axis_length': major_axis_length,
                        'minor_axis_length': minor_axis_length,
                        'convex_area': convex_area,
                        'extent': extent,
                        'filled_area': filled_area,
                        'equivalent_diameter': equivalent_diameter,

                        # derived
                        'circularity': circularity,
                        'roundness': roundness,
                    }

                    # Hu moments (optional but useful)
                    try:
                        hu = getattr(obj, "moments_hu", None)
                        if hu is not None and len(hu) == 7:
                            for i in range(7):
                                row2D[f"hu_moment_{i+1}"] = float(hu[i])
                    except Exception:
                        pass

                    # intensity measures per channel (if provided)
                    if intensity_frames:
                        for ch, img in intensity_frames.items():
                            if img is None:
                                continue
                            # use obj.label directly for the boolean mask (most robust)
                            lbl = getattr(obj, "label", None)
                            if lbl is None:
                                continue
                            pix = img[z][slice_mask == lbl]
                            if pix.size > 0:
                                row2D[f"intensity_mean_{ch}"] = float(np.mean(pix))
                                row2D[f"intensity_max_{ch}"]  = float(np.max(pix))
                                row2D[f"intensity_min_{ch}"]  = float(np.min(pix))
                                row2D[f"intensity_std_{ch}"]  = float(np.std(pix))

                    results_2D.append(row2D)


    # === NEW === moved return out of the for-loop so ALL timepoints are processed
    return pd.DataFrame(results_3D), pd.DataFrame(results_2D)




def save_measurements(df3D: pd.DataFrame, df2D: pd.DataFrame, exp_path: Path, experiment_name: str, is_tracked: bool) -> None:
    measured_dir = exp_path / "measured"
    measured_dir.mkdir(exist_ok=True)

    if not df3D.empty:
        csv3D = measured_dir / f"regionprops_{experiment_name}_{'tracked' if is_tracked else 'untracked'}_3D.csv"
        df3D.to_csv(csv3D, index=False)
        print(f" Saved 3D: {csv3D}")

    if not df2D.empty:
        csv2D = measured_dir / f"regionprops_{experiment_name}_{'tracked' if is_tracked else 'untracked'}_2D.csv"
        df2D.to_csv(csv2D, index=False)
        print(f" Saved 2D: {csv2D}")

def run_all_measurements(
    experiment_data: dict,
    is_tracked: bool,
    compute_surface: bool = True,
    enable_intensity_measurement: bool = False,
    intensity_dir: Path = None,
    force: bool = False,
    measure_mode: str = "all"
):
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

        print(f"Found intensity channels: {list(intensity_dict.keys())}")

    for experiment_name, data in experiment_data.items():
        print(f"\nMeasuring experiment: {experiment_name}")
        df3D, df2D = measure_experiment(
            experiment_name,
            data,
            is_tracked,
            compute_surface=compute_surface,
            intensity_dict=intensity_dict,
            force=force,
            mode=measure_mode
        )
        save_measurements(df3D, df2D, data["exp_path"], experiment_name, is_tracked)
        
        # === NEW === discover base masks AND any region sets
from pathlib import Path

def discover_all_mask_sets(outputs_dir: Path, is_tracked: bool = True) -> dict:
    """
    Returns a single experiment_data dict covering:
      - Base tracked/untracked masks: tracking/full_masks or raw_segmented_tiffs  (region='full_cell')
      - Each derived region: tracking/regions/<region_name>  (region='<region_name>')

    Keys are unique for saving:
      '<exp>' for base, '<exp>__<region>' for regions.

    Each value mirrors discover_experiments(...) and adds:
      - 'region'  : 'full_cell' or the region folder name (e.g., 'cytoplasm', 'membrane')
      - 'exp_base': base experiment name without the region suffix (e.g., 'demothelia')
    """
    outputs_dir = Path(outputs_dir)
    out = {}
    for exp_path in sorted([p for p in outputs_dir.iterdir() if p.is_dir() and p.name.startswith("outputs_")]):
        exp = exp_path.name.replace("outputs_", "")

        # 1) Base (full cells)
        base_dir = exp_path / ("tracking/full_masks" if is_tracked else "raw_segmented_tiffs")
        base_tifs = sorted(base_dir.glob("*.tif"))
        if base_dir.exists() and base_tifs:
            out[exp] = {
                "mask_dir": base_dir,
                "tif_paths": base_tifs,
                "n_tiffs": len(base_tifs),
                "exp_path": exp_path,
                "region": "full_cell",
                "exp_base": exp,
            }

        # 2) Region sets
        regions_root = exp_path / "tracking" / "regions"
        if regions_root.exists():
            for rdir in sorted([d for d in regions_root.iterdir() if d.is_dir()]):
                rtifs = sorted(rdir.glob("*.tif"))
                if not rtifs:
                    continue
                rname = rdir.name
                key = f"{exp}__{rname}"  # used for file names
                out[key] = {
                    "mask_dir": rdir,
                    "tif_paths": rtifs,
                    "n_tiffs": len(rtifs),
                    "exp_path": exp_path,
                    "region": rname,
                    "exp_base": exp,
                }
    return out


# === NEW === one-call helper: measure base + all regions
def run_all_measurements_for_all_sets(
    outputs_dir: Path,
    is_tracked: bool,
    compute_surface: bool = True,
    enable_intensity_measurement: bool = False,
    intensity_dir: Path = None,
    force: bool = False,
    measure_mode: str = "all"
):
    """
    Convenience wrapper:
      1) discover_all_mask_sets(...) to gather base + regions
      2) summarise_experiment_data(...) for a quick printout
      3) run_all_measurements(...) once with the combined dict
    """
    all_sets = discover_all_mask_sets(outputs_dir, is_tracked=is_tracked)
    summarise_experiment_data(all_sets)
    return run_all_measurements(
        experiment_data=all_sets,
        is_tracked=is_tracked,
        compute_surface=compute_surface,
        enable_intensity_measurement=enable_intensity_measurement,
        intensity_dir=intensity_dir,
        force=force,
        measure_mode=measure_mode,
    )

