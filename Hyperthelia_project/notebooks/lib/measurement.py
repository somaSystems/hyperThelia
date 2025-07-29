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

def run_all_measurements(
    experiment_data: dict,
    is_tracked: bool,
    compute_surface: bool = True,
    enable_intensity_measurement: bool = False,
    intensity_dir: Path = None,
    force: bool = False,
    measure_mode: str = "all"  # options: "2D", "3D", or "all"
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
            force=force
        )

        if measure_mode == "3D":
            df2D = pd.DataFrame()
        elif measure_mode == "2D":
            df3D = pd.DataFrame()
        elif measure_mode != "all":
            raise ValueError(f"Invalid measure_mode: {measure_mode}. Use '2D', '3D', or 'all'.")

        save_measurements(df3D, df2D, data["exp_path"], experiment_name, is_tracked)
