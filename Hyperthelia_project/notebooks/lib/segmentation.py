from pathlib import Path
from tifffile import imread, imwrite
import numpy as np
import pickle
from cellpose import models, core, io


def setup_cellpose_model(gpu=True):
    """
    Sets up Cellpose model with SAM backend and returns it.
    Also enables logging and checks GPU availability.
    """
    io.logger_setup()

    if gpu and not core.use_gpu():
        raise ImportError("❌ No GPU access. Change your Colab runtime to GPU.")

    print("✅ Cellpose model initialised (SAM backend)")
    return models.CellposeModel(gpu=gpu)

def get_tiff_groups_by_experiment(raw_dir):
    """
    Finds all experiment folders in `raw_dir` and collects raw TIFFs in each.
    Returns: dict {experiment_name: list of TIFF Paths}
    """
    print(f"\n Scanning raw data directory: {raw_dir.resolve()}")
    experiment_dirs = [d for d in raw_dir.iterdir() if d.is_dir()]
    tiff_groups = {}

    for exp_dir in experiment_dirs:
        #  Match all raw TIFFs
        tiffs = [f for f in exp_dir.iterdir()
                 if f.suffix.lower() in [".tif", ".tiff"]]
        if tiffs:
            tiff_groups[exp_dir.name] = sorted(tiffs)
            print(f"🧪 Found {len(tiffs)} TIFF(s) in {exp_dir.name}:")
            for f in tiffs:
                print(f"    {f.name}")
        else:
            print(f" No TIFFs found in {exp_dir.name}")

    return tiff_groups

def get_pickle_groups_by_experiment(raw_dir):
    """
    Finds all experiment folders in `raw_dir` and collects segmented pickle files in each.
    Returns: dict {experiment_name: list of PKL Paths}
    """
    print(f"\n Scanning raw data directory for pickles: {raw_dir.resolve()}")
    experiment_dirs = [d for d in raw_dir.iterdir() if d.is_dir()]
    pickle_groups = {}

    for exp_dir in experiment_dirs:
        pickles = sorted(exp_dir.glob("segmented_*.pkl"))
        if pickles:
            pickle_groups[exp_dir.name] = pickles
            print(f"🧪 Found {len(pickles)} PKL(s) in {exp_dir.name}:")
            for f in pickles:
                print(f"    {f.name}")
        else:
            print(f" No PKLs found in {exp_dir.name}")

    return pickle_groups

def segment_single_tiff(path, model, z_axis, channel_axis, batch_size, do_3D, stitch_threshold):
    """
    Segments a single TIFF and returns the stitched mask.
    All segmentation parameters must be passed explicitly.
    """
    print(f" Segmenting: {path.name}")
    img_3D = imread(path)

    masks_stitched, _, _ = model.eval(
        img_3D,
        z_axis=z_axis,
        channel_axis=channel_axis,
        batch_size=batch_size,
        do_3D=do_3D,
        stitch_threshold=stitch_threshold,
    )
    return masks_stitched

def save_segmentation_pickle(path, mask, output_base_dir):
    experiment_name = path.parent.name
    output_dir = output_base_dir / f"outputs_{experiment_name}" / "raw_segmented_pickles"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_filename = f"segmented_{path.stem}.pkl"
    with open(output_dir / output_filename, 'wb') as f:
        pickle.dump(mask, f)
    print(f" Saved: {output_filename} to {output_dir}")

def save_segmentation_tiff(path, mask, output_base_dir):
    """
    Saves the segmentation mask as a TIFF file to the appropriate output directory.
    """
    experiment_name = path.parent.name
    output_dir = output_base_dir / f"outputs_{experiment_name}" / "raw_segmented_tiffs"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_filename = f"segmented_{path.stem}.tif"
    imwrite(output_dir / output_filename, mask.astype(np.uint16))  #  ensure uint16 for label masks
    print(f" Saved TIFF: {output_filename} to {output_dir}")

def segment_and_save(tiff_paths, output_base_dir, model, z_axis, channel_axis, batch_size, do_3D, stitch_threshold):
    print(f" Will now segment {len(tiff_paths)} TIFF file(s)...")
    for path in tiff_paths:
        mask = segment_single_tiff(
            path, model,
            z_axis=z_axis,
            channel_axis=channel_axis,
            batch_size=batch_size,
            do_3D=do_3D,
            stitch_threshold=stitch_threshold,
        )
        save_segmentation_pickle(path, mask, output_base_dir)
        save_segmentation_tiff(path, mask, output_base_dir)

def print_experiment_summary(raw_dir):
    """
    Prints a summary of discovered folders and TIFF counts in raw_dir,
    including image shapes and a warning if shapes are inconsistent.
    """
    tiff_groups = get_tiff_groups_by_experiment(raw_dir)
    print(f" Found {len(tiff_groups)} experiment folder(s) to process.")
    for i, (exp_name, tiff_list) in enumerate(tiff_groups.items(), start=1):
        print(f"  {i}. {exp_name}: {len(tiff_list)} TIFF(s)")
        shapes = []
        for tiff in tiff_list:
            try:
                img = imread(tiff)
                shapes.append(img.shape)
            except Exception as e:
                print(f"     Error reading {tiff.name}: {e}")
        unique_shapes = list(set(shapes))
        if len(unique_shapes) > 1:
            print(f"     WARNING: Inconsistent image shapes: {unique_shapes}")
        elif unique_shapes:
            print(f"     Shape: {unique_shapes[0]}")

def run_segmentation_pipeline(raw_dir, output_dir, model, z_axis, channel_axis, batch_size, do_3D, stitch_threshold):
    tiff_groups = get_tiff_groups_by_experiment(raw_dir)
    for exp_name, tiff_list in tiff_groups.items():
        print(f"\n Starting segmentation for experiment: {exp_name} ({len(tiff_list)} TIFFs)")
        segment_and_save(
            tiff_list, output_dir, model,
            z_axis=z_axis,
            channel_axis=channel_axis,
            batch_size=batch_size,
            do_3D=do_3D,
            stitch_threshold=stitch_threshold,
        )
