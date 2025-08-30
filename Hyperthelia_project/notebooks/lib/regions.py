# === NEW FILE === notebooks/lib/regions.py
from pathlib import Path
import numpy as np
from tifffile import imread, imwrite
from skimage.morphology import binary_erosion, disk

# ---- 2D, per-slice, label-preserving erosion ----
def _erode_labels_2d(volume: np.ndarray, pixels: int) -> np.ndarray:
    if pixels <= 0:
        return volume.copy()
    out = np.zeros_like(volume, dtype=volume.dtype)
    se1 = disk(1)
    Z = volume.shape[0]
    for z in range(Z):
        sl = volume[z]
        out_sl = np.zeros_like(sl, dtype=sl.dtype)
        labels = np.unique(sl)
        labels = labels[labels != 0]
        for lbl in labels:
            m = (sl == lbl)
            if not m.any():
                continue
            eroded = m.copy()
            for _ in range(pixels):  # apply erosion N times
                eroded = binary_erosion(eroded, footprint=se1)
            out_sl[eroded] = lbl
        out[z] = out_sl
    return out


# ---- create cytoplasm by erosion ----
def create_cytoplasm_masks(output_base_dir: Path, erosion_px: int = 4, region_name: str = "cytoplasm"):
    output_base_dir = Path(output_base_dir)
    for exp_dir in sorted([p for p in output_base_dir.glob("outputs_*") if p.is_dir()]):
        src = exp_dir / "tracking" / "full_masks"
        tifs = sorted(src.glob("propagated_*.tif"))
        if not tifs:
            continue
        out_dir = exp_dir / "tracking" / "regions" / region_name
        out_dir.mkdir(parents=True, exist_ok=True)
        for t_idx, path in enumerate(tifs):
            vol = imread(path)
            eroded = _erode_labels_2d(vol, erosion_px)
            imwrite(out_dir / f"propagated_t{t_idx:03d}.tif", eroded.astype(np.uint16))
        # provenance note
        (out_dir / "README.txt").write_text(f"region={region_name}\nderivation=erode_2d_{erosion_px}px\nsource=tracking/full_masks\n")

# ---- membrane = full − cytoplasm, label-preserving ----
def create_membrane_from_cytoplasm(output_base_dir: Path, cytoplasm_region: str = "cytoplasm", membrane_region: str = "membrane"):
    output_base_dir = Path(output_base_dir)
    for exp_dir in sorted([p for p in output_base_dir.glob("outputs_*") if p.is_dir()]):
        full_dir = exp_dir / "tracking" / "full_masks"
        cyto_dir = exp_dir / "tracking" / "regions" / cytoplasm_region
        if not (full_dir.exists() and cyto_dir.exists()):
            continue
        full_tifs = sorted(full_dir.glob("propagated_*.tif"))
        cyto_tifs = sorted(cyto_dir.glob("propagated_*.tif"))
        n = min(len(full_tifs), len(cyto_tifs))
        out_dir = exp_dir / "tracking" / "regions" / membrane_region
        out_dir.mkdir(parents=True, exist_ok=True)
        for i in range(n):
            full_v = imread(full_tifs[i])
            cyto_v = imread(cyto_tifs[i])
            out_v = full_v.copy()
            out_v[cyto_v > 0] = 0  # subtract, keep parent labels elsewhere
            imwrite(out_dir / f"propagated_t{i:03d}.tif", out_v.astype(np.uint16))
        (out_dir / "README.txt").write_text(f"region={membrane_region}\nderivation=full_minus_{cytoplasm_region}\nsource=tracking/full_masks + tracking/regions/{cytoplasm_region}\n")

# ---- convenience: do both in one go (creation only) ----
def create_cytoplasm_and_membrane(output_base_dir: Path, erosion_px: int = 4, cytoplasm_region: str = "cytoplasm", membrane_region: str = "membrane"):
    create_cytoplasm_masks(output_base_dir, erosion_px=erosion_px, region_name=cytoplasm_region)
    create_membrane_from_cytoplasm(output_base_dir, cytoplasm_region=cytoplasm_region, membrane_region=membrane_region)

