# === NEW FILE === notebooks/lib/region_merge.py
from pathlib import Path
import pandas as pd

KEY3 = ["experiment", "label_id", "timepoint"]
KEY2 = ["experiment", "label_id", "timepoint", "Zslice"]

def _suffix_cols(df: pd.DataFrame, suffix: str, keys: list[str]) -> pd.DataFrame:
    keep = keys[:]
    vals = [c for c in df.columns if c not in set(keys + ["filename","source","is_tracked","region"])]
    return pd.concat([df[keep], df[vals].add_suffix(f"__{suffix}")], axis=1)

def merge_region_into_base(outputs_dir: Path, region: str):
    outputs_dir = Path(outputs_dir)
    for exp_dir in sorted([d for d in outputs_dir.glob("outputs_*") if d.is_dir()]):
        exp = exp_dir.name.replace("outputs_", "")
        mdir = exp_dir / "measured"
        if not mdir.exists():
            continue

        base3 = mdir / f"regionprops_{exp}_tracked_3D.csv"
        base2 = mdir / f"regionprops_{exp}_tracked_2D.csv"
        reg3  = mdir / f"regionprops_{exp}__{region}_tracked_3D.csv"
        reg2  = mdir / f"regionprops_{exp}__{region}_tracked_2D.csv"
        if base3.exists() and reg3.exists():
            b3 = pd.read_csv(base3)
            r3 = pd.read_csv(reg3)
            r3s = _suffix_cols(r3, region, KEY3)
            m3 = b3.merge(r3s, on=KEY3, how="left", validate="m:1")
            m3.to_csv(mdir / f"regionprops_{exp}_tracked_3D__with_{region}.csv", index=False)
            print(f"✅ merged 3D → {exp} ({region})")

        if base2.exists() and reg2.exists():
            b2 = pd.read_csv(base2)
            r2 = pd.read_csv(reg2)
            r2s = _suffix_cols(r2, region, KEY2)
            m2 = b2.merge(r2s, on=KEY2, how="left", validate="m:1")
            m2.to_csv(mdir / f"regionprops_{exp}_tracked_2D__with_{region}.csv", index=False)
            print(f"✅ merged 2D → {exp} ({region})")

