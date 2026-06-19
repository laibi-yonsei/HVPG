"""Skeletal muscle / muscle-density analysis for a single abdominal CT volume.

This reproduces the body-composition analysis described in the manuscript
(Gachon_DeepBody pipeline), applied to ONE CT volume together with its
segmentation mask.

Mask label convention (as provided by the user):

    0 = background
    1 = skeletal muscle              (HU  -25 .. 150, the whole muscle)
    2 = very low attenuation muscle  (VLAM, HU -30 ..   0)
    3 = low attenuation muscle       (LAM,  HU   0 ..  30)
    4 = high attenuation muscle      (HAM,  HU  30 .. 150)

Because labels 2/3/4 are HU-based sub-classes of label 1, a single integer
label per voxel cannot represent the overlap cleanly. To stay faithful to the
paper -- which defines *every* quantity by HU thresholds -- this script:

  1. Builds the "skeletal muscle region" from the mask (all muscle labels).
  2. Re-derives total muscle and the VLAM / LAM / HAM sub-classes directly from
     the CT Hounsfield-Unit values inside that region.
  3. Also reports the raw per-label voxel counts and mean HU, so you can verify
     what each label actually contains.

Metrics reported (whole 3D volume, all slices integrated):
  * Total skeletal muscle volume (cm^3) and mean muscle density (SMD, HU)
  * VLAM / LAM / HAM volumes and their proportions
  * Low-attenuation muscle proportion = vol(HU < 30) / vol(total muscle)
    -> the "muscle density proportion" used to define healthy / unhealthy muscle
  * Per-slice cross-sectional muscle area (cm^2), written to CSV
  * Optional single-slice L3 metrics (area, L3MI, sarcopenia) if --l3-slice given

Usage:
    python experiments/muscle_density_l3.py \
        --ct  /path/to/ct.nii.gz \
        --mask /path/to/mask.nii.gz \
        --sex M --height-m 1.70 \
        --out results/case01 \
        [--l3-slice 142] \
        [--muscle-labels 1 2 3 4]
"""

from __future__ import annotations

import argparse
import csv
import os
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np


# --------------------------------------------------------------------------- #
# HU thresholds (paper defaults; override via CLI if needed)
# --------------------------------------------------------------------------- #
# Total skeletal muscle range used for area/proportion definitions.
# The manuscript uses -29..150 for the proportion denominator; the Gachon
# software description uses -25..150. We default to -29..150 and expose a flag.
TOTAL_HU_DEFAULT = (-29.0, 150.0)
# Sub-class boundaries: VLAM [-30,0), LAM [0,30), HAM [30,150].
VLAM_HU_DEFAULT = (-30.0, 0.0)
LAM_HU_DEFAULT = (0.0, 30.0)
HAM_HU_DEFAULT = (30.0, 150.0)
# "Low attenuation muscle" = below this HU (within the total range).
LOW_ATTEN_CUTOFF_DEFAULT = 30.0

# Sarcopenia L3MI cut-offs (international consensus, cancer cachexia).
SARCOPENIA_L3MI = {"M": 55.0, "F": 39.0}


# --------------------------------------------------------------------------- #
# IO: read NIfTI with SimpleITK, fall back to nibabel.
# --------------------------------------------------------------------------- #
@dataclass
class Volume:
    array: np.ndarray            # shape (Z, Y, X)
    spacing_zyx: Tuple[float, float, float]  # mm, in (Z, Y, X) order

    @property
    def voxel_volume_cm3(self) -> float:
        sz, sy, sx = self.spacing_zyx
        return (sz * sy * sx) / 1000.0  # mm^3 -> cm^3

    @property
    def inplane_area_cm2(self) -> float:
        _, sy, sx = self.spacing_zyx
        return (sy * sx) / 100.0  # mm^2 -> cm^2


def _read_with_sitk(path: str) -> Volume:
    import SimpleITK as sitk

    img = sitk.ReadImage(path)
    arr = sitk.GetArrayFromImage(img)  # (Z, Y, X)
    # GetSpacing returns (X, Y, Z); reorder to (Z, Y, X) to match the array.
    sx, sy, sz = img.GetSpacing()
    return Volume(arr, (float(sz), float(sy), float(sx)))


def _read_with_nibabel(path: str) -> Volume:
    import nibabel as nib

    img = nib.load(path)
    arr = np.asanyarray(img.dataobj)  # (X, Y, Z)
    # Move to (Z, Y, X) to match the SimpleITK convention used here.
    arr = np.transpose(arr, (2, 1, 0))
    sx, sy, sz = img.header.get_zooms()[:3]
    return Volume(np.asarray(arr), (float(sz), float(sy), float(sx)))


def read_volume(path: str) -> Volume:
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    try:
        return _read_with_sitk(path)
    except ImportError:
        return _read_with_nibabel(path)


# --------------------------------------------------------------------------- #
# Analysis
# --------------------------------------------------------------------------- #
@dataclass
class Results:
    spacing_zyx: Tuple[float, float, float]
    voxel_volume_cm3: float
    inplane_area_cm2: float
    per_label_counts: Dict[int, int] = field(default_factory=dict)
    per_label_mean_hu: Dict[int, float] = field(default_factory=dict)

    total_muscle_voxels: int = 0
    total_muscle_volume_cm3: float = 0.0
    mean_muscle_hu: float = float("nan")  # skeletal muscle density (SMD)

    vlam_volume_cm3: float = 0.0
    lam_volume_cm3: float = 0.0
    ham_volume_cm3: float = 0.0
    low_atten_volume_cm3: float = 0.0
    low_atten_proportion: float = float("nan")  # vol(<cutoff) / vol(total)

    # Optional single-slice L3 metrics.
    l3_slice: Optional[int] = None
    l3_muscle_area_cm2: Optional[float] = None
    l3mi: Optional[float] = None
    sarcopenia: Optional[bool] = None

    per_slice_area_cm2: List[float] = field(default_factory=list)


def analyze(
    ct: Volume,
    mask: Volume,
    muscle_labels: Tuple[int, ...],
    total_hu: Tuple[float, float],
    vlam_hu: Tuple[float, float],
    lam_hu: Tuple[float, float],
    ham_hu: Tuple[float, float],
    low_cutoff: float,
    sex: Optional[str],
    height_m: Optional[float],
    l3_slice: Optional[int],
) -> Results:
    if ct.array.shape != mask.array.shape:
        raise ValueError(
            f"CT shape {ct.array.shape} != mask shape {mask.array.shape}. "
            "CT and mask must be co-registered / same grid."
        )

    hu = ct.array.astype(np.float32)
    lbl = mask.array.astype(np.int32)

    res = Results(
        spacing_zyx=ct.spacing_zyx,
        voxel_volume_cm3=ct.voxel_volume_cm3,
        inplane_area_cm2=ct.inplane_area_cm2,
    )

    # --- per-label diagnostics (verify what each label really is) ---------- #
    for v in sorted(int(x) for x in np.unique(lbl)):
        sel = lbl == v
        n = int(sel.sum())
        res.per_label_counts[v] = n
        res.per_label_mean_hu[v] = float(hu[sel].mean()) if n else float("nan")

    # --- skeletal muscle region from the mask ------------------------------ #
    muscle_region = np.isin(lbl, muscle_labels)

    # --- total skeletal muscle: gate the region by the total HU window ----- #
    lo, hi = total_hu
    total_mask = muscle_region & (hu >= lo) & (hu <= hi)
    res.total_muscle_voxels = int(total_mask.sum())
    res.total_muscle_volume_cm3 = res.total_muscle_voxels * ct.voxel_volume_cm3
    res.mean_muscle_hu = (
        float(hu[total_mask].mean()) if res.total_muscle_voxels else float("nan")
    )

    # --- HU-based sub-classes within the muscle region --------------------- #
    def _vol(lo_: float, hi_: float, include_hi: bool = False) -> float:
        if include_hi:
            m = muscle_region & (hu >= lo_) & (hu <= hi_)
        else:
            m = muscle_region & (hu >= lo_) & (hu < hi_)
        return int(m.sum()) * ct.voxel_volume_cm3

    res.vlam_volume_cm3 = _vol(*vlam_hu)
    res.lam_volume_cm3 = _vol(*lam_hu)
    res.ham_volume_cm3 = _vol(ham_hu[0], ham_hu[1], include_hi=True)

    low_mask = muscle_region & (hu >= lo) & (hu < low_cutoff)
    res.low_atten_volume_cm3 = int(low_mask.sum()) * ct.voxel_volume_cm3
    res.low_atten_proportion = (
        res.low_atten_volume_cm3 / res.total_muscle_volume_cm3
        if res.total_muscle_volume_cm3 > 0
        else float("nan")
    )

    # --- per-slice cross-sectional area (cm^2) ----------------------------- #
    per_slice = total_mask.reshape(total_mask.shape[0], -1).sum(axis=1)
    res.per_slice_area_cm2 = (per_slice * ct.inplane_area_cm2).astype(float).tolist()

    # --- optional single-slice L3 metrics ---------------------------------- #
    if l3_slice is not None:
        z = int(l3_slice)
        if not (0 <= z < total_mask.shape[0]):
            raise ValueError(f"--l3-slice {z} out of range [0,{total_mask.shape[0]})")
        res.l3_slice = z
        area = float(per_slice[z] * ct.inplane_area_cm2)
        res.l3_muscle_area_cm2 = area
        if height_m:
            res.l3mi = area / (height_m * height_m)
            if sex:
                cut = SARCOPENIA_L3MI.get(sex.upper())
                if cut is not None:
                    res.sarcopenia = res.l3mi < cut

    return res


# --------------------------------------------------------------------------- #
# Reporting
# --------------------------------------------------------------------------- #
def print_report(res: Results) -> None:
    p = print
    p("=" * 64)
    p("Skeletal muscle / muscle-density analysis (single CT volume)")
    p("=" * 64)
    sz, sy, sx = res.spacing_zyx
    p(f"Voxel spacing (Z,Y,X) mm : ({sz:.4f}, {sy:.4f}, {sx:.4f})")
    p(f"Voxel volume      cm^3   : {res.voxel_volume_cm3:.6f}")
    p(f"In-plane area     cm^2   : {res.inplane_area_cm2:.6f}")
    p("-" * 64)
    p("Per-label diagnostics (counts / mean HU):")
    for v in sorted(res.per_label_counts):
        p(f"  label {v}: {res.per_label_counts[v]:>10d} voxels | "
          f"mean HU = {res.per_label_mean_hu[v]:8.2f}")
    p("-" * 64)
    p(f"Total skeletal muscle voxels : {res.total_muscle_voxels}")
    p(f"Total skeletal muscle volume : {res.total_muscle_volume_cm3:.2f} cm^3")
    p(f"Mean muscle density (SMD)    : {res.mean_muscle_hu:.2f} HU")
    p("-" * 64)
    p(f"VLAM volume : {res.vlam_volume_cm3:8.2f} cm^3")
    p(f"LAM  volume : {res.lam_volume_cm3:8.2f} cm^3")
    p(f"HAM  volume : {res.ham_volume_cm3:8.2f} cm^3")
    p(f"Low-attenuation muscle volume     : {res.low_atten_volume_cm3:.2f} cm^3")
    p(f"Low-attenuation muscle proportion : {res.low_atten_proportion:.4f} "
      f"({100 * res.low_atten_proportion:.1f} %)")
    if res.l3_slice is not None:
        p("-" * 64)
        p(f"L3 slice index            : {res.l3_slice}")
        p(f"L3 muscle area    cm^2    : {res.l3_muscle_area_cm2:.2f}")
        if res.l3mi is not None:
            p(f"L3MI            cm^2/m^2  : {res.l3mi:.2f}")
        if res.sarcopenia is not None:
            p(f"Sarcopenia (L3MI cut-off) : {res.sarcopenia}")
    p("=" * 64)


def save_outputs(res: Results, out_dir: str) -> None:
    os.makedirs(out_dir, exist_ok=True)

    # Summary CSV.
    summary_path = os.path.join(out_dir, "summary.csv")
    with open(summary_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["metric", "value", "unit"])
        w.writerow(["total_muscle_volume", f"{res.total_muscle_volume_cm3:.4f}", "cm^3"])
        w.writerow(["mean_muscle_HU_SMD", f"{res.mean_muscle_hu:.4f}", "HU"])
        w.writerow(["vlam_volume", f"{res.vlam_volume_cm3:.4f}", "cm^3"])
        w.writerow(["lam_volume", f"{res.lam_volume_cm3:.4f}", "cm^3"])
        w.writerow(["ham_volume", f"{res.ham_volume_cm3:.4f}", "cm^3"])
        w.writerow(["low_atten_volume", f"{res.low_atten_volume_cm3:.4f}", "cm^3"])
        w.writerow(["low_atten_proportion", f"{res.low_atten_proportion:.6f}", "ratio"])
        if res.l3_slice is not None:
            w.writerow(["l3_slice", res.l3_slice, "index"])
            w.writerow(["l3_muscle_area", f"{res.l3_muscle_area_cm2:.4f}", "cm^2"])
            if res.l3mi is not None:
                w.writerow(["l3mi", f"{res.l3mi:.4f}", "cm^2/m^2"])
            if res.sarcopenia is not None:
                w.writerow(["sarcopenia", res.sarcopenia, "bool"])

    # Per-slice area CSV.
    slice_path = os.path.join(out_dir, "per_slice_area.csv")
    with open(slice_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["slice_index", "muscle_area_cm2"])
        for i, a in enumerate(res.per_slice_area_cm2):
            w.writerow([i, f"{a:.4f}"])

    print(f"[saved] {summary_path}")
    print(f"[saved] {slice_path}")


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ct", required=True, help="CT volume (.nii.gz), HU values")
    ap.add_argument("--mask", required=True, help="segmentation mask (.nii.gz)")
    ap.add_argument("--out", default="results/muscle_density",
                    help="output directory for CSVs")
    ap.add_argument("--muscle-labels", type=int, nargs="+", default=[1, 2, 3, 4],
                    help="mask labels that constitute skeletal muscle")
    ap.add_argument("--sex", choices=["M", "F", "m", "f"], default=None)
    ap.add_argument("--height-m", type=float, default=None,
                    help="patient height in meters (for L3MI)")
    ap.add_argument("--l3-slice", type=int, default=None,
                    help="slice index of the L3 level (for single-slice L3MI)")
    # HU overrides.
    ap.add_argument("--total-hu", type=float, nargs=2, default=list(TOTAL_HU_DEFAULT))
    ap.add_argument("--vlam-hu", type=float, nargs=2, default=list(VLAM_HU_DEFAULT))
    ap.add_argument("--lam-hu", type=float, nargs=2, default=list(LAM_HU_DEFAULT))
    ap.add_argument("--ham-hu", type=float, nargs=2, default=list(HAM_HU_DEFAULT))
    ap.add_argument("--low-cutoff", type=float, default=LOW_ATTEN_CUTOFF_DEFAULT)
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    ct = read_volume(args.ct)
    mask = read_volume(args.mask)

    res = analyze(
        ct=ct,
        mask=mask,
        muscle_labels=tuple(args.muscle_labels),
        total_hu=tuple(args.total_hu),
        vlam_hu=tuple(args.vlam_hu),
        lam_hu=tuple(args.lam_hu),
        ham_hu=tuple(args.ham_hu),
        low_cutoff=args.low_cutoff,
        sex=args.sex,
        height_m=args.height_m,
        l3_slice=args.l3_slice,
    )

    print_report(res)
    save_outputs(res, args.out)


if __name__ == "__main__":
    main()
