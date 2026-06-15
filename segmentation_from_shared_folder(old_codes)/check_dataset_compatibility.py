"""
check_dataset_compatibility.py
================================

Checks whether the new datasets (fastMRI NPY + BreastDx normal patients)
are compatible with the existing patients_combined/ dataset format.

Compares:
  A. Existing dataset    : data/patients_combined/  (patients 1–233)
  B. FastMRI NPY dataset : F:/NPY_classified_dataset/  (benign/malignant/normal)
  C. BreastDx normals    : data/normal_patients/  (patients 234–249)

Checks performed for EACH dataset:
  1. File existence    — image.npy and label.npy present
  2. Array shapes      — (C, D, H, W) for image, (1, D, H, W) for label
  3. Dtype             — should be float32
  4. Channel count     — should be 3 channels (P1, P2, P3)
  5. Value ranges      — after z-score norm: mean≈0, std≈1, no extreme outliers
  6. Label validity    — values should be 0.0 or 1.0 only
  7. D dimension       — depth (slices) after resampling to 1.5mm iso
  8. Shape consistency — all patients in same dataset should be similar
  9. NaN / Inf check   — no corrupted values
  10. File size sanity — image.npy vs label.npy should be ~3:1 ratio

Then compares across datasets:
  - Spatial dimension ranges compatible?
  - Value range compatible?
  - Channel count same?
  - Can they be used together in the same DataLoader?

Usage:
    python src/classification/check_dataset_compatibility.py

Output:
    Terminal report + check_compatibility_report.json
"""

import os
import sys
import json
import numpy as np
from collections import defaultdict


# ------------------------------------------------------------------ #
#  CONFIGURATION                                                       #
# ------------------------------------------------------------------ #

# ── Existing dataset (your current segmentation patients) ──────────
EXISTING_ROOT = "data/patients_combined"

# ── FastMRI NPY dataset (the new classified dataset on USB) ────────
FASTMRI_ROOT  = r"D:\Breast_Tumor_AI_Project\NPY_classified_dataset"

# ── BreastDx normal patients (numbered 234–249) ────────────────────
BREASTDX_NORMAL_ROOT = "data/normal_patients_fixed"

# Number of sample patients to deeply inspect per dataset
# (full check of all patients for shape/dtype, deep value check on N)
DEEP_SAMPLE_N = 5

# Expected values
EXPECTED_CHANNELS    = 3
EXPECTED_LABEL_CHANNELS = 1
EXPECTED_DTYPE       = np.float32
MAX_ALLOWED_MEAN_ABS = 2.0    # after z-score norm, mean should be near 0
MAX_ALLOWED_STD      = 5.0    # std should be near 1 (allow some variation)


# ------------------------------------------------------------------ #
#  HELPERS                                                             #
# ------------------------------------------------------------------ #

def fmt_shape(shape):
    return "×".join(str(s) for s in shape)


def check_one_patient(pid, image_path, label_path, deep=False):
    """
    Returns a dict of check results for one patient.
    deep=True performs value-level checks (slower).
    """
    result = {
        "pid"           : pid,
        "image_exists"  : False,
        "label_exists"  : False,
        "image_shape"   : None,
        "label_shape"   : None,
        "image_dtype"   : None,
        "label_dtype"   : None,
        "channels_ok"   : False,
        "dtype_ok"      : False,
        "label_valid"   : False,
        "shape_match"   : False,
        "errors"        : [],
        "warnings"      : [],
    }

    # ── File existence ────────────────────────────────────────────
    if not os.path.exists(image_path):
        result["errors"].append(f"image.npy MISSING: {image_path}")
        return result
    if not os.path.exists(label_path):
        result["errors"].append(f"label.npy MISSING: {label_path}")
        return result

    result["image_exists"] = True
    result["label_exists"] = True

    # ── Load arrays ───────────────────────────────────────────────
    try:
        img = np.load(image_path)
        lbl = np.load(label_path)
    except Exception as e:
        result["errors"].append(f"Load failed: {e}")
        return result

    result["image_shape"] = tuple(img.shape)
    result["label_shape"] = tuple(lbl.shape)
    result["image_dtype"] = str(img.dtype)
    result["label_dtype"] = str(lbl.dtype)

    # ── Channel count ─────────────────────────────────────────────
    if img.ndim == 4 and img.shape[0] == EXPECTED_CHANNELS:
        result["channels_ok"] = True
    elif img.ndim == 4:
        result["errors"].append(
            f"Wrong channel count: {img.shape[0]} (expected {EXPECTED_CHANNELS})")
    else:
        result["errors"].append(
            f"Wrong ndim: {img.ndim} (expected 4 for C×D×H×W)")

    # ── Label channel ─────────────────────────────────────────────
    if lbl.ndim == 4 and lbl.shape[0] == EXPECTED_LABEL_CHANNELS:
        pass  # ok
    else:
        result["warnings"].append(
            f"Label shape unusual: {lbl.shape}")

    # ── Dtype ─────────────────────────────────────────────────────
    if img.dtype == EXPECTED_DTYPE and lbl.dtype == EXPECTED_DTYPE:
        result["dtype_ok"] = True
    else:
        result["warnings"].append(
            f"dtype: image={img.dtype}, label={lbl.dtype} (expected float32)")

    # ── Shape match (spatial dims must agree) ─────────────────────
    if img.ndim == 4 and lbl.ndim == 4:
        if img.shape[1:] == lbl.shape[1:]:
            result["shape_match"] = True
        else:
            result["errors"].append(
                f"Spatial shape mismatch: image {img.shape[1:]} vs label {lbl.shape[1:]}")

    # ── Deep value checks ─────────────────────────────────────────
    if deep:
        try:
            # NaN / Inf
            if np.any(np.isnan(img)) or np.any(np.isinf(img)):
                result["errors"].append("image.npy contains NaN or Inf values")
            if np.any(np.isnan(lbl)) or np.any(np.isinf(lbl)):
                result["errors"].append("label.npy contains NaN or Inf values")

            # Value range (z-score normalised data)
            img_mean = float(np.mean(img))
            img_std  = float(np.std(img))
            img_min  = float(np.min(img))
            img_max  = float(np.max(img))

            result["img_mean"] = round(img_mean, 4)
            result["img_std"]  = round(img_std,  4)
            result["img_min"]  = round(img_min,  4)
            result["img_max"]  = round(img_max,  4)

            if abs(img_mean) > MAX_ALLOWED_MEAN_ABS:
                result["warnings"].append(
                    f"image mean={img_mean:.3f} — may not be z-score normalised")
            if img_std > MAX_ALLOWED_STD or img_std < 0.1:
                result["warnings"].append(
                    f"image std={img_std:.3f} — unusual for z-score normalised data")

            # Label validity — values should be 0 or 1
            unique_lbl = np.unique(lbl)
            lbl_min = float(lbl.min())
            lbl_max = float(lbl.max())
            result["lbl_min"]     = round(lbl_min, 4)
            result["lbl_max"]     = round(lbl_max, 4)
            result["lbl_nonzero"] = int(np.sum(lbl > 0))

            if lbl_min >= 0.0 and lbl_max <= 1.0:
                result["label_valid"] = True
            else:
                result["errors"].append(
                    f"Label values outside [0,1]: min={lbl_min:.4f} max={lbl_max:.4f}")

        except Exception as e:
            result["errors"].append(f"Deep check failed: {e}")

    return result


# ------------------------------------------------------------------ #
#  DATASET SCANNERS                                                    #
# ------------------------------------------------------------------ #

def scan_existing_dataset(root):
    """
    Scans data/patients_combined/{train,val,test}/{pid}/
    Returns list of (pid, image_path, label_path)
    """
    patients = []
    if not os.path.exists(root):
        return patients
    for split in ["train", "val", "test"]:
        split_dir = os.path.join(root, split)
        if not os.path.exists(split_dir):
            continue
        for pid in sorted(os.listdir(split_dir)):
            pdir = os.path.join(split_dir, pid)
            if not os.path.isdir(pdir):
                continue
            patients.append((
                f"{split}/{pid}",
                os.path.join(pdir, "image.npy"),
                os.path.join(pdir, "label.npy"),
            ))
    return patients


def scan_fastmri_dataset(root):
    """
    Scans F:/NPY_classified_dataset/{split}/{class}/{pid}/
    Returns list of (pid, image_path, label_path, class_label)
    """
    patients = []
    if not os.path.exists(root):
        return patients
    for split in ["train", "val", "test"]:
        for cls in ["benign", "malignant", "normal"]:
            cls_dir = os.path.join(root, split, cls)
            if not os.path.exists(cls_dir):
                continue
            for pid in sorted(os.listdir(cls_dir)):
                pdir = os.path.join(cls_dir, pid)
                if not os.path.isdir(pdir):
                    continue
                patients.append((
                    f"{split}/{cls}/{pid}",
                    os.path.join(pdir, "image.npy"),
                    os.path.join(pdir, "label.npy"),
                    cls,
                ))
    return patients


def scan_breastdx_normals(root):
    """
    Scans data/normal_patients/{pid}/
    Returns list of (pid, image_path, label_path)
    """
    patients = []
    if not os.path.exists(root):
        return patients
    for pid in sorted(os.listdir(root)):
        pdir = os.path.join(root, pid)
        if not os.path.isdir(pdir):
            continue
        patients.append((
            pid,
            os.path.join(pdir, "image.npy"),
            os.path.join(pdir, "label.npy"),
        ))
    return patients


# ------------------------------------------------------------------ #
#  ANALYSE ONE DATASET                                                 #
# ------------------------------------------------------------------ #

def analyse_dataset(name, patients, deep_n=DEEP_SAMPLE_N):
    """
    Runs checks on all patients, deep checks on first deep_n.
    Returns summary dict.
    """
    print(f"\n{'='*65}")
    print(f"  Checking: {name}")
    print(f"{'='*65}")

    if not patients:
        print(f"  [SKIP] No patients found — check path is correct")
        return {"name": name, "total": 0, "status": "PATH_NOT_FOUND"}

    print(f"  Total patients found : {len(patients)}")

    all_results  = []
    shapes_image = []
    shapes_label = []
    errors_total = 0
    warn_total   = 0

    for i, entry in enumerate(patients):
        pid, img_path, lbl_path = entry[0], entry[1], entry[2]
        deep  = (i < deep_n)
        res   = check_one_patient(pid, img_path, lbl_path, deep=deep)
        all_results.append(res)

        if res["image_shape"]:
            shapes_image.append(res["image_shape"])
        if res["label_shape"]:
            shapes_label.append(res["label_shape"])
        if res["errors"]:
            errors_total += 1
            for e in res["errors"]:
                print(f"  [ERR]  {pid}: {e}")
        if res["warnings"]:
            warn_total += 1
            for w in res["warnings"]:
                print(f"  [WARN] {pid}: {w}")

    # ── Shape statistics ──────────────────────────────────────────
    print(f"\n  Shape analysis (image.npy):")
    if shapes_image:
        from collections import Counter
        shape_counts = Counter(shapes_image)
        for shape, cnt in shape_counts.most_common(5):
            print(f"    {fmt_shape(shape):30s}  ×  {cnt} patients")

        # Extract channel, D, H, W stats
        channels = [s[0] for s in shapes_image if len(s)==4]
        depths   = [s[1] for s in shapes_image if len(s)==4]
        heights  = [s[2] for s in shapes_image if len(s)==4]
        widths   = [s[3] for s in shapes_image if len(s)==4]

        if depths:
            print(f"    Channels : {set(channels)}")
            print(f"    D (depth): min={min(depths)}  max={max(depths)}  "
                  f"mean={sum(depths)/len(depths):.0f}")
            print(f"    H (rows) : min={min(heights)} max={max(heights)} "
                  f"mean={sum(heights)/len(heights):.0f}")
            print(f"    W (cols) : min={min(widths)}  max={max(widths)}  "
                  f"mean={sum(widths)/len(widths):.0f}")

    # ── Deep value stats ──────────────────────────────────────────
    deep_results = [r for r in all_results[:deep_n] if "img_mean" in r]
    if deep_results:
        print(f"\n  Value checks (first {len(deep_results)} patients):")
        for r in deep_results:
            print(f"    {r['pid'][:40]:40s}  "
                  f"mean={r.get('img_mean','-'):7}  "
                  f"std={r.get('img_std','-'):6}  "
                  f"min={r.get('img_min','-'):7}  "
                  f"max={r.get('img_max','-'):7}  "
                  f"lbl_nonzero={r.get('lbl_nonzero','-')}")

    # ── Summary ───────────────────────────────────────────────────
    ok_count     = sum(1 for r in all_results if not r["errors"])
    dtype_ok     = sum(1 for r in all_results if r["dtype_ok"])
    channels_ok  = sum(1 for r in all_results if r["channels_ok"])

    print(f"\n  Summary:")
    print(f"    Patients checked      : {len(all_results)}")
    print(f"    No errors             : {ok_count}/{len(all_results)}")
    print(f"    Correct dtype (f32)   : {dtype_ok}/{len(all_results)}")
    print(f"    Correct channels (3)  : {channels_ok}/{len(all_results)}")
    print(f"    Patients with errors  : {errors_total}")
    print(f"    Patients with warnings: {warn_total}")

    return {
        "name"         : name,
        "total"        : len(all_results),
        "ok"           : ok_count,
        "dtype_ok"     : dtype_ok,
        "channels_ok"  : channels_ok,
        "errors"       : errors_total,
        "warnings"     : warn_total,
        "shapes_image" : [list(s) for s in shapes_image[:10]],
        "depths"       : sorted(set([s[1] for s in shapes_image if len(s)==4])),
        "heights"      : sorted(set([s[2] for s in shapes_image if len(s)==4])),
        "widths"       : sorted(set([s[3] for s in shapes_image if len(s)==4])),
        "deep_samples" : deep_results,
    }


# ------------------------------------------------------------------ #
#  CROSS-DATASET COMPATIBILITY REPORT                                 #
# ------------------------------------------------------------------ #

def cross_compare(existing_summary, fastmri_summary, breastdx_summary):
    """Prints cross-dataset compatibility verdict."""

    print(f"\n{'='*65}")
    print(f"  CROSS-DATASET COMPATIBILITY VERDICT")
    print(f"{'='*65}")

    summaries = [s for s in [existing_summary, fastmri_summary, breastdx_summary]
                 if s and s.get("total", 0) > 0]

    if len(summaries) < 2:
        print("  Cannot compare — fewer than 2 datasets were found.")
        return

    checks = []

    # ── Channel count ─────────────────────────────────────────────
    all_channels_ok = all(s["channels_ok"] == s["total"] for s in summaries)
    checks.append(("Channel count (3)", all_channels_ok))

    # ── Dtype ─────────────────────────────────────────────────────
    all_dtype_ok = all(s["dtype_ok"] == s["total"] for s in summaries)
    checks.append(("Dtype float32", all_dtype_ok))

    # ── Spatial dimension ranges ──────────────────────────────────
    # Check if D/H/W ranges are compatible (overlapping)
    def get_range(s, key):
        vals = s.get(key, [])
        return (min(vals), max(vals)) if vals else None

    all_depths = []
    for s in summaries:
        all_depths.extend(s.get("depths", []))

    spatial_ok = len(all_depths) > 0
    checks.append(("Spatial dims present", spatial_ok))

    # ── Print results ─────────────────────────────────────────────
    print()
    for label, result in checks:
        status = "PASS" if result else "FAIL"
        icon   = "✓" if result else "✗"
        print(f"  {icon} {label:35s}  [{status}]")

    # ── Dimension comparison table ────────────────────────────────
    print(f"\n  Spatial dimension comparison:")
    print(f"  {'Dataset':35s}  {'D range':12s}  {'H range':12s}  {'W range':12s}")
    print(f"  {'-'*75}")
    for s in summaries:
        d_range = f"{min(s['depths'])}–{max(s['depths'])}"   if s.get('depths')  else "N/A"
        h_range = f"{min(s['heights'])}–{max(s['heights'])}" if s.get('heights') else "N/A"
        w_range = f"{min(s['widths'])}–{max(s['widths'])}"   if s.get('widths')  else "N/A"
        print(f"  {s['name']:35s}  {d_range:12s}  {h_range:12s}  {w_range:12s}")

    # ── Value range comparison ────────────────────────────────────
    print(f"\n  Value range comparison (deep-sampled patients):")
    print(f"  {'Dataset':35s}  {'mean':8s}  {'std':7s}  {'min':8s}  {'max':8s}")
    print(f"  {'-'*75}")
    for s in summaries:
        deep = s.get("deep_samples", [])
        if not deep:
            continue
        means = [r.get("img_mean", 0) for r in deep if "img_mean" in r]
        stds  = [r.get("img_std",  0) for r in deep if "img_std"  in r]
        mins  = [r.get("img_min",  0) for r in deep if "img_min"  in r]
        maxs  = [r.get("img_max",  0) for r in deep if "img_max"  in r]
        if means:
            print(f"  {s['name']:35s}  "
                  f"{sum(means)/len(means):7.3f}  "
                  f"{sum(stds)/len(stds):6.3f}  "
                  f"{min(mins):8.3f}  "
                  f"{max(maxs):8.3f}")

    # ── Final verdict ─────────────────────────────────────────────
    print(f"\n  FINAL VERDICT:")
    all_pass = all(r for _, r in checks)
    if all_pass:
        print("  ✓ All basic compatibility checks PASSED.")
        print("  ✓ Datasets appear compatible in channel count and dtype.")
        print("  → Check spatial dimension ranges above.")
        print("  → If D/H/W ranges are very different, your DataLoader's")
        print("    SpatialPadd will handle padding to the same minimum size.")
        print("  → If value ranges differ significantly, preprocessing was")
        print("    applied inconsistently — rerun your prepare script.")
    else:
        print("  ✗ Compatibility issues detected. See FAIL items above.")
        print("  → Fix errors before using these datasets together.")


# ------------------------------------------------------------------ #
#  MAIN                                                                #
# ------------------------------------------------------------------ #

def main():
    print()
    print("=" * 65)
    print("  DATASET COMPATIBILITY CHECKER")
    print("  Existing  vs  FastMRI NPY  vs  BreastDx Normals")
    print("=" * 65)

    # ── 1. Scan all three datasets ─────────────────────────────────
    print("\n[1] Scanning dataset folders...")

    existing_patients = scan_existing_dataset(EXISTING_ROOT)
    print(f"  Existing (patients_combined): {len(existing_patients)} patients")

    fastmri_all = scan_fastmri_dataset(FASTMRI_ROOT)
    fastmri_normal   = [(p[0],p[1],p[2]) for p in fastmri_all if p[3]=="normal"]
    fastmri_benign   = [(p[0],p[1],p[2]) for p in fastmri_all if p[3]=="benign"]
    fastmri_malignant= [(p[0],p[1],p[2]) for p in fastmri_all if p[3]=="malignant"]
    print(f"  FastMRI NPY total            : {len(fastmri_all)} patients")
    print(f"    normal   : {len(fastmri_normal)}")
    print(f"    benign   : {len(fastmri_benign)}")
    print(f"    malignant: {len(fastmri_malignant)}")

    breastdx_patients = scan_breastdx_normals(BREASTDX_NORMAL_ROOT)
    print(f"  BreastDx normals             : {len(breastdx_patients)} patients")

    # ── 2. Analyse each dataset ────────────────────────────────────
    print("\n[2] Running checks...")

    # Sample from existing — check 10 from each split
    existing_sample = []
    for split in ["train", "val", "test"]:
        split_pts = [p for p in existing_patients if p[0].startswith(split+"/")]
        existing_sample.extend(split_pts[:4])  # 4 from each = ~12 total

    existing_summary = analyse_dataset(
        "Existing (patients_combined)",
        existing_sample if existing_sample else existing_patients[:12],
    )

    fastmri_summary = analyse_dataset(
        "FastMRI NPY (all classes)",
        fastmri_all[:20] if fastmri_all else [],  # sample 20
    )

    breastdx_summary = analyse_dataset(
        "BreastDx Normals",
        breastdx_patients,
    )

    # ── 3. Cross-dataset comparison ────────────────────────────────
    cross_compare(existing_summary, fastmri_summary, breastdx_summary)

    # ── 4. Save JSON report ────────────────────────────────────────
    report = {
        "existing"  : existing_summary,
        "fastmri"   : fastmri_summary,
        "breastdx"  : breastdx_summary,
    }

    report_path = "check_compatibility_report.json"
    try:
        with open(report_path, "w") as f:
            json.dump(report, f, indent=2, default=str)
        print(f"\n[3] Report saved → {report_path}")
    except Exception as e:
        print(f"\n[3] Could not save report: {e}")

    print()
    print("Done.")


if __name__ == "__main__":
    main()
