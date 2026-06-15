import numpy as np
import os

root   = "data/patients_combined"
splits = ["train", "val", "test"]

all_stats = []

for split in splits:
    sp = os.path.join(root, split)
    if not os.path.exists(sp):
        continue
    for pid in sorted(os.listdir(sp), key=lambda x: int(x) if x.isdigit() else x):
        lbl_path = os.path.join(sp, pid, "label.npy")
        img_path = os.path.join(sp, pid, "image.npy")
        if not (os.path.exists(lbl_path) and os.path.exists(img_path)):
            continue
        lbl   = np.load(lbl_path)
        img   = np.load(img_path)
        tv    = int((lbl > 0).sum())
        shape = img.shape
        D, H, W = shape[1], shape[2], shape[3]
        vol   = D * H * W
        ratio = tv / vol * 100
        group = "new" if int(pid) > 100 else "old"

        flags = []
        if tv < 100:
            flags.append("TINY_TUMOR")
        if D < 48:
            flags.append("THIN_D")
        if H > 300 or W > 300:
            flags.append("LARGE_HW")
        if img.max() > 10.0:
            flags.append("HIGH_INTENSITY")
        if ratio > 5.0:
            flags.append("LARGE_TUMOR_RATIO")

        all_stats.append({
            "pid": pid, "split": split, "group": group,
            "tv": tv, "ratio": round(ratio, 4),
            "D": D, "H": H, "W": W,
            "img_max": round(float(img.max()), 3),
            "flags": flags,
        })

# Print summary
print(f"\n{'PID':>6} {'split':>5} {'grp':>4} {'tumor_v':>8} {'ratio%':>7} "
      f"{'D':>4} {'H':>4} {'W':>4} {'img_max':>8}  flags")
print("-" * 80)
flagged = []
for s in sorted(all_stats, key=lambda x: int(x["pid"])):
    flag_str = ", ".join(s["flags"]) if s["flags"] else ""
    print(f"{s['pid']:>6} {s['split']:>5} {s['group']:>4} {s['tv']:>8} "
          f"{s['ratio']:>7} {s['D']:>4} {s['H']:>4} {s['W']:>4} "
          f"{s['img_max']:>8}  {flag_str}")
    if s["flags"]:
        flagged.append(s)

print(f"\n{'='*80}")
print(f"Total patients : {len(all_stats)}")
print(f"Flagged        : {len(flagged)}")
print(f"\nFlagged patients detail:")
for s in flagged:
    print(f"  Patient {s['pid']:>4} [{s['split']:>5}] {s['group']:>3} — "
          f"tv={s['tv']:>6}  shape=({s['D']},{s['H']},{s['W']})  "
          f"flags: {', '.join(s['flags'])}")