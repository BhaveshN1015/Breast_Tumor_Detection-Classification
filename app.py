"""
app.py — Breast Tumor AI · Streamlit Frontend
==============================================
3-Stage DCE-MRI Pipeline: Detection → Segmentation → Classification

Run:
    streamlit run app.py

Place this file at: D:/Breast_Tumor_AI_Project/app.py
"""

import os
import sys
import math
import warnings
import numpy as np
import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────────────────────────────────────
#  PROJECT ROOT & SRC PATHS
# ─────────────────────────────────────────────────────────────────────────────
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))

for sub in [
    "src/classification/stage1",
    "src/classification/stage3",
    "src/classification",
    "src/segmentation_3d",
    "src/dynunet_3d",
]:
    p = os.path.join(PROJECT_ROOT, sub)
    if p not in sys.path:
        sys.path.insert(0, p)

# ─────────────────────────────────────────────────────────────────────────────
#  CONSTANTS
# ─────────────────────────────────────────────────────────────────────────────
GITHUB_REPO  = "https://github.com/BhaveshN1015/Breast_Tumor_Detection-Classification"
GITHUB_DATA  = f"{GITHUB_REPO}/tree/main/sample_dataset"
HF_REPO      = "https://huggingface.co/B1015/breast-tumor-ai"

# ─────────────────────────────────────────────────────────────────────────────
#  COLOUR PALETTE  (matches notebook exactly)
# ─────────────────────────────────────────────────────────────────────────────
BG     = "#0d1117"
CARD   = "#161b22"
BORDER = "#30363d"
TEXT   = "#c9d1d9"
MUTED  = "#8b949e"
BLUE   = "#3b82f6"
GREEN  = "#10b981"
AMBER  = "#f59e0b"
RED    = "#ef4444"
PURPLE = "#8b5cf6"
GT_COL = (0.2, 0.9, 0.2)   # exact match from notebook

# Plotly base layout — NO margin here, set per chart
PLOTLY_BASE = dict(
    paper_bgcolor=BG,
    plot_bgcolor=CARD,
    font=dict(color=TEXT, family="monospace"),
)

# ─────────────────────────────────────────────────────────────────────────────
#  PAGE CONFIG
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Breast Tumor AI",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="collapsed",
)

st.markdown(f"""
<style>
    .stApp {{ background-color: {BG}; color: {TEXT}; }}
    .main .block-container {{ padding: 1.5rem 2rem; max-width: 1100px; }}
    h1,h2,h3,h4 {{ color: {TEXT}; font-family: monospace; }}
    p, li {{ color: {TEXT}; }}

    .disclaimer-box {{
        background: {CARD}; border: 2px solid {AMBER};
        border-radius: 12px; padding: 2rem; margin-bottom: 1.5rem;
    }}
    .disclaimer-title {{
        color: {AMBER}; font-size: 1.1rem; font-weight: bold;
        font-family: monospace; margin-bottom: 0.75rem;
    }}
    .disclaimer-text {{ color: {TEXT}; font-size: 0.9rem; line-height: 1.7; }}

    .step-card {{
        background: {CARD}; border: 1px solid {BORDER}; border-radius: 10px;
        padding: 1.2rem 1.4rem; margin-bottom: 0.75rem;
        display: flex; align-items: flex-start; gap: 1rem;
    }}
    .step-number {{
        background: {BLUE}; color: white; font-family: monospace;
        font-weight: bold; font-size: 1rem; min-width: 32px; height: 32px;
        border-radius: 50%; display: flex; align-items: center;
        justify-content: center; flex-shrink: 0;
    }}
    .step-content {{ flex: 1; }}
    .step-title {{ color: {TEXT}; font-weight: bold; font-family: monospace; margin-bottom: 0.3rem; }}
    .step-desc  {{ color: {MUTED}; font-size: 0.85rem; line-height: 1.5; }}

    .result-card {{
        background: {CARD}; border-radius: 10px; padding: 1.2rem;
        text-align: center; border: 1px solid {BORDER};
    }}
    .result-label {{ color: {MUTED}; font-size: 0.75rem; font-family: monospace;
                     text-transform: uppercase; letter-spacing: 1px; }}
    .result-value {{ font-size: 2rem; font-weight: bold; font-family: monospace; margin: 0.3rem 0; }}
    .result-sub   {{ color: {MUTED}; font-size: 0.8rem; font-family: monospace; }}

    .stage-header {{
        background: {CARD}; border-left: 3px solid {BLUE};
        border-radius: 0 8px 8px 0; padding: 0.6rem 1rem;
        margin: 1rem 0 0.5rem 0; font-family: monospace;
        font-weight: bold; color: {TEXT}; font-size: 0.95rem;
    }}

    div[data-testid="stSidebar"] {{ background: {CARD}; }}

    .stButton > button {{
        background: {BLUE}; color: white; border: none;
        border-radius: 8px; font-family: monospace;
        font-weight: bold; padding: 0.5rem 1.5rem;
    }}
    .stButton > button:hover {{ background: #2563eb; }}

    .stTabs [data-baseweb="tab-list"] {{ background: {CARD}; border-bottom: 1px solid {BORDER}; gap: 0; }}
    .stTabs [data-baseweb="tab"] {{ color: {MUTED}; font-family: monospace; padding: 0.6rem 1.2rem; }}
    .stTabs [aria-selected="true"] {{ color: {TEXT}; border-bottom: 2px solid {BLUE}; background: transparent; }}
    hr {{ border-color: {BORDER}; }}
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
#  SESSION STATE
# ─────────────────────────────────────────────────────────────────────────────
for k, v in [("disclaimer_accepted", False), ("models_loaded", False),
             ("current_step", 1), ("pipeline_results", None)]:
    if k not in st.session_state:
        st.session_state[k] = v


# ─────────────────────────────────────────────────────────────────────────────
#  HELPERS
# ─────────────────────────────────────────────────────────────────────────────
def result_card(label, value, sub, color=TEXT):
    return f"""<div class="result-card">
        <div class="result-label">{label}</div>
        <div class="result-value" style="color:{color};">{value}</div>
        <div class="result-sub">{sub}</div>
    </div>"""

def stage_header(icon, title, color=BLUE):
    return f'<div class="stage-header"><span style="color:{color};">{icon}</span> {title}</div>'

def prob_gauge(prob, threshold=0.5, positive_label="POSITIVE", negative_label="NEGATIVE",
               pos_color=RED, neg_color=GREEN):
    pred  = prob >= threshold
    color = pos_color if pred else neg_color
    label = positive_label if pred else negative_label
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=prob * 100,
        number=dict(suffix="%", font=dict(color=color, size=28, family="monospace")),
        gauge=dict(
            axis=dict(range=[0, 100], tickcolor=MUTED, tickfont=dict(color=MUTED, size=10)),
            bar=dict(color=color, thickness=0.25),
            bgcolor=CARD, bordercolor=BORDER,
            steps=[dict(range=[0, threshold*100], color="#1a2a1a"),
                   dict(range=[threshold*100, 100], color="#2a1a1a")],
            threshold=dict(line=dict(color=AMBER, width=3), thickness=0.8, value=threshold*100),
        ),
        title=dict(text=label, font=dict(color=color, size=13, family="monospace")),
    ))
    # margin set here directly — not via PLOTLY_BASE to avoid conflict
    fig.update_layout(paper_bgcolor=BG, plot_bgcolor=CARD,
                      font=dict(color=TEXT, family="monospace"),
                      height=220, margin=dict(t=50, b=10, l=20, r=20))
    return fig


# ─────────────────────────────────────────────────────────────────────────────
#  MODEL LOADING
# ─────────────────────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def load_all_models():
    import torch
    from download_models import download_all_models
    paths  = download_all_models()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    models = {"device": device, "paths": paths}

    try:
        from model_stage1 import build_model as build_s1
        m = build_s1()
        ck = torch.load(paths["stage1"], map_location=device, weights_only=False)
        m.load_state_dict(ck.get("model_state", ck) if isinstance(ck, dict) else ck)
        models["stage1"] = m.to(device).eval()
    except Exception as e:
        models["stage1_err"] = str(e)

    try:
        from model_3d import get_model as get_monai
        from model_dynunet import get_dynunet
        mm = get_monai(device)
        ck = torch.load(paths["seg_monai"], map_location=device, weights_only=False)
        mm.load_state_dict(ck.get("model_state", ck) if isinstance(ck, dict) else ck)
        models["seg_monai"] = mm.to(device).eval()
        md = get_dynunet(device)
        ck = torch.load(paths["seg_dyn"], map_location=device, weights_only=False)
        md.load_state_dict(ck.get("model_state", ck) if isinstance(ck, dict) else ck)
        models["seg_dyn"] = md.to(device).eval()
    except Exception as e:
        models["seg_err"] = str(e)

    try:
        from model_stage3 import get_model_stage3
        m = get_model_stage3(device)
        ck = torch.load(paths["stage3"], map_location=device, weights_only=False)
        m.load_state_dict(ck.get("model_state", ck) if isinstance(ck, dict) else ck)
        models["stage3"] = m.to(device).eval()
    except Exception as e:
        models["stage3_err"] = str(e)

    return models


# ─────────────────────────────────────────────────────────────────────────────
#  INFERENCE
# ─────────────────────────────────────────────────────────────────────────────
def run_stage1(models, img_tensor):
    import torch
    import torch.nn.functional as F
    device = models["device"]
    t = F.interpolate(img_tensor, size=(128, 192, 192), mode="trilinear", align_corners=False)
    with torch.no_grad():
        logit = models["stage1"](t.to(device))
    return float(torch.sigmoid(logit).cpu().item())


def run_segmentation(models, img_tensor):
    import torch
    from monai.inferers import sliding_window_inference
    from scipy import ndimage
    device = models["device"]
    img    = img_tensor.to(device)

    with torch.no_grad():
        pred_m = torch.sigmoid(sliding_window_inference(
            img, roi_size=(96,96,96), sw_batch_size=2,
            predictor=models["seg_monai"], overlap=0.5, mode="gaussian"))
        raw_d = sliding_window_inference(
            img, roi_size=(96,96,96), sw_batch_size=2,
            predictor=models["seg_dyn"], overlap=0.5, mode="gaussian")
        if isinstance(raw_d, (tuple, list)):
            raw_d = raw_d[0]
        pred_d = torch.sigmoid(raw_d)

    blend  = (0.35 * pred_m + 0.65 * pred_d).squeeze().cpu().numpy()
    binary = (blend > 0.5).astype(np.uint8)
    struct = ndimage.generate_binary_structure(3, 1)
    closed = ndimage.binary_closing(binary, structure=struct, iterations=2)
    labeled, n = ndimage.label(closed)
    if n > 0:
        sizes   = ndimage.sum(closed, labeled, range(1, n+1))
        largest = int(np.argmax(sizes)) + 1
        binary  = (labeled == largest).astype(np.uint8)
    else:
        binary = closed.astype(np.uint8)
    return binary, blend


def run_stage3(models, img_np, mask_np):
    import torch
    import torch.nn.functional as F
    device = models["device"]
    vox    = np.argwhere(mask_np > 0)
    if len(vox) > 0:
        centroid = vox.mean(axis=0).astype(int)
    else:
        centroid = np.array([s//2 for s in img_np.shape[1:]])
    d, h, w = centroid
    D, H, W = img_np.shape[1], img_np.shape[2], img_np.shape[3]
    d0,d1 = max(0,d-32), min(D,d+32)
    h0,h1 = max(0,h-32), min(H,h+32)
    w0,w1 = max(0,w-32), min(W,w+32)
    crop = img_np[:, d0:d1, h0:h1, w0:w1]
    t = F.interpolate(torch.tensor(crop).float().unsqueeze(0),
                      size=(64,64,64), mode="trilinear", align_corners=False)
    with torch.no_grad():
        logit = models["stage3"](t.to(device))
    return float(torch.sigmoid(logit).cpu().item())


def compute_features(img_np, mask_np):
    m3d = mask_np if mask_np.ndim == 3 else mask_np[0]
    vox = int(m3d.sum())
    if vox == 0:
        return None
    volume_mm3 = round(vox * 1.5**3, 1)
    diam_mm    = round((6 * volume_mm3 / math.pi) ** (1/3), 1)
    p1 = float(img_np[0][m3d > 0].mean())
    p2 = float(img_np[1][m3d > 0].mean())
    p3 = float(img_np[2][m3d > 0].mean())
    enhancement = round((p2 - p1) / (abs(p1) + 1e-6), 4)
    washout     = round((p2 - p3) / (abs(p2) + 1e-6), 4)
    mean_int    = round(float(img_np[1][m3d > 0].mean()), 4)
    std_int     = round(float(img_np[1][m3d > 0].std()),  4)
    min_int     = round(float(img_np[1][m3d > 0].min()),  4)
    max_int     = round(float(img_np[1][m3d > 0].max()),  4)
    contrast    = round(max_int - min_int, 4)
    homogeneity = round(max(0, 1 - std_int / (contrast + 1e-6)), 4)

    if p3 < p2 * 0.9:
        kinetic = "Type III — Washout"; kinetic_color = RED
    elif p3 > p2 * 1.1:
        kinetic = "Type I — Persistent"; kinetic_color = GREEN
    else:
        kinetic = "Type II — Plateau"; kinetic_color = AMBER

    # Surface area approximation
    from scipy import ndimage
    struct = ndimage.generate_binary_structure(3, 1)
    eroded = ndimage.binary_erosion(m3d.astype(bool), structure=struct)
    surface_vox = m3d.astype(bool) & ~eroded
    surface_mm2 = round(float(surface_vox.sum()) * 1.5**2, 1)

    # Sphericity
    if surface_mm2 > 0:
        sphericity = round((math.pi ** (1/3)) * (6 * volume_mm3) ** (2/3) / surface_mm2, 4)
    else:
        sphericity = 0.0

    # Bounding box
    vox_coords = np.argwhere(m3d > 0)
    bbox_min   = vox_coords.min(axis=0)
    bbox_max   = vox_coords.max(axis=0)
    bbox_dims  = (bbox_max - bbox_min + 1) * 1.5
    centroid   = vox_coords.mean(axis=0).astype(int).tolist()

    # T-stage estimate
    if volume_mm3 < 500:
        t_stage = "T1 (< 20mm)"
    elif volume_mm3 < 4000:
        t_stage = "T2 (20–50mm)"
    else:
        t_stage = "T3 (> 50mm)"

    return dict(
        voxels=vox, volume_mm3=volume_mm3, diam_mm=diam_mm,
        surface_area_mm2=surface_mm2, sphericity=sphericity,
        p1=p1, p2=p2, p3=p3, dce_means=[p1, p2, p3],
        enhancement_ratio=enhancement, washout_rate=washout,
        kinetic_type=kinetic, kinetic_color=kinetic_color,
        mean_intensity=mean_int, std_intensity=std_int,
        min_intensity=min_int, max_intensity=max_int,
        contrast=contrast, homogeneity=homogeneity,
        centroid=centroid,
        bbox_dims_mm=f"{bbox_dims[0]:.1f}×{bbox_dims[1]:.1f}×{bbox_dims[2]:.1f}",
        t_stage=t_stage,
    )


def compute_dice(pred, gt):
    p = pred.astype(bool); g = gt.astype(bool)
    inter = np.logical_and(p, g).sum()
    return round(float(2 * inter / (p.sum() + g.sum() + 1e-8)), 4)


# ─────────────────────────────────────────────────────────────────────────────
#  SEGMENTATION VISUALIZATIONS — matching notebook exactly (matplotlib)
# ─────────────────────────────────────────────────────────────────────────────
def vis_slice_overlay(img_np, lbl_np, pred_mask, feats, pid="patient", n_slices=5, channel=1):
    """
    VIS A — Exact match to notebook vis_slice_overlay.
    Row 1: P2 channel + GT contour (green) + Predicted contour (red)
    Row 2: Raw MRI slice
    """
    D = img_np.shape[1]
    # Find tumor center slice
    if lbl_np is not None and lbl_np.sum() > 0:
        cz = int(np.argwhere(lbl_np > 0.5).mean(0)[0])
    else:
        cz = D // 2
    sl_idx = np.linspace(max(0, cz - n_slices*2), min(D-1, cz + n_slices*2), n_slices, dtype=int)
    ch_name = ["P1 (pre-contrast)", "P2 (early post)", "P3 (late post)"][channel]

    fig, axes = plt.subplots(2, n_slices, figsize=(n_slices*3.5, 7),
                              facecolor=BG, gridspec_kw={"hspace": 0.04, "wspace": 0.04})
    dice_str = f"  |  Dice={feats.get('_dice','—')}" if feats and feats.get("_dice") else ""
    vol_str  = f"  |  Vol: {feats.get('volume_mm3','—')} mm³" if feats else ""
    fig.suptitle(f"Patient {pid}{dice_str}{vol_str}", color="white", fontsize=12, y=1.01)

    for si, sl in enumerate(sl_idx):
        # Row 1: channel + GT + pred contours
        ax = axes[0, si]; ax.set_facecolor("#000")
        slc = img_np[channel, sl]
        lo, hi = (np.percentile(slc, [1, 99]) if np.ptp(slc) > 0 else (0, 1))
        ax.imshow(slc, cmap="gray", vmin=lo, vmax=hi, aspect="equal")
        # GT contour — green (exact notebook colour)
        if lbl_np is not None and sl < lbl_np.shape[0] and lbl_np[sl].sum() > 0:
            ax.contour(lbl_np[sl], levels=[0.5], colors=[GT_COL], linewidths=2.0)
        # Predicted contour — red
        if pred_mask is not None and sl < pred_mask.shape[0] and pred_mask[sl].sum() > 0:
            ax.contour(pred_mask[sl], levels=[0.5], colors=["#ef4444"], linewidths=1.5)
        ax.set_title(f"Slice {sl}", color=MUTED, fontsize=8)
        if si == 0: ax.set_ylabel(f"{ch_name}\n+ GT contour", color="white", fontsize=7)
        ax.axis("off")

        # Row 2: raw MRI
        ax2 = axes[1, si]; ax2.set_facecolor("#000")
        ax2.imshow(slc, cmap="bone", vmin=lo, vmax=hi, aspect="equal")
        if si == 0: ax2.set_ylabel("Raw MRI", color="white", fontsize=7)
        ax2.set_title(f"Slice {sl}", color=MUTED, fontsize=8)
        ax2.axis("off")

    handles = [mpatches.Patch(edgecolor=GT_COL, facecolor="none", label="GT tumour", linewidth=2)]
    if pred_mask is not None:
        handles.append(mpatches.Patch(edgecolor="#ef4444", facecolor="none", label="Predicted", linewidth=2))
    fig.legend(handles=handles, loc="lower right", facecolor=CARD, labelcolor="white", fontsize=9)
    plt.tight_layout()
    return fig


def vis_all_channels(img_np, lbl_np, pid="patient", n_slices=5):
    """VIS B — All 3 DCE channels with GT contour, matching notebook exactly."""
    D  = img_np.shape[1]
    cz = int(np.argwhere(lbl_np > 0.5).mean(0)[0]) if (lbl_np is not None and lbl_np.sum() > 0) else D//2
    sl_idx   = np.linspace(max(0, cz-n_slices*2), min(D-1, cz+n_slices*2), n_slices, dtype=int)
    cmaps    = ["gray", "inferno", "viridis"]
    ch_names = ["P1 (pre-contrast)", "P2 (early post-contrast)", "P3 (late post-contrast)"]

    fig, axes = plt.subplots(3, n_slices, figsize=(n_slices*3.4, 9),
                              facecolor=BG, gridspec_kw={"hspace": 0.04, "wspace": 0.04})
    fig.suptitle(f"DCE-MRI — All Channels: Patient {pid}", color="white", fontsize=13, y=1.01)

    for ch in range(3):
        for si, sl in enumerate(sl_idx):
            ax = axes[ch, si]; ax.set_facecolor("#000")
            slc = img_np[ch, sl]
            lo, hi = (np.percentile(slc, [1, 99]) if np.ptp(slc) > 0 else (0, 1))
            ax.imshow(slc, cmap=cmaps[ch], vmin=lo, vmax=hi, aspect="equal")
            if lbl_np is not None and sl < lbl_np.shape[0] and lbl_np[sl].sum() > 0:
                ax.contour(lbl_np[sl], levels=[0.5], colors=[GT_COL], linewidths=1.8)
            if si == 0: ax.set_ylabel(ch_names[ch], color="white", fontsize=8)
            if ch == 0: ax.set_title(f"Slice {sl}", color=MUTED, fontsize=8)
            ax.axis("off")

    gt_patch = mpatches.Patch(edgecolor=GT_COL, facecolor="none", label="GT tumour", linewidth=2)
    fig.legend(handles=[gt_patch], loc="lower right", facecolor=CARD, labelcolor="white", fontsize=9)
    plt.tight_layout()
    return fig


def vis_mip(img_np, lbl_np, pred_mask, pid="patient", feats=None):
    """VIS C — MIP across 3 axes with GT (green fill) + Pred (red fill). Exact notebook match."""
    vol_ch = img_np[1]
    mips   = [vol_ch.max(axis=0), vol_ch.max(axis=1), vol_ch.max(axis=2)]
    gts    = ([lbl_np.max(axis=0), lbl_np.max(axis=1), lbl_np.max(axis=2)]
               if lbl_np is not None else [None, None, None])
    preds  = ([pred_mask.max(axis=0), pred_mask.max(axis=1), pred_mask.max(axis=2)]
               if pred_mask is not None else [None, None, None])
    titles = ["Axial (D)", "Coronal (H)", "Sagittal (W)"]
    dice_str = feats.get("_dice", "—") if feats else "—"

    # White background — matches notebook exactly
    fig, axes = plt.subplots(1, 3, figsize=(17, 6), facecolor="white")
    fig.suptitle(f"Max-Intensity Projection — Patient {pid}  |  Dice={dice_str}",
                 color="black", fontsize=13)

    for ax, mip, gt, pred, title in zip(axes, mips, gts, preds, titles):
        ax.set_facecolor("black")
        lo, hi = (np.percentile(mip, [1, 99]) if np.ptp(mip) > 0 else (0, 1))
        ax.imshow(mip, cmap="gray", vmin=lo, vmax=hi, aspect="auto")

        # GT — bright solid green fill (alpha 0.55, matches notebook)
        if gt is not None and gt.max() > 0:
            # Binarize properly — handles both float and int masks
            gt_bin  = (gt > 0).astype(np.float32)
            gt_rgba = np.zeros((*gt_bin.shape, 4), dtype=np.float32)
            gt_rgba[gt_bin > 0] = [0.20, 0.85, 0.20, 0.60]   # bright green, alpha=0.60
            ax.imshow(gt_rgba, aspect="auto", interpolation="nearest")

        # Prediction — solid red fill
        if pred is not None and pred.max() > 0:
            pred_bin  = (pred > 0).astype(np.float32)
            pred_rgba = np.zeros((*pred_bin.shape, 4), dtype=np.float32)
            pred_rgba[pred_bin > 0] = [0.85, 0.08, 0.08, 0.80]
            ax.imshow(pred_rgba, aspect="auto", interpolation="nearest")

        ax.set_title(title, fontsize=11, color="black", fontweight="bold")
        ax.axis("off")

    handles = [mpatches.Patch(facecolor=(0.20, 0.85, 0.20, 0.7), label="GT tumour (green)")]
    if pred_mask is not None:
        handles.append(mpatches.Patch(facecolor=(0.85, 0.08, 0.08, 0.85),
                                       label="Ensemble prediction (red)"))
    fig.legend(handles=handles, loc="lower center", facecolor="white",
               labelcolor="black", fontsize=10, ncol=2, framealpha=1.0)
    plt.tight_layout()
    return fig


def vis_feature_panel(img_np, lbl_np, feats, pid="patient"):
    """VIS E — Full feature panel: geometry, texture, DCE, MIP+bbox, histogram, radar."""
    f = feats or {}
    fig = plt.figure(figsize=(18, 9), facecolor=BG)
    fig.suptitle(f"Tumour Feature Report — Patient {pid}",
                 color="white", fontsize=15, y=1.01, fontweight="bold")
    gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.45, wspace=0.35)

    # ── Geometry panel ──────────────────────────────────────────────
    ax1 = fig.add_subplot(gs[0, 0]); ax1.set_facecolor(CARD); ax1.axis("off")
    geom = [
        ("Volume",         f"{f.get('volume_mm3', 0):.1f} mm³"),
        ("Voxels",         f"{f.get('voxels', 0):,}"),
        ("Diameter",       f"{f.get('diam_mm', 0):.1f} mm"),
        ("BBox (mm)",      f"{f.get('bbox_dims_mm', '—')}"),
        ("Centroid",       str(f.get("centroid", "—"))),
        ("Surface area",   f"{f.get('surface_area_mm2', 0):.1f} mm²"),
        ("Sphericity",     f"{f.get('sphericity', 0):.4f}"),
        ("T-stage est.",   f.get("t_stage", "—")),
    ]
    ax1.text(0.5, 0.98, "Shape & Geometry", ha="center", color=AMBER, fontsize=11,
             fontweight="bold", transform=ax1.transAxes, va="top")
    for i, (k, v) in enumerate(geom):
        y = 0.88 - i * 0.105
        ax1.text(0.02, y, k+":", color=MUTED, fontsize=9, transform=ax1.transAxes, va="center")
        ax1.text(0.98, y, v, color=TEXT, fontsize=9, transform=ax1.transAxes, va="center", ha="right")
        if i > 0: ax1.axhline(y+0.05, color=BORDER, lw=0.4, xmin=0.01, xmax=0.99)

    # ── Intensity & Texture ─────────────────────────────────────────
    ax2 = fig.add_subplot(gs[0, 1]); ax2.set_facecolor(CARD); ax2.axis("off")
    tex = [
        ("Mean intensity",  f"{f.get('mean_intensity', 0):.4f}"),
        ("Std intensity",   f"{f.get('std_intensity', 0):.4f}"),
        ("Min intensity",   f"{f.get('min_intensity', 0):.4f}"),
        ("Max intensity",   f"{f.get('max_intensity', 0):.4f}"),
        ("Contrast",        f"{f.get('contrast', 0):.4f}"),
        ("Homogeneity",     f"{f.get('homogeneity', 0):.4f}"),
        ("Enhancement",     f"{f.get('enhancement_ratio', 0):.4f}"),
        ("Washout rate",    f"{f.get('washout_rate', 0):.4f}"),
    ]
    ax2.text(0.5, 0.98, "Intensity & Texture (P2)", ha="center", color=PURPLE, fontsize=11,
             fontweight="bold", transform=ax2.transAxes, va="top")
    for i, (k, v) in enumerate(tex):
        y = 0.88 - i * 0.104
        ax2.text(0.02, y, k+":", color=MUTED, fontsize=9, transform=ax2.transAxes, va="center")
        ax2.text(0.98, y, v, color=TEXT, fontsize=9, transform=ax2.transAxes, va="center", ha="right")
        if i > 0: ax2.axhline(y+0.05, color=BORDER, lw=0.4, xmin=0.01, xmax=0.99)

    # ── DCE Metrics panel ───────────────────────────────────────────
    ax3 = fig.add_subplot(gs[0, 2]); ax3.set_facecolor(CARD); ax3.axis("off")
    dce_means = f.get("dce_means", [0, 0, 0])
    kin_items = [
        ("Kinetic type",  f.get("kinetic_type", "—")),
        ("Enhancement",   f"{f.get('enhancement_ratio', 0):.4f}"),
        ("Washout rate",  f"{f.get('washout_rate', 0):.4f}"),
        ("P1 mean",       f"{dce_means[0]:.4f}"),
        ("P2 mean",       f"{dce_means[1]:.4f}"),
        ("P3 mean",       f"{dce_means[2]:.4f}"),
    ]
    ax3.text(0.5, 0.98, "DCE Metrics", ha="center", color=BLUE, fontsize=11,
             fontweight="bold", transform=ax3.transAxes, va="top")
    for i, (k, v) in enumerate(kin_items):
        y = 0.88 - i * 0.135
        ax3.text(0.02, y, k+":", color=MUTED, fontsize=9, transform=ax3.transAxes, va="center")
        ax3.text(0.98, y, v, color=TEXT, fontsize=9, transform=ax3.transAxes, va="center", ha="right")
        if i > 0: ax3.axhline(y+0.07, color=BORDER, lw=0.4, xmin=0.01, xmax=0.99)

    # ── Axial MIP + bbox + centroid ─────────────────────────────────
    ax4 = fig.add_subplot(gs[1, 0]); ax4.set_facecolor("#000")
    mip = img_np[1].max(axis=0)
    lo, hi = (np.percentile(mip, [1, 99]) if np.ptp(mip) > 0 else (0, 1))
    ax4.imshow(mip, cmap="gray", vmin=lo, vmax=hi, aspect="auto")
    if lbl_np is not None:
        gp = lbl_np.max(axis=0)
        if gp.max() > 0:
            ax4.contour(gp, levels=[0.5], colors=[GT_COL], linewidths=2)
    cent = f.get("centroid")
    if cent:
        ax4.scatter([cent[2]], [cent[1]], c=RED, s=80, zorder=5, marker="+", linewidths=2)
    ax4.set_title("Axial MIP + Centroid (+)", color="white", fontsize=9)
    ax4.axis("off")

    # ── Intensity histogram inside mask ─────────────────────────────
    ax5 = fig.add_subplot(gs[1, 1]); ax5.set_facecolor(CARD)
    if lbl_np is not None and lbl_np.sum() > 0:
        mask = lbl_np.astype(bool)
        for ch, col, lbl_ch in zip([0, 1, 2], [BLUE, GREEN, AMBER], ["P1", "P2", "P3"]):
            ax5.hist(img_np[ch][mask], bins=40, color=col, alpha=0.6, label=lbl_ch, density=True)
        ax5.set_xlabel("Intensity", color=MUTED, fontsize=9)
        ax5.set_ylabel("Density",   color=MUTED, fontsize=9)
        ax5.legend(facecolor=BG, labelcolor="white", fontsize=8)
    ax5.set_title("Intensity Histogram (mask)", color="white", fontsize=9)
    ax5.tick_params(colors="white")
    ax5.spines[:].set_color(BORDER)

    # ── Radar chart ─────────────────────────────────────────────────
    ax6 = fig.add_subplot(gs[1, 2], polar=True); ax6.set_facecolor(CARD)
    labels_r = ["Sphericity", "Homogeneity", "Enhancement", "Washout-inv", "Compactness"]
    vals_r = [
        min(1.0, f.get("sphericity", 0)),
        min(1.0, f.get("homogeneity", 0)),
        min(1.0, abs(f.get("enhancement_ratio", 0)) / 2.0),
        max(0.0, 1.0 - abs(f.get("washout_rate", 0))),
        min(1.0, f.get("sphericity", 0) * 0.8),   # compactness proxy
    ]
    N      = len(labels_r)
    angles = [2*math.pi/N*i for i in range(N)] + [0]
    vals_p = vals_r + [vals_r[0]]
    ax6.plot(angles, vals_p, color=BLUE, lw=2)
    ax6.fill(angles, vals_p, color=BLUE, alpha=0.25)
    ax6.set_xticks(angles[:-1])
    ax6.set_xticklabels(labels_r, color="white", fontsize=8)
    ax6.set_yticks([0.25, 0.5, 0.75, 1.0]); ax6.set_yticklabels([])
    ax6.spines["polar"].set_color(BORDER)
    ax6.grid(color=BORDER, lw=0.7)
    ax6.set_facecolor(CARD)
    ax6.set_title("Shape Radar", color="white", fontsize=9, pad=12)

    plt.tight_layout()
    return fig


# ─────────────────────────────────────────────────────────────────────────────
#  DISCLAIMER
# ─────────────────────────────────────────────────────────────────────────────
if not st.session_state.disclaimer_accepted:
    st.markdown(f"""
    <div style="max-width:700px; margin: 3rem auto;">
        <div style="text-align:center; margin-bottom:2rem;">
            <span style="font-size:3rem;">🩺</span>
            <h1 style="font-family:monospace; color:{TEXT}; margin:0.5rem 0;">Breast Tumor AI</h1>
            <p style="color:{MUTED}; font-family:monospace;">3-Stage DCE-MRI Analysis Pipeline</p>
        </div>
    </div>""", unsafe_allow_html=True)

    st.markdown(f"""
    <div class="disclaimer-box">
        <div class="disclaimer-title">⚠️ IMPORTANT DISCLAIMER — Please Read Before Proceeding</div>
        <div class="disclaimer-text">
            <b>This application is strictly for research and educational demonstration purposes only.</b><br><br>
            • This tool is <b>NOT a medical device</b> and is <b>NOT intended for clinical use</b>.<br>
            • Results must <b>not</b> be used to make any medical or diagnostic decisions.<br>
            • The AI models were trained on a limited research dataset and may not generalise to all patient populations.<br>
            • Always consult a qualified radiologist or medical professional for any health-related concerns.<br>
            • The author assumes <b>no liability</b> for any decisions made based on this tool's output.<br><br>
            <b>By clicking "I Understand" you confirm research/educational use only.</b>
        </div>
    </div>""", unsafe_allow_html=True)

    c1, c2, c3 = st.columns([2,2,2])
    with c2:
        if st.button("✅ I Understand — Continue", type="primary", use_container_width=True):
            st.session_state.disclaimer_accepted = True
            st.rerun()
    st.stop()


# ─────────────────────────────────────────────────────────────────────────────
#  HEADER
# ─────────────────────────────────────────────────────────────────────────────
st.markdown(f"""
<div style="display:flex; align-items:center; gap:1rem; margin-bottom:0.5rem;">
    <span style="font-size:2.2rem;">🩺</span>
    <div>
        <h1 style="margin:0; font-family:monospace; font-size:1.8rem; color:{TEXT};">Breast Tumor AI</h1>
        <p style="margin:0; color:{MUTED}; font-family:monospace; font-size:0.85rem;">
            3-Stage DCE-MRI Pipeline · Detection → Segmentation → Classification
        </p>
    </div>
    <div style="margin-left:auto; text-align:right;">
        <a href="{GITHUB_REPO}" target="_blank" style="color:{BLUE}; font-family:monospace; font-size:0.8rem; text-decoration:none;">GitHub ↗</a>
        &nbsp;&nbsp;
        <a href="{HF_REPO}" target="_blank" style="color:{AMBER}; font-family:monospace; font-size:0.8rem; text-decoration:none;">Models ↗</a>
    </div>
</div>
<div style="background:{AMBER}22; border:1px solid {AMBER}44; border-radius:8px; padding:0.5rem 1rem; margin-bottom:1.5rem;">
    <span style="color:{AMBER}; font-size:0.8rem; font-family:monospace;">
        ⚠️ For research demonstration only · Not for clinical diagnosis
    </span>
</div>
""", unsafe_allow_html=True)

tab_run, tab_about = st.tabs(["▶️  Run Pipeline", "ℹ️  About the Project"])


# ═══════════════════════════════════════════════════════════════════════════════
#  TAB 1 — RUN PIPELINE
# ═══════════════════════════════════════════════════════════════════════════════
with tab_run:

    # Step tracker
    steps = ["Get Sample Data", "Load Models", "Upload Patient", "View Results"]
    step_cols = st.columns(4)
    for i, (col, label) in enumerate(zip(step_cols, steps)):
        active = (i+1 == st.session_state.current_step)
        done   = (i+1 < st.session_state.current_step)
        bg     = BLUE if active else (GREEN+"33" if done else CARD)
        border = BLUE if active else (GREEN if done else BORDER)
        tc     = "white" if active else (GREEN if done else MUTED)
        icon   = "✓" if done else str(i+1)
        col.markdown(f"""
        <div style="background:{bg}; border:1px solid {border}; border-radius:8px;
                    padding:0.6rem; text-align:center;">
            <div style="color:{tc}; font-family:monospace; font-weight:bold; font-size:0.75rem;">
                {icon}. {label}
            </div>
        </div>""", unsafe_allow_html=True)
    st.markdown("<br>", unsafe_allow_html=True)

    # ── STEP 1 ────────────────────────────────────────────────────────────────
    with st.expander("📥  Step 1 — Get Sample Patient Data",
                     expanded=(st.session_state.current_step == 1)):
        st.markdown(f"""
        <div class="step-card">
            <div class="step-number">1</div>
            <div class="step-content">
                <div class="step-title">Open the GitHub sample dataset</div>
                <div class="step-desc">
                    <a href="{GITHUB_DATA}" target="_blank" style="color:{BLUE}; font-family:monospace;">
                        {GITHUB_DATA} ↗
                    </a>
                </div>
            </div>
        </div>
        <div class="step-card">
            <div class="step-number">2</div>
            <div class="step-content">
                <div class="step-title">Choose a sample patient folder</div>
                <div class="step-desc">
                    📁 <b style="color:{RED};">positive/67</b> — Malignant tumor
                    <span style="color:{MUTED};">(Best segmentation result · Dice 0.9582)</span><br>
                    📁 <b style="color:{GREEN};">negative/234</b> — No tumor present (normal)
                </div>
            </div>
        </div>
        <div class="step-card">
            <div class="step-number">3</div>
            <div class="step-content">
                <div class="step-title">Download the .npy files</div>
                <div class="step-desc">
                    Inside the patient folder, download:<br>
                    &nbsp;&nbsp;• <b>image.npy</b> — 3-channel DCE-MRI volume (required)<br>
                    &nbsp;&nbsp;• <b>label.npy</b> — ground truth mask (optional · enables Dice score + GT overlay)<br><br>
                    Click the file → click the <b>Download raw file</b> button (↓) on GitHub.
                </div>
            </div>
        </div>
        <div class="step-card">
            <div class="step-number">4</div>
            <div class="step-content">
                <div class="step-title">Come back here and go to Step 2</div>
                <div class="step-desc">Once downloaded, click below to continue.</div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        if st.button("I have the files — Next →", key="step1_next"):
            st.session_state.current_step = 2
            st.rerun()

    # ── STEP 2 ────────────────────────────────────────────────────────────────
    with st.expander("⚙️  Step 2 — Load AI Models",
                     expanded=(st.session_state.current_step == 2)):
        st.markdown(f"""
        <div class="step-card">
            <div class="step-number" style="background:{PURPLE};">↓</div>
            <div class="step-content">
                <div class="step-title">Download models from Hugging Face Hub</div>
                <div class="step-desc">
                    Stored at: <a href="{HF_REPO}" target="_blank" style="color:{AMBER}; font-family:monospace;">{HF_REPO} ↗</a><br>
                    Downloads once — cached locally after. Total size ~570 MB.
                </div>
            </div>
        </div>""", unsafe_allow_html=True)

        model_info = [
            ("stage1",    "Stage 1 — ResNet18 + CBAM",   "~382 MB", BLUE),
            ("seg_monai", "Segmentation — MONAI UNet3D",  "~50 MB",  GREEN),
            ("seg_dyn",   "Segmentation — DynUNet",       "~64 MB",  GREEN),
            ("stage3",    "Stage 3 — EfficientNet-B0",    "~54 MB",  PURPLE),
        ]

        if not st.session_state.models_loaded:
            if st.button("🚀 Load All Models", type="primary", key="load_btn"):
                bars = {}
                for key, name, size, color in model_info:
                    st.markdown(f'<div style="color:{MUTED}; font-family:monospace; font-size:0.85rem;">{name} ({size})</div>',
                                unsafe_allow_html=True)
                    bars[key] = st.progress(0)
                try:
                    with st.spinner("Downloading and loading models..."):
                        models = load_all_models()
                    for key, _, _, _ in model_info:
                        bars[key].progress(100)
                    st.session_state.models_loaded = True
                    errs = [k for k in ["stage1","seg_monai","seg_dyn","stage3"]
                            if f"{k}_err" in models or ("seg_err" in models and k.startswith("seg"))]
                    if errs:
                        st.warning(f"Some models failed to load: {errs}")
                    else:
                        st.success("✅ All 4 models loaded successfully!")
                except Exception as e:
                    st.error(f"Loading failed: {e}")
        else:
            for _, name, _, _ in model_info:
                st.markdown(f'<span style="color:{GREEN}; font-family:monospace; font-size:0.85rem;">✓ {name}</span>',
                            unsafe_allow_html=True)
            st.success("Models are ready.")

        if st.session_state.models_loaded:
            if st.button("Next →", key="step2_next"):
                st.session_state.current_step = 3
                st.rerun()

    # ── STEP 3 ────────────────────────────────────────────────────────────────
    with st.expander("📂  Step 3 — Upload Patient File",
                     expanded=(st.session_state.current_step == 3)):
        col_img, col_mask = st.columns(2)
        with col_img:
            img_file = st.file_uploader("image.npy — required",  type=["npy"], key="img_up")
        with col_mask:
            mask_file = st.file_uploader("label.npy — optional (GT mask)", type=["npy"], key="mask_up")

        if img_file is not None:
            img_np = np.load(img_file)
            if img_np.ndim == 3:
                img_np = img_np[np.newaxis]

            if img_np.ndim != 4 or img_np.shape[0] != 3:
                st.error(f"Expected (3,D,H,W) but got {img_np.shape}. Upload image.npy, not label.npy.")
            else:
                D, H, W = img_np.shape[1], img_np.shape[2], img_np.shape[3]
                mask_np = None
                if mask_file is not None:
                    raw = np.load(mask_file).squeeze()
                    mask_np = raw if raw.ndim == 3 else raw[np.newaxis].squeeze()

                st.markdown(f"""
                <div style="background:{CARD}; border:1px solid {BORDER}; border-radius:8px;
                            padding:0.8rem 1rem; font-family:monospace; font-size:0.85rem;">
                    <span style="color:{GREEN};">✓</span> Volume loaded &nbsp;·&nbsp;
                    Shape: <span style="color:{BLUE};">{D}×{H}×{W}</span> voxels &nbsp;·&nbsp;
                    Size: <span style="color:{BLUE};">{D*1.5:.0f}×{H*1.5:.0f}×{W*1.5:.0f} mm</span>
                    {"&nbsp;·&nbsp;<span style='color:"+GREEN+";'>GT mask loaded ✓</span>" if mask_np is not None else ""}
                </div>""", unsafe_allow_html=True)

                # Quick preview
                sl_prev = st.slider("Preview slice", 0, D-1, D//2, key="prev_sl")
                fig_prev, axes_prev = plt.subplots(1, 3, figsize=(12, 4), facecolor=BG,
                                                    gridspec_kw={"wspace": 0.04})
                ch_names_p = ["P1 pre-contrast", "P2 peak enhancement", "P3 delayed"]
                cmaps_p    = ["gray", "gray", "bone"]
                for ch in range(3):
                    ax = axes_prev[ch]; ax.set_facecolor("#000")
                    slc = img_np[ch, sl_prev]
                    lo, hi = (np.percentile(slc,[1,99]) if np.ptp(slc)>0 else (0,1))
                    ax.imshow(slc, cmap=cmaps_p[ch], vmin=lo, vmax=hi, aspect="equal")
                    if mask_np is not None and mask_np[sl_prev].sum() > 0:
                        ax.contour(mask_np[sl_prev], levels=[0.5], colors=[GT_COL], linewidths=2)
                    ax.set_title(ch_names_p[ch], color=TEXT, fontsize=9)
                    ax.axis("off")
                plt.tight_layout()
                st.pyplot(fig_prev, use_container_width=True)
                plt.close(fig_prev)

                if not st.session_state.models_loaded:
                    st.warning("⚠️ Models not loaded. Go back to Step 2.")
                else:
                    if st.button("▶️ Run Full Pipeline", type="primary", key="run_btn"):
                        import torch
                        models    = load_all_models()
                        img_tensor = torch.tensor(img_np).float().unsqueeze(0)
                        results    = {}

                        # ── Stage 1 ──────────────────────────────────────────
                        bar = st.progress(0, text="Stage 1 — Tumor Detection...")
                        if "stage1" in models:
                            prob_s1 = run_stage1(models, img_tensor)
                            results["prob_s1"] = prob_s1
                            detected = prob_s1 >= 0.5
                            c1_ = RED if detected else GREEN
                            st.markdown(f'<span style="color:{c1_}; font-family:monospace; font-weight:bold;">{"🔴 Tumor Detected" if detected else "🟢 No Tumor"} — P(tumor) = {prob_s1:.4f}</span>',
                                        unsafe_allow_html=True)
                        else:
                            st.warning("Stage 1 model unavailable.")
                            detected = True
                        bar.progress(25, text="Stage 1 done ✓")

                        # ── Segmentation ─────────────────────────────────────
                        pred_mask = None
                        if detected and "seg_monai" in models and "seg_dyn" in models:
                            bar.progress(30, text="Segmentation — running sliding window inference...")
                            pred_mask, blend = run_segmentation(models, img_tensor)
                            results["pred_mask"] = pred_mask
                            results["blend"]     = blend
                            tvox = pred_mask.sum()
                            st.markdown(f'<span style="color:{GREEN}; font-family:monospace;">✓ Segmentation done — {tvox} tumor voxels · {tvox*1.5**3:.0f} mm³</span>',
                                        unsafe_allow_html=True)
                            if mask_np is not None:
                                dice = compute_dice(pred_mask, mask_np)
                                results["dice"] = dice
                                dc = GREEN if dice >= 0.7 else AMBER if dice >= 0.4 else RED
                                st.markdown(f'<span style="color:{dc}; font-family:monospace;">Dice vs GT: {dice}</span>',
                                            unsafe_allow_html=True)
                        bar.progress(65, text="Segmentation done ✓")

                        # ── Stage 3 ──────────────────────────────────────────
                        if detected and "stage3" in models:
                            bar.progress(70, text="Stage 3 — Benign vs Malignant...")
                            use_mask = pred_mask if pred_mask is not None else (
                                mask_np if mask_np is not None else
                                np.ones(img_np.shape[1:], dtype=np.uint8)
                            )
                            prob_s3 = run_stage3(models, img_np, use_mask)
                            results["prob_s3"] = prob_s3
                            c3_ = RED if prob_s3 >= 0.5 else AMBER
                            st.markdown(f'<span style="color:{c3_}; font-family:monospace; font-weight:bold;">{"🔴 MALIGNANT" if prob_s3>=0.5 else "🟡 BENIGN"} — P(malignant) = {prob_s3:.4f}</span>',
                                        unsafe_allow_html=True)
                        bar.progress(85, text="Stage 3 done ✓")

                        # ── Features ─────────────────────────────────────────
                        use_m = pred_mask if pred_mask is not None else mask_np
                        if use_m is not None:
                            feats = compute_features(img_np, use_m)
                            if feats and "dice" in results:
                                feats["_dice"] = results["dice"]
                            results["feats"] = feats

                        results.update({"img_np": img_np, "mask_np": mask_np})
                        bar.progress(100, text="Pipeline complete ✅")

                        st.session_state.pipeline_results = results
                        st.session_state.current_step = 4
                        st.success("✅ Pipeline complete — see results below!")
                        st.rerun()

    # ── STEP 4 — RESULTS ──────────────────────────────────────────────────────
    with st.expander("📊  Step 4 — Results",
                     expanded=(st.session_state.current_step == 4)):

        res = st.session_state.pipeline_results
        if res is None:
            st.info("Run the pipeline in Step 3 to see results here.")
        else:
            img_np    = res["img_np"]
            mask_np   = res.get("mask_np")
            pred_mask = res.get("pred_mask")
            feats     = res.get("feats") or {}
            D         = img_np.shape[1]

            # Normalise mask shapes
            lbl_3d  = (mask_np   if mask_np   is not None and mask_np.ndim   == 3 else
                       mask_np[0] if mask_np   is not None else None)
            pred_3d = (pred_mask if pred_mask is not None and pred_mask.ndim == 3 else None)

            # ── Summary metrics ───────────────────────────────────────────────
            st.markdown(stage_header("📋", "Summary"), unsafe_allow_html=True)
            mc = st.columns(4)
            if "prob_s1" in res:
                p = res["prob_s1"]
                mc[0].markdown(result_card("Stage 1 — Detection", f"{p*100:.1f}%",
                               "Tumor Detected" if p>=0.5 else "No Tumor",
                               RED if p>=0.5 else GREEN), unsafe_allow_html=True)
            if "dice" in res:
                d = res["dice"]
                mc[1].markdown(result_card("Segmentation Dice", str(d), "vs Ground Truth",
                               GREEN if d>=0.7 else AMBER if d>=0.4 else RED), unsafe_allow_html=True)
            elif pred_mask is not None:
                mc[1].markdown(result_card("Tumor Volume", f"{pred_mask.sum()*1.5**3:.0f}",
                               "mm³ (predicted)", BLUE), unsafe_allow_html=True)
            if "prob_s3" in res:
                p = res["prob_s3"]
                mc[2].markdown(result_card("Stage 3 — Classification", f"{p*100:.1f}%",
                               "Malignant" if p>=0.5 else "Benign",
                               RED if p>=0.5 else AMBER), unsafe_allow_html=True)
            if feats.get("kinetic_type"):
                mc[3].markdown(result_card("DCE Kinetic Type",
                               feats["kinetic_type"].split("—")[0].strip(),
                               feats["kinetic_type"],
                               feats.get("kinetic_color", TEXT)), unsafe_allow_html=True)

            # ── Probability gauges ────────────────────────────────────────────
            if "prob_s1" in res or "prob_s3" in res:
                st.markdown(stage_header("📈", "Probability Gauges"), unsafe_allow_html=True)
                gc1, gc2 = st.columns(2)
                if "prob_s1" in res:
                    gc1.plotly_chart(prob_gauge(res["prob_s1"],
                                               positive_label="TUMOR DETECTED",
                                               negative_label="NO TUMOR"),
                                     use_container_width=True)
                    gc1.markdown(f'<p style="text-align:center;color:{MUTED};font-family:monospace;font-size:0.8rem;">Stage 1 — Tumor Detection (ResNet18+CBAM)</p>',
                                 unsafe_allow_html=True)
                if "prob_s3" in res:
                    gc2.plotly_chart(prob_gauge(res["prob_s3"],
                                               positive_label="MALIGNANT",
                                               negative_label="BENIGN",
                                               pos_color=RED, neg_color=AMBER),
                                     use_container_width=True)
                    gc2.markdown(f'<p style="text-align:center;color:{MUTED};font-family:monospace;font-size:0.8rem;">Stage 3 — Benign vs Malignant (EfficientNet-B0)</p>',
                                 unsafe_allow_html=True)

            # ── VIS A: Slice overlay ──────────────────────────────────────────
            st.markdown(stage_header("🔬", "Segmentation — Slice Overlay (VIS A)"), unsafe_allow_html=True)
            n_sl = st.slider("Number of slices to show", 3, 7, 5, key="n_sl")
            fig_a = vis_slice_overlay(img_np, lbl_3d, pred_3d, feats,
                                      pid=img_file.name.replace(".npy","") if "img_file" in dir() else "patient",
                                      n_slices=n_sl)
            st.pyplot(fig_a, use_container_width=True)
            plt.close(fig_a)
            st.markdown(f'<p style="color:{MUTED};font-family:monospace;font-size:0.78rem;">'
                        f'<span style="color:lime;">■</span> Green = GT mask &nbsp;·&nbsp;'
                        f'<span style="color:{RED};">■</span> Red = Predicted mask</p>',
                        unsafe_allow_html=True)

            # ── VIS B: All 3 channels ─────────────────────────────────────────
            if lbl_3d is not None:
                st.markdown(stage_header("🎨", "All 3 DCE Channels with GT Contour (VIS B)"), unsafe_allow_html=True)
                fig_b = vis_all_channels(img_np, lbl_3d, n_slices=n_sl)
                st.pyplot(fig_b, use_container_width=True)
                plt.close(fig_b)

            # ── VIS C: MIP ────────────────────────────────────────────────────
            st.markdown(stage_header("🗺️", "Max-Intensity Projection — 3 Axes (VIS C)"), unsafe_allow_html=True)
            fig_c = vis_mip(img_np, lbl_3d, pred_3d, feats=feats)
            st.pyplot(fig_c, use_container_width=True)
            plt.close(fig_c)

            # ── VIS E: Feature panel ──────────────────────────────────────────
            if feats and lbl_3d is not None:
                st.markdown(stage_header("📐", "Tumour Feature Report (VIS E)"), unsafe_allow_html=True)
                fig_e = vis_feature_panel(img_np, lbl_3d, feats)
                st.pyplot(fig_e, use_container_width=True)
                plt.close(fig_e)

            # ── Reset ─────────────────────────────────────────────────────────
            st.markdown("<br>", unsafe_allow_html=True)
            if st.button("🔄 Run another patient", key="reset_btn"):
                st.session_state.pipeline_results = None
                st.session_state.current_step = 3
                st.rerun()


# ═══════════════════════════════════════════════════════════════════════════════
#  TAB 2 — ABOUT
# ═══════════════════════════════════════════════════════════════════════════════
with tab_about:
    st.markdown(f"<h2 style='font-family:monospace;'>About the Project</h2>", unsafe_allow_html=True)

    col_a, col_b, col_c = st.columns(3)
    for col, icon, stage, model, metric, color in [
        (col_a,"🔵","Stage 1 — Detection","3D ResNet18 + CBAM\n33.4M params\nInput: (3,128,192,192)","AUC = 0.8765",BLUE),
        (col_b,"🟢","Segmentation","MONAI UNet3D + DynUNet\nBlend: 0.35 / 0.65\nSliding window 96³","Mean Dice = 0.80",GREEN),
        (col_c,"🟣","Stage 3 — Classification","3D EfficientNet-B0\n4.7M params\nInput: 64³ centroid crop","AUC = 0.9200",PURPLE),
    ]:
        col.markdown(f"""
        <div style="background:{CARD}; border:1px solid {color}44; border-radius:10px;
                    padding:1.2rem; text-align:center;">
            <div style="font-size:1.8rem;">{icon}</div>
            <div style="color:{color}; font-family:monospace; font-weight:bold; margin:0.5rem 0;">{stage}</div>
            <div style="color:{MUTED}; font-family:monospace; font-size:0.78rem; white-space:pre-line; margin-bottom:0.8rem;">{model}</div>
            <div style="color:{color}; font-family:monospace; font-weight:bold; font-size:1.1rem;">{metric}</div>
        </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown(f"""
    ### DCE-MRI — What the 3 channels mean

    | Channel | Phase | Clinical significance |
    |---|---|---|
    | **P1** | Pre-contrast | Baseline tissue · no contrast yet |
    | **P2** | Peak enhancement | Malignant tumors enhance aggressively (high vascularity) |
    | **P3** | Delayed (washout) | Malignant tumors wash out fast · benign tumors persist |

    The P2→P3 relationship defines the **kinetic type** (I Persistent / II Plateau / III Washout) —
    the radiological basis of BI-RADS classification.
    """)

    st.divider()
    lc1, lc2 = st.columns(2)
    lc1.markdown(f"""
    <div style="background:{CARD}; border:1px solid {BORDER}; border-radius:8px; padding:1rem;">
        <div style="color:{TEXT}; font-family:monospace; font-weight:bold; margin-bottom:0.5rem;">📁 GitHub Repository</div>
        <a href="{GITHUB_REPO}" target="_blank" style="color:{BLUE}; font-family:monospace; font-size:0.85rem;">{GITHUB_REPO} ↗</a><br>
        <div style="color:{MUTED}; font-size:0.8rem; margin-top:0.5rem;">Source code · Sample dataset · README</div>
    </div>""", unsafe_allow_html=True)
    lc2.markdown(f"""
    <div style="background:{CARD}; border:1px solid {BORDER}; border-radius:8px; padding:1rem;">
        <div style="color:{TEXT}; font-family:monospace; font-weight:bold; margin-bottom:0.5rem;">🤗 Model Weights (Hugging Face)</div>
        <a href="{HF_REPO}" target="_blank" style="color:{AMBER}; font-family:monospace; font-size:0.85rem;">{HF_REPO} ↗</a><br>
        <div style="color:{MUTED}; font-size:0.8rem; margin-top:0.5rem;">Stage1 · Stage3 · MONAI UNet3D · DynUNet</div>
    </div>""", unsafe_allow_html=True)

# Footer
st.markdown(f"""
<div style="text-align:center; color:{MUTED}; font-family:monospace; font-size:0.72rem;
            padding:1.5rem 0 0.5rem 0; border-top:1px solid {BORDER}; margin-top:2rem;">
    Breast Tumor AI · Research Demo · Not for clinical use<br>
    ResNet18+CBAM · MONAI UNet3D · DynUNet · EfficientNet-B0 · 516 patients · DCE-MRI
</div>""", unsafe_allow_html=True)