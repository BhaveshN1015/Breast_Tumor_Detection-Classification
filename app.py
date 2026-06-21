"""
app.py — Breast Tumor AI · Streamlit Frontend
==============================================
2-Stage DCE-MRI Pipeline: Segmentation -> Classification

Run:
    streamlit run app.py
"""

# ── Standard library ──────────────────────────────────────────────────────────
import os
import sys
import math
import warnings

# ── Third-party ───────────────────────────────────────────────────────────────
import numpy as np
import torch
import torch.nn.functional as F
from scipy import ndimage
from monai.inferers import sliding_window_inference

import streamlit as st
import plotly.graph_objects as go

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

for _sub in [
    "src/classification/stage3",
    "src/classification",
    "src/segmentation_3d",
    "src/dynunet_3d",
]:
    _p = os.path.join(PROJECT_ROOT, _sub)
    if _p not in sys.path:
        sys.path.insert(0, _p)

# ─────────────────────────────────────────────────────────────────────────────
#  CONSTANTS
# ─────────────────────────────────────────────────────────────────────────────
GITHUB_REPO = (
    "https://github.com/BhaveshN1015/Breast_Tumor_Detection-Classification"
)
GITHUB_DATA = f"{GITHUB_REPO}/tree/main/sample_dataset"
HF_REPO     = "https://huggingface.co/B1015/breast-tumor-ai"

# ─────────────────────────────────────────────────────────────────────────────
#  COLOUR PALETTE
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
GT_COL = (0.2, 0.9, 0.2)

# ─────────────────────────────────────────────────────────────────────────────
#  PAGE CONFIG
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Breast Tumor AI",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ── CSS ───────────────────────────────────────────────────────────────────────
_CSS = f"""
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
    .step-title {{
        color: {TEXT}; font-weight: bold;
        font-family: monospace; margin-bottom: 0.3rem;
    }}
    .step-desc {{ color: {MUTED}; font-size: 0.85rem; line-height: 1.5; }}

    .result-card {{
        background: {CARD}; border-radius: 10px; padding: 1.2rem;
        text-align: center; border: 1px solid {BORDER};
    }}
    .result-label {{
        color: {MUTED}; font-size: 0.75rem; font-family: monospace;
        text-transform: uppercase; letter-spacing: 1px;
    }}
    .result-value {{
        font-size: 2rem; font-weight: bold;
        font-family: monospace; margin: 0.3rem 0;
    }}
    .result-sub {{ color: {MUTED}; font-size: 0.8rem; font-family: monospace; }}

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

    .stTabs [data-baseweb="tab-list"] {{
        background: {CARD}; border-bottom: 1px solid {BORDER}; gap: 0;
    }}
    .stTabs [data-baseweb="tab"] {{
        color: {MUTED}; font-family: monospace; padding: 0.6rem 1.2rem;
    }}
    .stTabs [aria-selected="true"] {{
        color: {TEXT}; border-bottom: 2px solid {BLUE}; background: transparent;
    }}
    hr {{ border-color: {BORDER}; }}
</style>
"""
st.markdown(_CSS, unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
#  SESSION STATE
# ─────────────────────────────────────────────────────────────────────────────
_DEFAULTS = {
    "disclaimer_accepted": False,
    "models_loaded": False,
    "current_step": 1,
    "pipeline_results": None,
}
for _k, _v in _DEFAULTS.items():
    if _k not in st.session_state:
        st.session_state[_k] = _v


# ─────────────────────────────────────────────────────────────────────────────
#  SMALL HTML HELPERS
# ─────────────────────────────────────────────────────────────────────────────
def result_card(label: str, value: str, sub: str, color: str = TEXT) -> str:
    return (
        '<div class="result-card">'
        f'<div class="result-label">{label}</div>'
        f'<div class="result-value" style="color:{color};">{value}</div>'
        f'<div class="result-sub">{sub}</div>'
        "</div>"
    )


def stage_header(icon: str, title: str, color: str = BLUE) -> str:
    return (
        '<div class="stage-header">'
        f'<span style="color:{color};">{icon}</span> {title}'
        "</div>"
    )


# ─────────────────────────────────────────────────────────────────────────────
#  PLOTLY GAUGE
# ─────────────────────────────────────────────────────────────────────────────
def prob_gauge(
    prob: float,
    threshold: float = 0.5,
    positive_label: str = "POSITIVE",
    negative_label: str = "NEGATIVE",
    pos_color: str = RED,
    neg_color: str = GREEN,
):
    is_pos = prob >= threshold
    color  = pos_color if is_pos else neg_color
    label  = positive_label if is_pos else negative_label
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=prob * 100,
        number=dict(suffix="%", font=dict(color=color, size=28, family="monospace")),
        gauge=dict(
            axis=dict(
                range=[0, 100],
                tickcolor=MUTED,
                tickfont=dict(color=MUTED, size=10),
            ),
            bar=dict(color=color, thickness=0.25),
            bgcolor=CARD,
            bordercolor=BORDER,
            steps=[
                dict(range=[0, threshold * 100], color="#1a2a1a"),
                dict(range=[threshold * 100, 100], color="#2a1a1a"),
            ],
            threshold=dict(
                line=dict(color=AMBER, width=3),
                thickness=0.8,
                value=threshold * 100,
            ),
        ),
        title=dict(text=label, font=dict(color=color, size=13, family="monospace")),
    ))
    fig.update_layout(
        paper_bgcolor=BG,
        plot_bgcolor=CARD,
        font=dict(color=TEXT, family="monospace"),
        height=220,
        margin=dict(t=50, b=10, l=20, r=20),
    )
    return fig


# ─────────────────────────────────────────────────────────────────────────────
#  MODEL LOADING
# ─────────────────────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def load_all_models() -> dict:
    # Patch sys.path inside the cached function.
    # @st.cache_resource runs in a context where the module-level
    # sys.path inserts may not be present, so we redo them here.
    _root = os.path.dirname(os.path.abspath(__file__))
    _src_paths = [
        _root,                                               # download_models.py
        os.path.join(_root, "src", "segmentation_3d"),      # model_3d.py
        os.path.join(_root, "src", "dynunet_3d"),           # model_dynunet.py
        os.path.join(_root, "src", "classification", "stage3"),  # model_stage3.py
        os.path.join(_root, "src", "classification"),
    ]
    for _sp in _src_paths:
        if _sp not in sys.path:
            sys.path.insert(0, _sp)

    from download_models import download_all_models      # noqa: PLC0415
    from model_3d import get_model as get_monai          # noqa: PLC0415
    from model_dynunet import get_dynunet                # noqa: PLC0415
    from model_stage3 import get_model_stage3            # noqa: PLC0415

    paths  = download_all_models()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    result: dict = {"device": device, "paths": paths}

    # ── Segmentation models ───────────────────────────────────────────────────
    try:
        mm = get_monai(device)
        ck = torch.load(paths["seg_monai"], map_location=device, weights_only=False)
        state = ck.get("model_state", ck) if isinstance(ck, dict) else ck
        mm.load_state_dict(state)
        result["seg_monai"] = mm.to(device).eval()

        md = get_dynunet(device)
        ck = torch.load(paths["seg_dyn"], map_location=device, weights_only=False)
        state = ck.get("model_state", ck) if isinstance(ck, dict) else ck
        md.load_state_dict(state)
        result["seg_dyn"] = md.to(device).eval()
    except Exception as exc:
        result["seg_err"] = str(exc)

    # ── Classification model ──────────────────────────────────────────────────
    try:
        m3 = get_model_stage3(device)
        ck = torch.load(paths["stage3"], map_location=device, weights_only=False)
        state = ck.get("model_state", ck) if isinstance(ck, dict) else ck
        m3.load_state_dict(state)
        result["stage3"] = m3.to(device).eval()
    except Exception as exc:
        result["stage3_err"] = str(exc)

    return result


# ─────────────────────────────────────────────────────────────────────────────
#  INFERENCE
# ─────────────────────────────────────────────────────────────────────────────
def run_segmentation(models: dict, img_tensor: torch.Tensor):
    device = models["device"]
    img    = img_tensor.to(device)

    with torch.no_grad():
        pred_m = torch.sigmoid(
            sliding_window_inference(
                img,
                roi_size=(96, 96, 96),
                sw_batch_size=2,
                predictor=models["seg_monai"],
                overlap=0.5,
                mode="gaussian",
            )
        )
        raw_d = sliding_window_inference(
            img,
            roi_size=(96, 96, 96),
            sw_batch_size=2,
            predictor=models["seg_dyn"],
            overlap=0.5,
            mode="gaussian",
        )
        if isinstance(raw_d, (tuple, list)):
            raw_d = raw_d[0]
        pred_d = torch.sigmoid(raw_d)

    blend  = (0.35 * pred_m + 0.65 * pred_d).squeeze().cpu().numpy()
    binary = (blend > 0.5).astype(np.uint8)

    struct = ndimage.generate_binary_structure(3, 1)
    closed = ndimage.binary_closing(binary, structure=struct, iterations=2)
    labeled, n = ndimage.label(closed)
    if n > 0:
        sizes   = ndimage.sum(closed, labeled, range(1, n + 1))
        largest = int(np.argmax(sizes)) + 1
        binary  = (labeled == largest).astype(np.uint8)
    else:
        binary = closed.astype(np.uint8)
    return binary, blend


def run_classification(models: dict, img_np: np.ndarray, mask_np: np.ndarray) -> float:
    device   = models["device"]
    vox      = np.argwhere(mask_np > 0)
    centroid = (
        vox.mean(axis=0).astype(int)
        if len(vox) > 0
        else np.array([s // 2 for s in img_np.shape[1:]])
    )
    d, h, w = centroid
    D = img_np.shape[1]
    H = img_np.shape[2]
    W = img_np.shape[3]
    d0, d1 = max(0, d - 32), min(D, d + 32)
    h0, h1 = max(0, h - 32), min(H, h + 32)
    w0, w1 = max(0, w - 32), min(W, w + 32)
    crop = img_np[:, d0:d1, h0:h1, w0:w1]
    t = F.interpolate(
        torch.tensor(crop).float().unsqueeze(0),
        size=(64, 64, 64),
        mode="trilinear",
        align_corners=False,
    )
    with torch.no_grad():
        logit = models["stage3"](t.to(device))
    return float(torch.sigmoid(logit).cpu().item())


def compute_features(img_np: np.ndarray, mask_np: np.ndarray):
    m3d = mask_np if mask_np.ndim == 3 else mask_np[0]
    vox = int(m3d.sum())
    if vox == 0:
        return None

    volume_mm3  = round(vox * 1.5 ** 3, 1)
    diam_mm     = round((6 * volume_mm3 / math.pi) ** (1 / 3), 1)

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
    homogeneity = round(max(0.0, 1.0 - std_int / (contrast + 1e-6)), 4)

    if p3 < p2 * 0.9:
        kinetic       = "Type III — Washout"
        kinetic_color = RED
    elif p3 > p2 * 1.1:
        kinetic       = "Type I — Persistent"
        kinetic_color = GREEN
    else:
        kinetic       = "Type II — Plateau"
        kinetic_color = AMBER

    struct      = ndimage.generate_binary_structure(3, 1)
    eroded      = ndimage.binary_erosion(m3d.astype(bool), structure=struct)
    surface_vox = m3d.astype(bool) & ~eroded
    surface_mm2 = round(float(surface_vox.sum()) * 1.5 ** 2, 1)
    sphericity  = (
        round((math.pi ** (1 / 3)) * (6 * volume_mm3) ** (2 / 3) / surface_mm2, 4)
        if surface_mm2 > 0
        else 0.0
    )

    vox_coords = np.argwhere(m3d > 0)
    bbox_min   = vox_coords.min(axis=0)
    bbox_max   = vox_coords.max(axis=0)
    bbox_dims  = (bbox_max - bbox_min + 1) * 1.5
    centroid   = vox_coords.mean(axis=0).astype(int).tolist()

    if volume_mm3 < 500:
        t_stage = "T1 (< 20mm)"
    elif volume_mm3 < 4000:
        t_stage = "T2 (20-50mm)"
    else:
        t_stage = "T3 (> 50mm)"

    bbox_str = (
        f"{bbox_dims[0]:.1f}"
        f"x{bbox_dims[1]:.1f}"
        f"x{bbox_dims[2]:.1f}"
    )

    return dict(
        voxels=vox,
        volume_mm3=volume_mm3,
        diam_mm=diam_mm,
        surface_area_mm2=surface_mm2,
        sphericity=sphericity,
        p1=p1, p2=p2, p3=p3,
        dce_means=[p1, p2, p3],
        enhancement_ratio=enhancement,
        washout_rate=washout,
        kinetic_type=kinetic,
        kinetic_color=kinetic_color,
        mean_intensity=mean_int,
        std_intensity=std_int,
        min_intensity=min_int,
        max_intensity=max_int,
        contrast=contrast,
        homogeneity=homogeneity,
        centroid=centroid,
        bbox_dims_mm=bbox_str,
        t_stage=t_stage,
    )


def compute_dice(pred: np.ndarray, gt: np.ndarray) -> float:
    p     = pred.astype(bool)
    g     = gt.astype(bool)
    inter = np.logical_and(p, g).sum()
    return round(float(2 * inter / (p.sum() + g.sum() + 1e-8)), 4)


# ─────────────────────────────────────────────────────────────────────────────
#  VISUALIZATIONS
# ─────────────────────────────────────────────────────────────────────────────
def vis_slice_overlay(img_np, lbl_np, pred_mask, feats, pid="patient", n_slices=5, channel=1):
    D  = img_np.shape[1]
    if lbl_np is not None and lbl_np.sum() > 0:
        cz = int(np.argwhere(lbl_np > 0.5).mean(0)[0])
    else:
        cz = D // 2

    sl_idx  = np.linspace(
        max(0, cz - n_slices * 2),
        min(D - 1, cz + n_slices * 2),
        n_slices,
        dtype=int,
    )
    ch_name = ["P1 (pre-contrast)", "P2 (early post)", "P3 (late post)"][channel]

    fig, axes = plt.subplots(
        2, n_slices,
        figsize=(n_slices * 3.5, 7),
        facecolor=BG,
        gridspec_kw={"hspace": 0.04, "wspace": 0.04},
    )

    dice_str = ""
    if feats and feats.get("_dice"):
        dice_str = f"  |  Dice={feats['_dice']}"
    vol_str = ""
    if feats:
        vol_str = f"  |  Vol: {feats.get('volume_mm3', '?')} mm3"

    fig.suptitle(
        f"Patient {pid}{dice_str}{vol_str}",
        color="white", fontsize=12, y=1.01,
    )

    for si, sl in enumerate(sl_idx):
        ax = axes[0, si]
        ax.set_facecolor("#000")
        slc    = img_np[channel, sl]
        lo, hi = (np.percentile(slc, [1, 99]) if np.ptp(slc) > 0 else (0, 1))
        ax.imshow(slc, cmap="gray", vmin=lo, vmax=hi, aspect="equal")

        if lbl_np is not None and sl < lbl_np.shape[0] and lbl_np[sl].sum() > 0:
            ax.contour(lbl_np[sl], levels=[0.5], colors=[GT_COL], linewidths=2.0)
        if (pred_mask is not None
                and sl < pred_mask.shape[0]
                and pred_mask[sl].sum() > 0):
            ax.contour(pred_mask[sl], levels=[0.5], colors=["#ef4444"], linewidths=1.5)

        ax.set_title(f"Slice {sl}", color=MUTED, fontsize=8)
        if si == 0:
            ax.set_ylabel(f"{ch_name}\n+ GT contour", color="white", fontsize=7)
        ax.axis("off")

        ax2 = axes[1, si]
        ax2.set_facecolor("#000")
        ax2.imshow(slc, cmap="bone", vmin=lo, vmax=hi, aspect="equal")
        if si == 0:
            ax2.set_ylabel("Raw MRI", color="white", fontsize=7)
        ax2.set_title(f"Slice {sl}", color=MUTED, fontsize=8)
        ax2.axis("off")

    handles = [
        mpatches.Patch(edgecolor=GT_COL, facecolor="none", label="GT tumour", linewidth=2)
    ]
    if pred_mask is not None:
        handles.append(
            mpatches.Patch(edgecolor="#ef4444", facecolor="none", label="Predicted", linewidth=2)
        )
    fig.legend(handles=handles, loc="lower right", facecolor=CARD, labelcolor="white", fontsize=9)
    plt.tight_layout()
    return fig


def vis_all_channels(img_np, lbl_np, pid="patient", n_slices=5):
    D  = img_np.shape[1]
    if lbl_np is not None and lbl_np.sum() > 0:
        cz = int(np.argwhere(lbl_np > 0.5).mean(0)[0])
    else:
        cz = D // 2

    sl_idx   = np.linspace(
        max(0, cz - n_slices * 2),
        min(D - 1, cz + n_slices * 2),
        n_slices,
        dtype=int,
    )
    cmaps    = ["gray", "inferno", "viridis"]
    ch_names = [
        "P1 (pre-contrast)",
        "P2 (early post-contrast)",
        "P3 (late post-contrast)",
    ]

    fig, axes = plt.subplots(
        3, n_slices,
        figsize=(n_slices * 3.4, 9),
        facecolor=BG,
        gridspec_kw={"hspace": 0.04, "wspace": 0.04},
    )
    fig.suptitle(f"DCE-MRI — All Channels: Patient {pid}", color="white", fontsize=13, y=1.01)

    for ch in range(3):
        for si, sl in enumerate(sl_idx):
            ax = axes[ch, si]
            ax.set_facecolor("#000")
            slc    = img_np[ch, sl]
            lo, hi = (np.percentile(slc, [1, 99]) if np.ptp(slc) > 0 else (0, 1))
            ax.imshow(slc, cmap=cmaps[ch], vmin=lo, vmax=hi, aspect="equal")
            if lbl_np is not None and sl < lbl_np.shape[0] and lbl_np[sl].sum() > 0:
                ax.contour(lbl_np[sl], levels=[0.5], colors=[GT_COL], linewidths=1.8)
            if si == 0:
                ax.set_ylabel(ch_names[ch], color="white", fontsize=8)
            if ch == 0:
                ax.set_title(f"Slice {sl}", color=MUTED, fontsize=8)
            ax.axis("off")

    gt_patch = mpatches.Patch(edgecolor=GT_COL, facecolor="none", label="GT tumour", linewidth=2)
    fig.legend(
        handles=[gt_patch], loc="lower right",
        facecolor=CARD, labelcolor="white", fontsize=9,
    )
    plt.tight_layout()
    return fig


def vis_mip(img_np, lbl_np, pred_mask, pid="patient", feats=None):
    vol_ch   = img_np[1]
    mips     = [vol_ch.max(axis=0), vol_ch.max(axis=1), vol_ch.max(axis=2)]
    gts      = (
        [lbl_np.max(axis=0), lbl_np.max(axis=1), lbl_np.max(axis=2)]
        if lbl_np is not None
        else [None, None, None]
    )
    preds    = (
        [pred_mask.max(axis=0), pred_mask.max(axis=1), pred_mask.max(axis=2)]
        if pred_mask is not None
        else [None, None, None]
    )
    titles   = ["Axial (D)", "Coronal (H)", "Sagittal (W)"]
    dice_str = feats.get("_dice", "?") if feats else "?"

    fig, axes = plt.subplots(1, 3, figsize=(17, 6), facecolor="white")
    fig.suptitle(
        f"Max-Intensity Projection — Patient {pid}  |  Dice={dice_str}",
        color="black", fontsize=13,
    )

    for ax, mip, gt, pred, title in zip(axes, mips, gts, preds, titles):
        ax.set_facecolor("black")
        lo, hi = (np.percentile(mip, [1, 99]) if np.ptp(mip) > 0 else (0, 1))
        ax.imshow(mip, cmap="gray", vmin=lo, vmax=hi, aspect="auto")

        if gt is not None and gt.max() > 0:
            gt_bin  = (gt > 0).astype(np.float32)
            gt_rgba = np.zeros((*gt_bin.shape, 4), dtype=np.float32)
            gt_rgba[gt_bin > 0] = [0.20, 0.85, 0.20, 0.60]
            ax.imshow(gt_rgba, aspect="auto", interpolation="nearest")

        if pred is not None and pred.max() > 0:
            pred_bin  = (pred > 0).astype(np.float32)
            pred_rgba = np.zeros((*pred_bin.shape, 4), dtype=np.float32)
            pred_rgba[pred_bin > 0] = [0.85, 0.08, 0.08, 0.80]
            ax.imshow(pred_rgba, aspect="auto", interpolation="nearest")

        ax.set_title(title, fontsize=11, color="black", fontweight="bold")
        ax.axis("off")

    handles = [mpatches.Patch(facecolor=(0.20, 0.85, 0.20, 0.7), label="GT tumour (green)")]
    if pred_mask is not None:
        handles.append(
            mpatches.Patch(facecolor=(0.85, 0.08, 0.08, 0.85), label="Ensemble prediction (red)")
        )
    fig.legend(
        handles=handles, loc="lower center",
        facecolor="white", labelcolor="black",
        fontsize=10, ncol=2, framealpha=1.0,
    )
    plt.tight_layout()
    return fig


def vis_feature_panel(img_np, lbl_np, feats, pid="patient"):
    f   = feats or {}
    fig = plt.figure(figsize=(18, 9), facecolor=BG)
    fig.suptitle(
        f"Tumour Feature Report — Patient {pid}",
        color="white", fontsize=15, y=1.01, fontweight="bold",
    )
    gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.45, wspace=0.35)

    # ── Geometry ──────────────────────────────────────────────────────────────
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.set_facecolor(CARD)
    ax1.axis("off")
    geom = [
        ("Volume",       f"{f.get('volume_mm3', 0):.1f} mm3"),
        ("Voxels",       f"{f.get('voxels', 0):,}"),
        ("Diameter",     f"{f.get('diam_mm', 0):.1f} mm"),
        ("BBox (mm)",    str(f.get("bbox_dims_mm", "?"))),
        ("Centroid",     str(f.get("centroid", "?"))),
        ("Surface area", f"{f.get('surface_area_mm2', 0):.1f} mm2"),
        ("Sphericity",   f"{f.get('sphericity', 0):.4f}"),
        ("T-stage est.", str(f.get("t_stage", "?"))),
    ]
    ax1.text(0.5, 0.98, "Shape & Geometry", ha="center", color=AMBER,
             fontsize=11, fontweight="bold", transform=ax1.transAxes, va="top")
    for i, (k, v) in enumerate(geom):
        y = 0.88 - i * 0.105
        ax1.text(0.02, y, k + ":", color=MUTED, fontsize=9,
                 transform=ax1.transAxes, va="center")
        ax1.text(0.98, y, v, color=TEXT, fontsize=9,
                 transform=ax1.transAxes, va="center", ha="right")
        if i > 0:
            ax1.axhline(y + 0.05, color=BORDER, lw=0.4, xmin=0.01, xmax=0.99)

    # ── Texture ───────────────────────────────────────────────────────────────
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.set_facecolor(CARD)
    ax2.axis("off")
    tex = [
        ("Mean intensity",  f"{f.get('mean_intensity', 0):.4f}"),
        ("Std intensity",   f"{f.get('std_intensity',  0):.4f}"),
        ("Min intensity",   f"{f.get('min_intensity',  0):.4f}"),
        ("Max intensity",   f"{f.get('max_intensity',  0):.4f}"),
        ("Contrast",        f"{f.get('contrast',       0):.4f}"),
        ("Homogeneity",     f"{f.get('homogeneity',    0):.4f}"),
        ("Enhancement",     f"{f.get('enhancement_ratio', 0):.4f}"),
        ("Washout rate",    f"{f.get('washout_rate',   0):.4f}"),
    ]
    ax2.text(0.5, 0.98, "Intensity & Texture (P2)", ha="center", color=PURPLE,
             fontsize=11, fontweight="bold", transform=ax2.transAxes, va="top")
    for i, (k, v) in enumerate(tex):
        y = 0.88 - i * 0.104
        ax2.text(0.02, y, k + ":", color=MUTED, fontsize=9,
                 transform=ax2.transAxes, va="center")
        ax2.text(0.98, y, v, color=TEXT, fontsize=9,
                 transform=ax2.transAxes, va="center", ha="right")
        if i > 0:
            ax2.axhline(y + 0.05, color=BORDER, lw=0.4, xmin=0.01, xmax=0.99)

    # ── DCE Metrics ───────────────────────────────────────────────────────────
    ax3 = fig.add_subplot(gs[0, 2])
    ax3.set_facecolor(CARD)
    ax3.axis("off")
    dce_means = f.get("dce_means", [0.0, 0.0, 0.0])
    kin_items = [
        ("Kinetic type", str(f.get("kinetic_type", "?"))),
        ("Enhancement",  f"{f.get('enhancement_ratio', 0):.4f}"),
        ("Washout rate", f"{f.get('washout_rate',      0):.4f}"),
        ("P1 mean",      f"{dce_means[0]:.4f}"),
        ("P2 mean",      f"{dce_means[1]:.4f}"),
        ("P3 mean",      f"{dce_means[2]:.4f}"),
    ]
    ax3.text(0.5, 0.98, "DCE Metrics", ha="center", color=BLUE,
             fontsize=11, fontweight="bold", transform=ax3.transAxes, va="top")
    for i, (k, v) in enumerate(kin_items):
        y = 0.88 - i * 0.135
        ax3.text(0.02, y, k + ":", color=MUTED, fontsize=9,
                 transform=ax3.transAxes, va="center")
        ax3.text(0.98, y, v, color=TEXT, fontsize=9,
                 transform=ax3.transAxes, va="center", ha="right")
        if i > 0:
            ax3.axhline(y + 0.07, color=BORDER, lw=0.4, xmin=0.01, xmax=0.99)

    # ── Axial MIP + centroid ──────────────────────────────────────────────────
    ax4 = fig.add_subplot(gs[1, 0])
    ax4.set_facecolor("#000")
    mip    = img_np[1].max(axis=0)
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

    # ── Intensity histogram ───────────────────────────────────────────────────
    ax5 = fig.add_subplot(gs[1, 1])
    ax5.set_facecolor(CARD)
    if lbl_np is not None and lbl_np.sum() > 0:
        mask_bool = lbl_np.astype(bool)
        for ch, col, lbl_ch in zip([0, 1, 2], [BLUE, GREEN, AMBER], ["P1", "P2", "P3"]):
            ax5.hist(img_np[ch][mask_bool], bins=40, color=col,
                     alpha=0.6, label=lbl_ch, density=True)
        ax5.set_xlabel("Intensity", color=MUTED, fontsize=9)
        ax5.set_ylabel("Density",   color=MUTED, fontsize=9)
        ax5.legend(facecolor=BG, labelcolor="white", fontsize=8)
    ax5.set_title("Intensity Histogram (mask)", color="white", fontsize=9)
    ax5.tick_params(colors="white")
    ax5.spines[:].set_color(BORDER)

    # ── Radar chart ───────────────────────────────────────────────────────────
    ax6 = fig.add_subplot(gs[1, 2], polar=True)
    ax6.set_facecolor(CARD)
    labels_r = ["Sphericity", "Homogeneity", "Enhancement", "Washout-inv", "Compactness"]
    vals_r   = [
        min(1.0, f.get("sphericity", 0)),
        min(1.0, f.get("homogeneity", 0)),
        min(1.0, abs(f.get("enhancement_ratio", 0)) / 2.0),
        max(0.0, 1.0 - abs(f.get("washout_rate", 0))),
        min(1.0, f.get("sphericity", 0) * 0.8),
    ]
    N      = len(labels_r)
    angles = [2 * math.pi / N * i for i in range(N)] + [0]
    vals_p = vals_r + [vals_r[0]]
    ax6.plot(angles, vals_p, color=BLUE, lw=2)
    ax6.fill(angles, vals_p, color=BLUE, alpha=0.25)
    ax6.set_xticks(angles[:-1])
    ax6.set_xticklabels(labels_r, color="white", fontsize=8)
    ax6.set_yticks([0.25, 0.5, 0.75, 1.0])
    ax6.set_yticklabels([])
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
    st.markdown(
        f'<div style="max-width:700px;margin:3rem auto;text-align:center;">'
        f'<span style="font-size:3rem;">🩺</span>'
        f'<h1 style="font-family:monospace;color:{TEXT};margin:0.5rem 0;">'
        f"Breast Tumor AI</h1>"
        f'<p style="color:{MUTED};font-family:monospace;">'
        f"DCE-MRI Segmentation &amp; Classification Pipeline</p>"
        f"</div>",
        unsafe_allow_html=True,
    )

    _DISC_HTML = (
        '<div class="disclaimer-box">'
        '<div class="disclaimer-title">'
        "&#9888;&#65039; IMPORTANT DISCLAIMER &mdash; Please Read Before Proceeding"
        "</div>"
        '<div class="disclaimer-text">'
        "<b>This application is strictly for research and educational demonstration purposes only.</b><br><br>"
        "&#8226; This tool is <b>NOT a medical device</b> and is <b>NOT intended for clinical use</b>.<br>"
        "&#8226; Results must <b>not</b> be used to make any medical or diagnostic decisions.<br>"
        "&#8226; The AI models were trained on a limited research dataset and may not generalise to all patient populations.<br>"
        "&#8226; Always consult a qualified radiologist or medical professional for any health-related concerns.<br>"
        "&#8226; The author assumes <b>no liability</b> for any decisions made based on this tool&#39;s output.<br><br>"
        "<b>By clicking &quot;I Understand&quot; you confirm research/educational use only.</b>"
        "</div></div>"
    )
    st.markdown(_DISC_HTML, unsafe_allow_html=True)

    _dc1, _dc2, _dc3 = st.columns([2, 2, 2])
    with _dc2:
        if st.button("I Understand — Continue", type="primary", use_container_width=True):
            st.session_state.disclaimer_accepted = True
            st.rerun()
    st.stop()


# ─────────────────────────────────────────────────────────────────────────────
#  HEADER
# ─────────────────────────────────────────────────────────────────────────────
_HEADER = (
    '<div style="display:flex;align-items:center;gap:1rem;margin-bottom:0.5rem;">'
    '<span style="font-size:2.2rem;">🩺</span>'
    "<div>"
    f'<h1 style="margin:0;font-family:monospace;font-size:1.8rem;color:{TEXT};">'
    "Breast Tumor AI</h1>"
    f'<p style="margin:0;color:{MUTED};font-family:monospace;font-size:0.85rem;">'
    "DCE-MRI Pipeline &middot; Segmentation &rarr; Classification</p>"
    "</div>"
    '<div style="margin-left:auto;text-align:right;">'
    f'<a href="{GITHUB_REPO}" target="_blank" '
    f'style="color:{BLUE};font-family:monospace;font-size:0.8rem;text-decoration:none;">'
    "GitHub &#8599;</a>&nbsp;&nbsp;"
    f'<a href="{HF_REPO}" target="_blank" '
    f'style="color:{AMBER};font-family:monospace;font-size:0.8rem;text-decoration:none;">'
    "Models &#8599;</a>"
    "</div></div>"
    f'<div style="background:{AMBER}22;border:1px solid {AMBER}44;'
    f'border-radius:8px;padding:0.5rem 1rem;margin-bottom:1.5rem;">'
    f'<span style="color:{AMBER};font-size:0.8rem;font-family:monospace;">'
    "&#9888;&#65039; For research demonstration only &middot; Not for clinical diagnosis"
    "</span></div>"
)
st.markdown(_HEADER, unsafe_allow_html=True)

tab_run, tab_about = st.tabs(["Run Pipeline", "About the Project"])


# ═══════════════════════════════════════════════════════════════════════════════
#  TAB 1 — RUN PIPELINE
# ═══════════════════════════════════════════════════════════════════════════════
with tab_run:

    _steps     = ["Get Sample Data", "Load Models", "Upload Patient", "View Results"]
    _step_cols = st.columns(4)
    for _i, (_col, _label) in enumerate(zip(_step_cols, _steps)):
        _active = (_i + 1 == st.session_state.current_step)
        _done   = (_i + 1 < st.session_state.current_step)
        _bg     = BLUE if _active else (GREEN + "33" if _done else CARD)
        _border = BLUE if _active else (GREEN if _done else BORDER)
        _tc     = "white" if _active else (GREEN if _done else MUTED)
        _icon   = "&#10003;" if _done else str(_i + 1)
        _col.markdown(
            f'<div style="background:{_bg};border:1px solid {_border};'
            f'border-radius:8px;padding:0.6rem;text-align:center;">'
            f'<div style="color:{_tc};font-family:monospace;font-weight:bold;font-size:0.75rem;">'
            f"{_icon}. {_label}</div></div>",
            unsafe_allow_html=True,
        )
    st.markdown("<br>", unsafe_allow_html=True)

    # ── STEP 1 ────────────────────────────────────────────────────────────────
    with st.expander(
        "Step 1 — Get Sample Patient Data",
        expanded=(st.session_state.current_step == 1),
    ):
        _S1 = (
            '<div class="step-card">'
            '<div class="step-number">1</div>'
            '<div class="step-content">'
            '<div class="step-title">Open the GitHub sample dataset</div>'
            '<div class="step-desc">'
            f'<a href="{GITHUB_DATA}" target="_blank" '
            f'style="color:{BLUE};font-family:monospace;">{GITHUB_DATA} &#8599;</a>'
            "</div></div></div>"

            '<div class="step-card">'
            '<div class="step-number">2</div>'
            '<div class="step-content">'
            '<div class="step-title">Choose a sample patient folder</div>'
            '<div class="step-desc">'
            f'&#128193; <b style="color:{RED};">positive/67</b> — Tumor present '
            f'<span style="color:{MUTED};">(Best segmentation result &middot; Dice 0.9582)</span><br>'
            f'&#128193; <b style="color:{GREEN};">negative/234</b> — No tumor (normal patient)'
            "</div></div></div>"

            '<div class="step-card">'
            '<div class="step-number">3</div>'
            '<div class="step-content">'
            '<div class="step-title">Download the .npy files</div>'
            '<div class="step-desc">'
            "Inside the patient folder, download:<br>"
            "&nbsp;&nbsp;&#8226; <b>image.npy</b> &mdash; 3-channel DCE-MRI volume (required)<br>"
            "&nbsp;&nbsp;&#8226; <b>label.npy</b> &mdash; ground truth mask "
            "(optional &middot; enables Dice score + GT overlay)<br><br>"
            "Click the file &rarr; click the <b>Download raw file</b> button on GitHub."
            "</div></div></div>"

            '<div class="step-card">'
            '<div class="step-number">4</div>'
            '<div class="step-content">'
            '<div class="step-title">Come back here and go to Step 2</div>'
            '<div class="step-desc">Once downloaded, click below to continue.</div>'
            "</div></div>"
        )
        st.markdown(_S1, unsafe_allow_html=True)

        if st.button("I have the files — Next", key="step1_next"):
            st.session_state.current_step = 2
            st.rerun()

    # ── STEP 2 ────────────────────────────────────────────────────────────────
    with st.expander(
        "Step 2 — Load AI Models",
        expanded=(st.session_state.current_step == 2),
    ):
        _S2 = (
            '<div class="step-card">'
            f'<div class="step-number" style="background:{PURPLE};">&#8595;</div>'
            '<div class="step-content">'
            '<div class="step-title">Download models from Hugging Face Hub</div>'
            '<div class="step-desc">'
            f'Stored at: <a href="{HF_REPO}" target="_blank" '
            f'style="color:{AMBER};font-family:monospace;">{HF_REPO} &#8599;</a><br>'
            "Downloads once &mdash; cached locally after. Total size ~168 MB."
            "</div></div></div>"
        )
        st.markdown(_S2, unsafe_allow_html=True)

        _model_info = [
            ("seg_monai", "Segmentation — MONAI UNet3D",      "~50 MB",  GREEN),
            ("seg_dyn",   "Segmentation — DynUNet",            "~64 MB",  GREEN),
            ("stage3",    "Classification — EfficientNet-B0",  "~54 MB",  PURPLE),
        ]

        if not st.session_state.models_loaded:
            if st.button("Load All Models", type="primary", key="load_btn"):
                for _key, _name, _size, _color in _model_info:
                    st.markdown(
                        f'<div style="color:{MUTED};font-family:monospace;font-size:0.85rem;">'
                        f"{_name} ({_size})</div>",
                        unsafe_allow_html=True,
                    )
                try:
                    with st.spinner("Downloading and loading models..."):
                        _models = load_all_models()
                    st.session_state.models_loaded = True
                    _errs = [
                        k for k in ["seg_monai", "seg_dyn", "stage3"]
                        if (f"{k}_err" in _models
                            or ("seg_err" in _models and k.startswith("seg")))
                    ]
                    if _errs:
                        st.warning(f"Some models failed to load: {_errs}")
                    else:
                        st.success("All 3 models loaded successfully!")
                except Exception as _exc:
                    st.error(f"Loading failed: {_exc}")
        else:
            for _, _name, _, _ in _model_info:
                st.markdown(
                    f'<span style="color:{GREEN};font-family:monospace;font-size:0.85rem;">'
                    f"&#10003; {_name}</span>",
                    unsafe_allow_html=True,
                )
            st.success("Models are ready.")

        if st.session_state.models_loaded:
            if st.button("Next", key="step2_next"):
                st.session_state.current_step = 3
                st.rerun()

    # ── STEP 3 ────────────────────────────────────────────────────────────────
    with st.expander(
        "Step 3 — Upload Patient File",
        expanded=(st.session_state.current_step == 3),
    ):
        _col_img, _col_mask = st.columns(2)
        with _col_img:
            img_file = st.file_uploader("image.npy — required", type=["npy"], key="img_up")
        with _col_mask:
            mask_file = st.file_uploader(
                "label.npy — optional (GT mask)", type=["npy"], key="mask_up"
            )

        if img_file is not None:
            img_np = np.load(img_file)
            if img_np.ndim == 3:
                img_np = img_np[np.newaxis]

            if img_np.ndim != 4 or img_np.shape[0] != 3:
                st.error(
                    f"Expected shape (3,D,H,W) but got {img_np.shape}. "
                    "Please upload image.npy, not label.npy."
                )
            else:
                D = img_np.shape[1]
                H = img_np.shape[2]
                W = img_np.shape[3]

                mask_np = None
                if mask_file is not None:
                    _raw    = np.load(mask_file).squeeze()
                    mask_np = _raw if _raw.ndim == 3 else _raw[np.newaxis].squeeze()

                _gt_txt = (
                    f'&nbsp;&middot;&nbsp;<span style="color:{GREEN};">GT mask loaded</span>'
                    if mask_np is not None
                    else ""
                )
                st.markdown(
                    f'<div style="background:{CARD};border:1px solid {BORDER};'
                    f'border-radius:8px;padding:0.8rem 1rem;'
                    f'font-family:monospace;font-size:0.85rem;">'
                    f'<span style="color:{GREEN};">&#10003;</span> Volume loaded'
                    f'&nbsp;&middot;&nbsp;Shape: <span style="color:{BLUE};">'
                    f"{D}x{H}x{W}</span> voxels"
                    f'&nbsp;&middot;&nbsp;Size: <span style="color:{BLUE};">'
                    f"{D*1.5:.0f}x{H*1.5:.0f}x{W*1.5:.0f} mm</span>"
                    f"{_gt_txt}</div>",
                    unsafe_allow_html=True,
                )

                _sl_prev = st.slider("Preview slice", 0, D - 1, D // 2, key="prev_sl")
                _fig_prev, _axes_prev = plt.subplots(
                    1, 3, figsize=(12, 4), facecolor=BG,
                    gridspec_kw={"wspace": 0.04},
                )
                _ch_names_p = ["P1 pre-contrast", "P2 peak enhancement", "P3 delayed"]
                _cmaps_p    = ["gray", "gray", "bone"]
                for _ch in range(3):
                    _ax = _axes_prev[_ch]
                    _ax.set_facecolor("#000")
                    _slc    = img_np[_ch, _sl_prev]
                    _lo, _hi = (
                        np.percentile(_slc, [1, 99]) if np.ptp(_slc) > 0 else (0, 1)
                    )
                    _ax.imshow(_slc, cmap=_cmaps_p[_ch], vmin=_lo, vmax=_hi, aspect="equal")
                    if mask_np is not None and mask_np[_sl_prev].sum() > 0:
                        _ax.contour(
                            mask_np[_sl_prev], levels=[0.5], colors=[GT_COL], linewidths=2
                        )
                    _ax.set_title(_ch_names_p[_ch], color=TEXT, fontsize=9)
                    _ax.axis("off")
                plt.tight_layout()
                st.pyplot(_fig_prev, use_container_width=True)
                plt.close(_fig_prev)

                if not st.session_state.models_loaded:
                    st.warning("Models not loaded. Go back to Step 2.")
                else:
                    if st.button("Run Full Pipeline", type="primary", key="run_btn"):
                        _models    = load_all_models()
                        img_tensor = torch.tensor(img_np).float().unsqueeze(0)
                        results: dict = {}

                        # Segmentation
                        _bar = st.progress(0, text="Segmentation — running sliding window...")
                        if "seg_monai" in _models and "seg_dyn" in _models:
                            pred_mask, _blend = run_segmentation(_models, img_tensor)
                            results["pred_mask"] = pred_mask
                            results["blend"]     = _blend
                            _tvox = int(pred_mask.sum())
                            st.markdown(
                                f'<span style="color:{GREEN};font-family:monospace;">'
                                f"Segmentation done &mdash; {_tvox} tumor voxels "
                                f"&middot; {_tvox * 1.5**3:.0f} mm3</span>",
                                unsafe_allow_html=True,
                            )
                            if mask_np is not None:
                                _dice = compute_dice(pred_mask, mask_np)
                                results["dice"] = _dice
                                _dc = GREEN if _dice >= 0.7 else (AMBER if _dice >= 0.4 else RED)
                                st.markdown(
                                    f'<span style="color:{_dc};font-family:monospace;">'
                                    f"Dice vs GT: {_dice}</span>",
                                    unsafe_allow_html=True,
                                )
                        else:
                            st.warning("Segmentation models unavailable.")
                            pred_mask = None
                        _bar.progress(60, text="Segmentation done")

                        # Classification
                        if "stage3" in _models:
                            _bar.progress(65, text="Classification — Benign vs Malignant...")
                            _use_mask = (
                                pred_mask if pred_mask is not None
                                else (mask_np if mask_np is not None
                                      else np.ones(img_np.shape[1:], dtype=np.uint8))
                            )
                            _prob_s3 = run_classification(_models, img_np, _use_mask)
                            results["prob_s3"] = _prob_s3
                            _is_mal  = _prob_s3 >= 0.5
                            _c3      = RED if _is_mal else AMBER
                            _lbl     = "MALIGNANT" if _is_mal else "BENIGN"
                            st.markdown(
                                f'<span style="color:{_c3};font-family:monospace;'
                                f'font-weight:bold;">{_lbl} &mdash; '
                                f"P(malignant) = {_prob_s3:.4f}</span>",
                                unsafe_allow_html=True,
                            )
                        _bar.progress(85, text="Classification done")

                        # Features
                        _use_m = pred_mask if pred_mask is not None else mask_np
                        if _use_m is not None:
                            _feats = compute_features(img_np, _use_m)
                            if _feats and "dice" in results:
                                _feats["_dice"] = results["dice"]
                            results["feats"] = _feats

                        results["img_np"]  = img_np
                        results["mask_np"] = mask_np
                        _bar.progress(100, text="Pipeline complete")

                        st.session_state.pipeline_results = results
                        st.session_state.current_step = 4
                        st.success("Pipeline complete — see results below!")
                        st.rerun()

    # ── STEP 4 — RESULTS ──────────────────────────────────────────────────────
    with st.expander(
        "Step 4 — Results",
        expanded=(st.session_state.current_step == 4),
    ):
        res = st.session_state.pipeline_results
        if res is None:
            st.info("Run the pipeline in Step 3 to see results here.")
        else:
            _img    = res["img_np"]
            _mask   = res.get("mask_np")
            _pred   = res.get("pred_mask")
            _feats  = res.get("feats") or {}

            _lbl_3d  = (_mask[0] if _mask is not None and _mask.ndim == 4 else _mask)
            _pred_3d = (_pred    if _pred  is not None and _pred.ndim  == 3 else None)

            # Summary cards
            st.markdown(stage_header("Results Summary", ""), unsafe_allow_html=True)
            _mc = st.columns(3)

            if _pred is not None:
                _mc[0].markdown(
                    result_card(
                        "Tumor Volume",
                        f"{int(_pred.sum()) * 1.5**3:.0f}",
                        "mm3 (predicted)",
                        BLUE,
                    ),
                    unsafe_allow_html=True,
                )
            if "dice" in res:
                _d = res["dice"]
                _mc[1].markdown(
                    result_card(
                        "Segmentation Dice",
                        str(_d),
                        "vs Ground Truth",
                        GREEN if _d >= 0.7 else (AMBER if _d >= 0.4 else RED),
                    ),
                    unsafe_allow_html=True,
                )
            if "prob_s3" in res:
                _p = res["prob_s3"]
                _mc[2].markdown(
                    result_card(
                        "Classification",
                        f"{_p * 100:.1f}%",
                        "Malignant" if _p >= 0.5 else "Benign",
                        RED if _p >= 0.5 else AMBER,
                    ),
                    unsafe_allow_html=True,
                )

            # Gauge
            if "prob_s3" in res:
                st.markdown(
                    stage_header("Classification Probability", ""),
                    unsafe_allow_html=True,
                )
                _gc1, _gc2, _gc3 = st.columns([1, 2, 1])
                with _gc2:
                    st.plotly_chart(
                        prob_gauge(
                            res["prob_s3"],
                            positive_label="MALIGNANT",
                            negative_label="BENIGN",
                            pos_color=RED,
                            neg_color=AMBER,
                        ),
                        use_container_width=True,
                    )
                    st.markdown(
                        f'<p style="text-align:center;color:{MUTED};'
                        f'font-family:monospace;font-size:0.8rem;">'
                        "Benign vs Malignant &mdash; 3D EfficientNet-B0</p>",
                        unsafe_allow_html=True,
                    )

            # Kinetic type
            if _feats.get("kinetic_type"):
                _kc = st.columns([1, 2, 1])
                with _kc[1]:
                    _kt = _feats["kinetic_type"].split("—")[0].strip()
                    st.markdown(
                        result_card(
                            "DCE Kinetic Type",
                            _kt,
                            _feats["kinetic_type"],
                            _feats.get("kinetic_color", TEXT),
                        ),
                        unsafe_allow_html=True,
                    )

            # VIS A
            st.markdown(
                stage_header("Segmentation — Slice Overlay", ""),
                unsafe_allow_html=True,
            )
            _n_sl = st.slider("Number of slices to show", 3, 7, 5, key="n_sl")
            _pid  = (
                img_file.name.replace(".npy", "")
                if "img_file" in dir() and img_file is not None
                else "patient"
            )
            _fig_a = vis_slice_overlay(_img, _lbl_3d, _pred_3d, _feats, pid=_pid, n_slices=_n_sl)
            st.pyplot(_fig_a, use_container_width=True)
            plt.close(_fig_a)
            st.markdown(
                f'<p style="color:{MUTED};font-family:monospace;font-size:0.78rem;">'
                f'<span style="color:lime;">&#9632;</span> Green = GT mask &nbsp;&middot;&nbsp;'
                f'<span style="color:{RED};">&#9632;</span> Red = Predicted mask</p>',
                unsafe_allow_html=True,
            )

            # VIS B
            if _lbl_3d is not None:
                st.markdown(
                    stage_header("All 3 DCE Channels with GT Contour", ""),
                    unsafe_allow_html=True,
                )
                _fig_b = vis_all_channels(_img, _lbl_3d, n_slices=_n_sl)
                st.pyplot(_fig_b, use_container_width=True)
                plt.close(_fig_b)

            # VIS C
            st.markdown(
                stage_header("Max-Intensity Projection — 3 Axes", ""),
                unsafe_allow_html=True,
            )
            _fig_c = vis_mip(_img, _lbl_3d, _pred_3d, feats=_feats)
            st.pyplot(_fig_c, use_container_width=True)
            plt.close(_fig_c)

            # VIS E
            if _feats and _lbl_3d is not None:
                st.markdown(
                    stage_header("Tumour Feature Report", ""),
                    unsafe_allow_html=True,
                )
                _fig_e = vis_feature_panel(_img, _lbl_3d, _feats)
                st.pyplot(_fig_e, use_container_width=True)
                plt.close(_fig_e)

            st.markdown("<br>", unsafe_allow_html=True)
            if st.button("Run another patient", key="reset_btn"):
                st.session_state.pipeline_results = None
                st.session_state.current_step = 3
                st.rerun()


# ═══════════════════════════════════════════════════════════════════════════════
#  TAB 2 — ABOUT
# ═══════════════════════════════════════════════════════════════════════════════
with tab_about:
    st.markdown(
        f'<h2 style="font-family:monospace;">About the Project</h2>',
        unsafe_allow_html=True,
    )

    _ab_col_a, _ab_col_b = st.columns(2)
    _ABOUT_CARDS = [
        (
            _ab_col_a, "Segmentation",
            "MONAI UNet3D + DynUNet\nEnsemble: 0.35 / 0.65\nSliding window 96^3",
            "Mean Dice = 0.80  |  Best = 0.9582",
            GREEN,
        ),
        (
            _ab_col_b, "Classification",
            "3D EfficientNet-B0\n4.7M parameters\nInput: 64^3 centroid crop",
            "AUC = 0.9200",
            PURPLE,
        ),
    ]
    for _col, _stage, _model, _metric, _color in _ABOUT_CARDS:
        _col.markdown(
            f'<div style="background:{CARD};border:1px solid {_color}44;'
            f'border-radius:10px;padding:1.2rem;text-align:center;">'
            f'<div style="color:{_color};font-family:monospace;'
            f'font-weight:bold;margin:0.5rem 0;">{_stage}</div>'
            f'<div style="color:{MUTED};font-family:monospace;font-size:0.78rem;'
            f'white-space:pre-line;margin-bottom:0.8rem;">{_model}</div>'
            f'<div style="color:{_color};font-family:monospace;'
            f'font-weight:bold;font-size:1.1rem;">{_metric}</div>'
            "</div>",
            unsafe_allow_html=True,
        )

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("### DCE-MRI — What the 3 channels mean")
    st.markdown(
        "| Channel | Phase | Clinical significance |\n"
        "|---|---|---|\n"
        "| **P1** | Pre-contrast | Baseline tissue, no contrast yet |\n"
        "| **P2** | Peak enhancement | Malignant tumors enhance aggressively |\n"
        "| **P3** | Delayed (washout) | Malignant tumors wash out fast, benign persist |"
    )
    st.markdown(
        "The P2 to P3 relationship defines the **kinetic type** "
        "(Type I Persistent / Type II Plateau / Type III Washout) "
        "— the radiological basis of BI-RADS classification."
    )

    st.divider()
    _lc1, _lc2 = st.columns(2)
    _lc1.markdown(
        f'<div style="background:{CARD};border:1px solid {BORDER};'
        f'border-radius:8px;padding:1rem;">'
        f'<div style="color:{TEXT};font-family:monospace;font-weight:bold;margin-bottom:0.5rem;">'
        "GitHub Repository</div>"
        f'<a href="{GITHUB_REPO}" target="_blank" '
        f'style="color:{BLUE};font-family:monospace;font-size:0.85rem;">'
        f"{GITHUB_REPO} &#8599;</a><br>"
        f'<div style="color:{MUTED};font-size:0.8rem;margin-top:0.5rem;">'
        "Source code &middot; Sample dataset &middot; README</div>"
        "</div>",
        unsafe_allow_html=True,
    )
    _lc2.markdown(
        f'<div style="background:{CARD};border:1px solid {BORDER};'
        f'border-radius:8px;padding:1rem;">'
        f'<div style="color:{TEXT};font-family:monospace;font-weight:bold;margin-bottom:0.5rem;">'
        "Model Weights (Hugging Face)</div>"
        f'<a href="{HF_REPO}" target="_blank" '
        f'style="color:{AMBER};font-family:monospace;font-size:0.85rem;">'
        f"{HF_REPO} &#8599;</a><br>"
        f'<div style="color:{MUTED};font-size:0.8rem;margin-top:0.5rem;">'
        "MONAI UNet3D &middot; DynUNet &middot; EfficientNet-B0</div>"
        "</div>",
        unsafe_allow_html=True,
    )

# ── Footer ─────────────────────────────────────────────────────────────────────
st.markdown(
    f'<div style="text-align:center;color:{MUTED};font-family:monospace;'
    f'font-size:0.72rem;padding:1.5rem 0 0.5rem 0;'
    f'border-top:1px solid {BORDER};margin-top:2rem;">'
    "Breast Tumor AI &middot; Research Demo &middot; Not for clinical use<br>"
    "MONAI UNet3D &middot; DynUNet &middot; EfficientNet-B0 &middot; DCE-MRI"
    "</div>",
    unsafe_allow_html=True,
)
