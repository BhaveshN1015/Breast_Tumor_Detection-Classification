"""
download_models.py
==================
Downloads the 3 model checkpoints from Hugging Face Hub
into the local models/ directory at startup.

Called once by app.py — subsequent runs use cached files.
"""

import os
from huggingface_hub import hf_hub_download

HF_REPO_ID = "B1015/breast-tumor-ai"

# Local paths where models will be saved
# key : (path_in_hf_repo, local_relative_path)
MODELS = {
    "seg_monai": (
        "models/segmentation_3d/unet3d_best_raw.pth",
        "models/segmentation_3d/unet3d_best_raw.pth",
    ),
    "seg_dyn": (
        "models/dynunet_3d/dynunet_best_raw.pth",
        "models/dynunet_3d/dynunet_best_raw.pth",
    ),
    "stage3": (
        "models/classification_stage3/best_model.pth",
        "models/classification_stage3/best_model.pth",
    ),
}


def get_model_paths():
    """Return local absolute paths for all 3 models."""
    base = os.path.dirname(os.path.abspath(__file__))
    return {
        key: os.path.join(base, local_path)
        for key, (_, local_path) in MODELS.items()
    }


def download_all_models(progress_callback=None):
    """
    Download all models from HF Hub if not already cached locally.
    progress_callback(key, status) is called for UI updates if provided.
    Returns dict of {key: local_absolute_path}.
    """
    base = os.path.dirname(os.path.abspath(__file__))
    local_paths = {}

    for key, (hf_path, local_path) in MODELS.items():
        full_local = os.path.join(base, local_path)

        if progress_callback:
            progress_callback(key, "checking")

        if os.path.isfile(full_local):
            if progress_callback:
                progress_callback(key, "cached")
            local_paths[key] = full_local
            continue

        os.makedirs(os.path.dirname(full_local), exist_ok=True)

        if progress_callback:
            progress_callback(key, "downloading")

        try:
            downloaded = hf_hub_download(
                repo_id=HF_REPO_ID,
                filename=hf_path,
                repo_type="model",
                local_dir=base,
            )
            local_paths[key] = downloaded
            if progress_callback:
                progress_callback(key, "done")
        except Exception as exc:
            if progress_callback:
                progress_callback(key, f"error: {exc}")
            raise RuntimeError(f"Failed to download {key}: {exc}") from exc

    return local_paths


def all_models_cached():
    """Quick check — returns True if all 3 model files exist locally."""
    base = os.path.dirname(os.path.abspath(__file__))
    return all(
        os.path.isfile(os.path.join(base, local_path))
        for _, local_path in MODELS.values()
    )
