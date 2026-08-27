"""Runtime Hugging Face weight fetching via an OmegaConf resolver.

Registers a `${hf:<repo_id>,<filename>[,<local_name>]}` resolver so model checkpoints
are pulled from the Hub at config-resolution time (cached on disk by huggingface_hub) and
the resolved value is the local path that the module loaders expect.

Why the optional third argument matters
---------------------------------------
tracklab's BoT-SORT ReID backend (`ReIDDetectMultiBackend`) infers the network
architecture from the weights *filename* (`get_model_name`) and only loads weights when
the suffix is `.pt`. Our SoccerNet sports ReID checkpoint is a torchreid `osnet_x1_0`
saved as `sports_model.pth.tar-60` — that name matches no architecture and the suffix
isn't `.pt`, so feeding it raw would silently leave the ReID model randomly initialised.
Passing a third argument materialises a copy named e.g. `osnet_x1_0_sports.pt`, so the
backend builds `osnet_x1_0` and torchreid's `load_pretrained_weights` (which strips
`module.` and shape-filters the classifier) loads it correctly — identical to the notebook.

This module is imported from `config_finder.py` (loaded at plugin discovery), guaranteeing
the resolver exists before any config interpolation is evaluated.
"""
import logging
import shutil
from pathlib import Path

from omegaconf import OmegaConf

log = logging.getLogger(__name__)


def hf_path(repo_id: str, filename: str, local_name: str = None) -> str:
    """Download `filename` from `repo_id` and return a local path.

    If `local_name` is given, the downloaded file is copied (once) to a sibling file with
    that name and the copy's path is returned — used to give checkpoints a loader-friendly
    name/suffix without mutating the immutable huggingface_hub cache entry.
    """
    from huggingface_hub import hf_hub_download  # local import keeps module import cheap

    repo_id = str(repo_id).strip()
    filename = str(filename).strip()
    src = Path(hf_hub_download(repo_id=repo_id, filename=filename))

    if local_name:
        dst = src.parent / str(local_name).strip()
        if not dst.exists():
            shutil.copy(src, dst)
        log.info(f"[hf] {repo_id}/{filename} -> {dst}")
        return str(dst)

    log.info(f"[hf] {repo_id}/{filename} -> {src}")
    return str(src)


# Register once. `use_cache=True` avoids re-downloading for repeated identical interpolations
# within a single run (huggingface_hub also caches across runs on disk).
if not OmegaConf.has_resolver("hf"):
    OmegaConf.register_new_resolver("hf", hf_path, use_cache=True)
