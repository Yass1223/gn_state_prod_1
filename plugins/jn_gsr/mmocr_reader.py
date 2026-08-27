"""mmocr_reader.py -- second recogniser (mmocr 1.x checkpoint) behind the same
read_many([crops]) -> [(tens_logl[11], units_logl[11])] contract as
common.build_parseq_batch_reader, so the worker caches both models' outputs
per frame and every consolidation rule reads them identically.

Contract (must match the PARSeq reader exactly, or maxconf's two magnitudes
are not comparable across models):
  * columns: the model's dictionary indices of '0'..'9', then its END index;
  * positions: decoder steps 0 and 1 (tens, units);
  * values: log(p + 1e-9) where p is the model's per-step probability over its
    FULL dictionary restricted to those 11 columns -- i.e. NOT renormalised
    over the eleven. evaluate_jn renormalises for c(f) and uses the raw value
    for s(f), exactly as it does for PARSeq.

Config resolution, in order: an explicit --recog2-cfg; a .py shipped next to
the checkpoint; the config text mmengine stores inside the checkpoint itself
(message_hub.runtime_info.cfg, else meta.cfg). The last is the usual case
here: the published download is a bare torch container.

Dictionary resolution: an explicit --recog2-dict; a .txt shipped next to the
checkpoint; the path the config names, if it exists on this machine; else the
same basename inside the installed mmocr package (mmocr ships its dicts under
mmocr/.mim/dicts/). The training machine's absolute path is what a recovered
config carries, so that last step is what normally resolves it. Whatever is
chosen, the resulting dictionary size is checked against the checkpoint's own
classifier output dimension -- a wrong dictionary would otherwise mis-index
every column silently.
"""
import glob
import os

import numpy as np

DIGITS = "0123456789"


def find_files(path):
    """(pth, cfg_or_None, dict_txt_or_None) from a directory, or a file path."""
    if os.path.isfile(path):
        d = os.path.dirname(path) or "."
        pth = path
    else:
        d = path
        pths = sorted(glob.glob(os.path.join(d, "**", "*.pth"), recursive=True)
                      + glob.glob(os.path.join(d, "**", "*.ckpt"), recursive=True))
        if not pths:
            raise FileNotFoundError(f"no .pth/.ckpt under {d}")
        if len(pths) > 1:
            print(f"[recog2] {len(pths)} checkpoints under {d}; using {pths[0]}")
        pth = pths[0]
    cfgs = sorted(glob.glob(os.path.join(d, "**", "*.py"), recursive=True))
    # a config shipped in the download beats one recovered from the checkpoint
    cfgs.sort(key=lambda p: os.path.basename(p) == "config_from_checkpoint.py")
    txts = sorted(glob.glob(os.path.join(d, "**", "*.txt"), recursive=True))
    return pth, (cfgs[0] if cfgs else None), (txts[0] if txts else None)


def config_from_checkpoint(pth, out_path):
    """Write the config text mmengine stored in the checkpoint. -> path | None."""
    import torch
    raw = torch.load(pth, map_location="cpu")
    text = None
    if isinstance(raw, dict):
        try:
            text = raw["message_hub"]["runtime_info"]["cfg"]
        except Exception:
            text = (raw.get("meta") or {}).get("cfg")
    del raw
    if not isinstance(text, str) or not text.strip():
        return None
    with open(out_path, "w") as f:
        f.write(text)
    return out_path


def classifier_out_dim(pth):
    """Output dimension of the checkpoint's last classifier layer, or None.
    This is the number of classes the weights were trained with."""
    import torch
    raw = torch.load(pth, map_location="cpu")
    sd = raw.get("state_dict", raw) if isinstance(raw, dict) else None
    if not isinstance(sd, dict):
        return None
    cands = [(k, tuple(v.shape)) for k, v in sd.items()
             if hasattr(v, "shape") and getattr(v, "ndim", 0) == 2
             and k.endswith("weight")
             and any(s in k for s in ("classifier", "cls", "head", "fc",
                                      "generator", "prediction"))]
    del raw
    if not cands:
        return None
    # the recognition head is the last such layer in state-dict order
    return int(cands[-1][1][0])


def mmocr_dict_dirs():
    """Directories where the installed mmocr keeps its character dicts."""
    out = []
    try:
        import mmocr
        base = os.path.dirname(mmocr.__file__)
        out += [os.path.join(base, ".mim", "dicts"), os.path.join(base, "dicts"),
                os.path.join(os.path.dirname(base), "dicts")]
    except Exception:
        pass
    return [d for d in out if os.path.isdir(d)]


def resolve_dict_file(named, override=None, shipped=None):
    """The dictionary file to actually use, and a note about where it came
    from. `named` is what the config asks for (possibly a path that only
    existed on the training machine)."""
    if override:
        if not os.path.exists(override):
            raise FileNotFoundError(f"--recog2-dict {override} does not exist")
        return override, "explicit override"
    if shipped:
        return shipped, "shipped next to the checkpoint"
    if named and os.path.exists(named):
        return named, "the path in the config"
    if named:
        base = os.path.basename(named)
        for d in mmocr_dict_dirs():
            cand = os.path.join(d, base)
            if os.path.exists(cand):
                return cand, f"basename match inside mmocr ({d})"
    return None, ("unresolved -- pass --recog2-dict with the character file "
                  "this checkpoint was trained on")


def _patch_dict_file(cfg, dict_file):
    """Point every `dict_file` in the config at `dict_file`; -> how many."""
    n = 0

    def walk(node):
        nonlocal n
        if isinstance(node, dict):
            if "dict_file" in node:
                node["dict_file"] = dict_file
                n += 1
            for v in node.values():
                walk(v)
        elif isinstance(node, (list, tuple)):
            for v in node:
                walk(v)
    walk(cfg._cfg_dict)
    return n


def _named_dict_file(cfg):
    """The first dict_file the config names, or None."""
    found = []

    def walk(node):
        if isinstance(node, dict):
            if isinstance(node.get("dict_file"), str):
                found.append(node["dict_file"])
            for v in node.values():
                walk(v)
        elif isinstance(node, (list, tuple)):
            for v in node:
                walk(v)
    walk(cfg._cfg_dict)
    return found[0] if found else None


class MMOCRRecogniser:
    def __init__(self, path, config=None, dict_file=None, device=None,
                 batch=64):
        os.environ["MPLBACKEND"] = "Agg"
        try:
            import crop_classifier as CC
            CC.numpy2_pickle_compat()
        except Exception:
            pass
        import torch
        from mmengine.config import Config
        from mmocr.apis import TextRecInferencer

        pth, cfg_found, txt_found = find_files(path)
        self.pth = pth
        self.device = device or ("cuda:0" if torch.cuda.is_available() else "cpu")
        self.batch = batch

        cfg_path = config or cfg_found
        if not cfg_path:
            out = os.path.join(os.path.dirname(pth) or ".",
                               "config_from_checkpoint.py")
            cfg_path = config_from_checkpoint(pth, out)
            if not cfg_path:
                raise RuntimeError(
                    f"[recog2] {pth} ships no config and carries none inside "
                    f"it. Pass --recog2-cfg with the mmocr config this "
                    f"checkpoint was trained with.")
            print(f"[recog2] config recovered from the checkpoint -> {cfg_path}")
        self.cfg_path = cfg_path

        cfg = Config.fromfile(cfg_path)
        named = _named_dict_file(cfg)
        chosen, why = resolve_dict_file(named, dict_file, txt_found)
        if chosen is None:
            raise RuntimeError(
                f"[recog2] the config asks for dictionary {named!r}, which is "
                f"not on this machine and has no match in "
                f"{mmocr_dict_dirs() or 'the installed mmocr'}. Pass "
                f"--recog2-dict.")
        k = _patch_dict_file(cfg, chosen)
        print(f"[recog2] dictionary -> {chosen} ({why}; {k} config entries; "
              f"config asked for {named})")
        self.dict_file = chosen

        self.inf = TextRecInferencer(model=cfg, weights=pth, device=self.device)
        self.model = self.inf.model.eval()

        dec = getattr(self.model, "decoder", None)
        dictionary = getattr(dec, "dictionary", None) or \
            getattr(self.model, "dictionary", None)
        if dictionary is None:
            raise RuntimeError("[recog2] model exposes no decoder.dictionary; "
                               "cannot map columns to characters")
        chars = list(dictionary.dict)
        missing = [c for c in DIGITS if c not in chars]
        if missing:
            raise RuntimeError(
                f"[recog2] the dictionary in use lacks the digits {missing} "
                f"({self.dict_file}) -- wrong character file for this "
                f"checkpoint. Pass --recog2-dict.")
        self.digit_idx = [chars.index(c) for c in DIGITS]
        self.end_idx = int(dictionary.end_idx)
        if self.end_idx in self.digit_idx or self.end_idx < 0:
            raise RuntimeError(f"[recog2] bad END index {self.end_idx}")
        self.cols = self.digit_idx + [self.end_idx]
        self.n_classes = int(dictionary.num_classes)

        # A dictionary of the wrong SIZE loads happily and mis-indexes every
        # column, which would silently corrupt every t2/u2. Check it against
        # what the weights themselves were trained with.
        want = classifier_out_dim(pth)
        if want is not None and want != self.n_classes:
            raise RuntimeError(
                f"[recog2] dictionary gives {self.n_classes} classes but the "
                f"checkpoint's classifier has {want} outputs -- the character "
                f"file {self.dict_file} is not the one this model was trained "
                f"on. Pass --recog2-dict with the right file.")

        self.arch = type(self.model).__name__ + "/" + type(dec).__name__
        print(f"[recog2] loaded {self.arch}  classes={self.n_classes}  "
              f"digits -> {self.digit_idx}  END -> {self.end_idx}  "
              f"ckpt={os.path.basename(pth)}")
        self._softmax_checked = False
        self.applies_softmax = None

    # ---------------------------------------------------------------- forward
    def _scores(self, bgr_list):
        """Per-step class scores [B, T, C] for a list of BGR uint8 arrays.

        The inferencer's own test pipeline (resize / pad / normalise as the
        checkpoint was trained), its collate_fn and the model's
        data_preprocessor -- the same tensors the inferencer would forward.
        Only the last step differs: _forward returns the decoder's per-step
        scores instead of a decoded string.
        """
        import torch
        collate = getattr(self.inf, "collate_fn", None)
        if collate is None:
            from mmengine.dataset import pseudo_collate as collate
        items = [self.inf.pipeline(arr) for arr in bgr_list]
        data = collate(items)
        data = self.model.data_preprocessor(data, False)
        with torch.no_grad():
            out = self.model._forward(**data)
        if isinstance(out, (list, tuple)):
            out = out[0]
        return out.float()

    def read_many(self, crops):
        """crops: list of PIL images -> [(tens_logl[11], units_logl[11])]."""
        import torch
        out = []
        for i in range(0, len(crops), self.batch):
            chunk = crops[i:i + self.batch]
            arrs = [np.ascontiguousarray(np.asarray(c.convert("RGB"))[:, :, ::-1])
                    for c in chunk]
            sc = self._scores(arrs)                       # [B, T, C]
            if sc.ndim != 3 or sc.shape[1] < 2:
                raise RuntimeError(f"[recog2] decoder returned shape "
                                   f"{tuple(sc.shape)}; need [B, >=2, C]")
            if sc.shape[2] != self.n_classes:
                raise RuntimeError(f"[recog2] decoder returned "
                                   f"{sc.shape[2]} classes, dictionary has "
                                   f"{self.n_classes}")
            # mmocr decoders apply softmax in forward_test; be safe either way.
            if not self._softmax_checked:
                self._softmax_checked = True
                rs = sc[:, :2, :].sum(-1)
                self.applies_softmax = bool(
                    torch.allclose(rs, torch.ones_like(rs), atol=1e-3))
                print(f"[recog2] decoder output "
                      f"{'is' if self.applies_softmax else 'is not'} a "
                      f"probability simplex; "
                      f"{'used as is' if self.applies_softmax else 'softmax applied'}")
            probs = sc if self.applies_softmax else sc.softmax(-1)
            m = probs[:, :2, :][:, :, self.cols].double().cpu().numpy()  # [B,2,11]
            logl = np.log(m + 1e-9)
            out.extend((logl[b, 0], logl[b, 1]) for b in range(len(logl)))
        return out


def build_recog2_reader(path, config=None, dict_file=None, device=None):
    r = MMOCRRecogniser(path, config=config, dict_file=dict_file, device=device)
    return r, r.read_many


if __name__ == "__main__":
    # pure-logic tests: no mmocr, no torch, no GPU
    import tempfile

    class _Cfg:
        def __init__(self, d):
            self._cfg_dict = d

    c = _Cfg({"model": {"decoder": {"dictionary": {"dict_file": "/train/a.txt"}}},
              "dictionary": {"dict_file": "/train/a.txt"},
              "x": [{"dict_file": "/train/a.txt"}]})
    assert _named_dict_file(c) == "/train/a.txt"
    assert _patch_dict_file(c, "new.txt") == 3
    assert c._cfg_dict["model"]["decoder"]["dictionary"]["dict_file"] == "new.txt"
    assert _named_dict_file(_Cfg({"a": 1})) is None

    with tempfile.TemporaryDirectory() as d:
        os.makedirs(os.path.join(d, "sub"))
        for f in ("sub/a.pth", "sub/config_from_checkpoint.py", "sub/dict.txt"):
            open(os.path.join(d, f), "w").close()
        p, cfg, txt = find_files(d)
        assert p.endswith("a.pth") and txt.endswith("dict.txt")
        assert cfg.endswith("config_from_checkpoint.py")
        # a config shipped in the download outranks the recovered one
        open(os.path.join(d, "sub", "satrn.py"), "w").close()
        _, cfg2, _ = find_files(d)
        assert cfg2.endswith("satrn.py"), cfg2
        p2, _, _ = find_files(p)
        assert p2 == p

        # dictionary resolution order
        real = os.path.join(d, "real.txt")
        open(real, "w").close()
        assert resolve_dict_file("/train/x.txt", override=real)[0] == real
        assert resolve_dict_file("/train/x.txt", shipped=real)[0] == real
        assert resolve_dict_file(real)[0] == real
        assert resolve_dict_file("/train/definitely_absent_9f2.txt")[0] is None
        try:
            resolve_dict_file("x", override=os.path.join(d, "nope.txt"))
            raise AssertionError("missing override should raise")
        except FileNotFoundError:
            pass
    print("mmocr_reader self-tests OK")
