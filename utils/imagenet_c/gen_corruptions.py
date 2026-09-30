#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Fast parallel generator of the ImageNet-C-style "Corruption" datasets used by CoVer.

For every source image it applies the CLIP val preprocessing (Resize(224) +
CenterCrop(224)) ONCE, then writes all 18 corruptions x 5 severities (read-once,
write-many).  Output layout matches exactly what utils/train_eval_util.py expects:

  ImageNet   :  datasets/DistortedImageNet/val_224/<corruption>/<sev>/<wnid>/<file>
  iNaturalist:  datasets/ImageNet_OOD_dataset/Distorted_CLIP/iNaturalist/<corruption>/<sev>/images/<file>
  SUN        :  datasets/ImageNet_OOD_dataset/Distorted_CLIP/SUN/<corruption>/<sev>/images/<file>
  Places     :  datasets/ImageNet_OOD_dataset/Distorted_CLIP/Places/<corruption>/<sev>/images/<file>
  dtd        :  datasets/ImageNet_OOD_dataset/Distorted_CLIP/dtd/images/<corruption>/<sev>/<category>/<file>

The 18 corruptions are exactly the set consumed by the loader (no `frost`).
glass_blur (≈63% of CPU cost) is JIT-compiled with numba when available; the
algorithm/severity constants are identical to the reference implementation
(the corruptions are stochastic & unseeded, so a fresh realization is expected
either way).

Usage:
  python gen_corruptions.py --dataset all   --workers 120
  python gen_corruptions.py --dataset ImageNet --workers 120 --limit 200   # benchmark
"""
import os, sys, time, argparse, importlib.util, traceback
import multiprocessing as mp

# Pin every numerical backend to a single thread BEFORE importing numpy/cv2/skimage:
# we get parallelism from the process Pool, so per-process BLAS/OpenMP threads would
# only oversubscribe the 128 logical cores.  Must be set before the first numpy import.
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_v, "1")
# wand needs to find libMagickWand before corruptions.py is imported
os.environ.setdefault("MAGICK_HOME", sys.prefix)

import numpy as np
from PIL import Image
import torchvision.transforms as trn

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", ".."))
DATASETS = os.path.join(REPO, "datasets")

# ---- import the (patched) corruption functions directly, bypassing utils/__init__ ----
_spec = importlib.util.spec_from_file_location("cover_corruptions", os.path.join(HERE, "corruptions.py"))
C = importlib.util.module_from_spec(_spec); _spec.loader.exec_module(C)

# exact 18 corruptions the loader iterates over (utils/train_eval_util.py)
CORRUPTIONS = ["fog", "motion_blur", "brightness", "snow", "defocus_blur", "glass_blur",
               "gaussian_noise", "shot_noise", "impulse_noise", "contrast", "elastic_transform",
               "pixelate", "jpeg_compression", "speckle_noise", "spatter", "gaussian_blur",
               "saturate", "zoom_blur"]
SEVERITIES = [1, 2, 3, 4, 5]

# ---------------------------------------------------------------- JIT glass_blur
try:
    from numba import njit
    from skimage.filters import gaussian as _sk_gaussian

    @njit(cache=True, nogil=True)
    def _glass_shuffle(x, md, iters):
        for _ in range(iters):
            for h in range(224 - md, md, -1):
                for w in range(224 - md, md, -1):
                    dx = np.random.randint(-md, md)
                    dy = np.random.randint(-md, md)
                    hp = h + dy; wp = w + dx
                    for ch in range(3):
                        t = x[h, w, ch]; x[h, w, ch] = x[hp, wp, ch]; x[hp, wp, ch] = t
        return x

    def glass_blur_jit(pil_img, severity=1):
        # sigma, max_delta, iterations  (identical constants to corruptions.glass_blur)
        c = [(0.7, 1, 2), (0.9, 2, 1), (1, 2, 3), (1.1, 3, 2), (1.5, 4, 2)][severity - 1]
        x = np.uint8(_sk_gaussian(np.asarray(pil_img) / 255., sigma=c[0], channel_axis=-1) * 255)
        x = np.ascontiguousarray(x)
        _glass_shuffle(x, c[1], c[2])
        return np.clip(_sk_gaussian(x / 255., sigma=c[0], channel_axis=-1), 0, 1) * 255
    HAVE_NUMBA = True
except Exception:                                              # pragma: no cover
    HAVE_NUMBA = False

# ---------------------------------------------------------------- dataset configs
# src is the ImageFolder root; out is the corrupted root; both keep the class subdir.
def _cfg(src_base, dst_base):
    """src paths read the original images (NFS); out paths write corruptions (may be local)."""
    src_ood = os.path.join(src_base, "ImageNet_OOD_dataset")
    dst_ood = os.path.join(dst_base, "ImageNet_OOD_dataset")
    return {
        "ImageNet":    dict(src=os.path.join(src_base, "ImageNet", "val"),
                            out=os.path.join(dst_base, "DistortedImageNet", "val_224")),
        "iNaturalist": dict(src=os.path.join(src_ood, "iNaturalist"),
                            out=os.path.join(dst_ood, "Distorted_CLIP", "iNaturalist")),
        "SUN":         dict(src=os.path.join(src_ood, "SUN"),
                            out=os.path.join(dst_ood, "Distorted_CLIP", "SUN")),
        "Places":      dict(src=os.path.join(src_ood, "Places"),
                            out=os.path.join(dst_ood, "Distorted_CLIP", "Places")),
        "dtd":         dict(src=os.path.join(src_ood, "dtd", "images"),
                            out=os.path.join(dst_ood, "Distorted_CLIP", "dtd", "images")),
    }

IMG_EXT = (".jpg", ".jpeg", ".png", ".ppm", ".bmp", ".pgm", ".JPEG", ".JPG")

def list_images(src):
    """Return list of (class_name, filename) under an ImageFolder-style root."""
    items = []
    for cls in sorted(os.listdir(src)):
        d = os.path.join(src, cls)
        if not os.path.isdir(d):
            continue
        for fn in sorted(os.listdir(d)):
            if fn.lower().endswith(IMG_EXT):
                items.append((cls, fn))
    return items

# globals set per worker
_W = {}

def _init_worker(src, out, quality, optimize, resume, use_jit):
    import warnings; warnings.simplefilter("ignore")
    try:
        C.cv2.setNumThreads(0)
    except Exception:
        pass
    disp = {}
    for name in CORRUPTIONS:
        disp[name] = getattr(C, name)
    if use_jit and HAVE_NUMBA:
        disp["glass_blur"] = glass_blur_jit
        # warm up JIT once per worker
        try:
            disp["glass_blur"](Image.fromarray(np.zeros((224, 224, 3), np.uint8)), 1)
        except Exception:
            pass
    _W.update(dict(src=src, out=out, quality=quality, optimize=optimize, resume=resume,
                   disp=disp, prep=trn.Compose([trn.Resize(224), trn.CenterCrop(224)]),
                   made=set()))

def _ensure_dir(path):
    if path not in _W["made"]:
        os.makedirs(path, exist_ok=True)
        _W["made"].add(path)

def _process_chunk(chunk):
    src, out, q, opt, resume = _W["src"], _W["out"], _W["quality"], _W["optimize"], _W["resume"]
    disp, prep = _W["disp"], _W["prep"]
    done = skipped = errs = 0
    last_corr, last_sev = CORRUPTIONS[-1], SEVERITIES[-1]
    for cls, fn in chunk:
        try:
            if resume:
                sentinel = os.path.join(out, last_corr, str(last_sev), cls, fn)
                if os.path.exists(sentinel):
                    skipped += 1; continue
            base = np.asarray(prep(Image.open(os.path.join(src, cls, fn)).convert("RGB")))
            for name in CORRUPTIONS:
                fn_corr = disp[name]
                for sev in SEVERITIES:
                    od = os.path.join(out, name, str(sev), cls)
                    _ensure_dir(od)
                    op = os.path.join(od, fn)
                    if resume and os.path.exists(op):
                        continue
                    arr = np.uint8(fn_corr(Image.fromarray(base), sev))
                    Image.fromarray(arr).save(op, quality=q, optimize=opt)
            done += 1
        except Exception:
            errs += 1
            sys.stderr.write(f"ERR {cls}/{fn}: {traceback.format_exc().splitlines()[-1]}\n")
    return (done, skipped, errs)

def chunked(lst, n):
    for i in range(0, len(lst), n):
        yield lst[i:i + n]

def run_dataset(name, cfg, args):
    src, out = cfg["src"], cfg["out"]
    assert os.path.isdir(src), f"missing source: {src}"
    items = list_images(src)
    if args.limit:
        items = items[:args.limit]
    n = len(items)
    print(f"[{time.strftime('%H:%M:%S')}] {name}: {n} source imgs -> {out}", flush=True)
    print(f"    will write up to {n*len(CORRUPTIONS)*len(SEVERITIES):,} corrupted imgs "
          f"({len(CORRUPTIONS)} corr x {len(SEVERITIES)} sev: {CORRUPTIONS} @ {SEVERITIES}) "
          f"| jit_glass={args.jit and HAVE_NUMBA}", flush=True)
    os.makedirs(out, exist_ok=True)
    chunk = max(1, args.chunk)
    chunks = list(chunked(items, chunk))
    t0 = time.time(); tot_done = tot_skip = tot_err = 0; nchunks = len(chunks)
    with mp.Pool(args.workers, initializer=_init_worker,
                 initargs=(src, out, args.quality, not args.no_optimize, not args.no_resume,
                           args.jit)) as pool:
        for i, (d, s, e) in enumerate(pool.imap_unordered(_process_chunk, chunks), 1):
            tot_done += d; tot_skip += s; tot_err += e
            if i % max(1, nchunks // 50) == 0 or i == nchunks:
                el = time.time() - t0
                rate = (tot_done + tot_skip) / el if el > 0 else 0
                eta = (n - tot_done - tot_skip) / rate if rate > 0 else 0
                print(f"    [{i}/{nchunks} chunks] done={tot_done} skip={tot_skip} err={tot_err} "
                      f"| {rate:.1f} img/s | elapsed {el/60:.1f}m eta {eta/60:.1f}m", flush=True)
    print(f"[{time.strftime('%H:%M:%S')}] {name} FINISHED: done={tot_done} skip={tot_skip} "
          f"err={tot_err} in {(time.time()-t0)/60:.1f} min", flush=True)
    return tot_err

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="all",
                    choices=["all", "ImageNet", "iNaturalist", "SUN", "Places", "dtd"])
    ap.add_argument("--workers", type=int, default=max(1, os.cpu_count() - 8))
    ap.add_argument("--chunk", type=int, default=64, help="images per task")
    ap.add_argument("--quality", type=int, default=85)
    ap.add_argument("--no-optimize", action="store_true", help="disable JPEG optimize (faster)")
    ap.add_argument("--no-resume", action="store_true", help="re-generate even if outputs exist")
    ap.add_argument("--jit", dest="jit", action="store_true", default=True)
    ap.add_argument("--no-jit", dest="jit", action="store_false")
    ap.add_argument("--limit", type=int, default=0, help="only first N source imgs (benchmark)")
    ap.add_argument("--src-root", default=DATASETS, help="root holding the original images (read)")
    ap.add_argument("--dst-root", default="", help="root to write corruptions into (default = src-root)")
    ap.add_argument("--corruptions", default="", help="comma list to restrict corruptions (default all 18)")
    ap.add_argument("--severities", default="", help="comma list to restrict severities (default 1-5)")
    args = ap.parse_args()

    # restrict the corruption/severity set (module globals are inherited by forked workers)
    global CORRUPTIONS, SEVERITIES
    if args.corruptions:
        sel = [c.strip() for c in args.corruptions.split(",") if c.strip()]
        bad = [c for c in sel if c not in CORRUPTIONS]
        assert not bad, f"unknown corruptions: {bad}"
        CORRUPTIONS = sel
    if args.severities:
        SEVERITIES = [int(s) for s in args.severities.split(",") if s.strip()]

    dst_root = args.dst_root or args.src_root
    cfgs = _cfg(args.src_root, dst_root)
    order = ["ImageNet", "iNaturalist", "SUN", "Places", "dtd"] if args.dataset == "all" else [args.dataset]
    print(f"workers={args.workers} chunk={args.chunk} quality={args.quality} "
          f"optimize={not args.no_optimize} resume={not args.no_resume} "
          f"numba={HAVE_NUMBA} datasets={order}\n  src_root={args.src_root}\n  dst_root={dst_root}", flush=True)
    total_err = 0
    for name in order:
        total_err += run_dataset(name, cfgs[name], args)
    print(f"ALL DONE. total errors={total_err}", flush=True)

if __name__ == "__main__":
    main()
