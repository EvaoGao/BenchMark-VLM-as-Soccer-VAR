#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Two-Stage RF (Stage-1 on FULL; Stage-2 on FOUL-ONLY)
- Head-1: Is Foul?            (binary, FULL)
- Head-2: Card                (multiclass: None/Yellow/Red, FOUL-ONLY)
- Head-3: Advantage           (binary, FOUL-ONLY)

Train:
- Stage-1 FULL train:   complete_consistent_data.json
- Stage-2 FOUL train:   fouls_data.json

Test:
- Stage-1 FULL test:    balanced_data.json
- Stage-2 FOUL test:    balanced_fouls_data.json

Key features:
- FPS=10, per video use up to 50 frames
- One StandardScaler fitted on FULL-TRAIN then applied to all four sets
- Balanced-Accuracy threshold search with minimum specificity constraint (Head-1/Head-3)
- Print specificity/TPR/FPR for chosen thresholds
- Save/Load bundle for future test-only runs on NEW FULL data

CLI examples (train + eval + save):

python new_randomforest.py \
  --train-full "/Users/zhangxinyue/Desktop/BenchMark-VLM-as-Soccer-VAR/data/complete_consistent_data.json" \
  --train-foul "/Users/zhangxinyue/Desktop/BenchMark-VLM-as-Soccer-VAR/data/fouls_data.json" \
  --test-full "/Users/zhangxinyue/Desktop/BenchMark-VLM-as-Soccer-VAR/data/balanced_data.json" \
  --test-foul "/Users/zhangxinyue/Desktop/BenchMark-VLM-as-Soccer-VAR/data/balanced_fouls_data.json" \
  --frames-root "/Users/zhangxinyue/Desktop/BenchMark-VLM-as-Soccer-VAR/data/frames_output" \
  --use-question false --fps 10 --max-imgs 50 \
  --cache-full-train emb_full_train_fps10_50.npz \
  --cache-full-test emb_full_test_fps10_50.npz \
  --cache-foul-train emb_foul_train_fps10_50.npz \
  --cache-foul-test emb_foul_test_fps10_50.npz \
  --foul-min-spec 0.60 \
  --save-dir "saved_models/two_stage_rf_v3"

Test-only on NEW FULL:

python new_randomforest.py \
  --test-full "/path/to/new_full.json" \
  --frames-root "/Users/zhangxinyue/Desktop/BenchMark-VLM-as-Soccer-VAR/data/frames_output" \
  --use-question false --fps 10 --max-imgs 50 \
  --load-dir "saved_models/two_stage_rf_v3" --test-only
"""

import os, sys, io, json, argparse, contextlib, time, traceback
from typing import Dict, Any, List, Optional, Tuple
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
import joblib

# --- Project imports (adjust if needed) ---
CUR_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(CUR_DIR, os.pardir))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import frame_extraction as fx
import build_embeddings as be

# ---------------- Defaults ----------------
TRAIN_FULL_DEFAULT = "/Users/zhangxinyue/Desktop/BenchMark-VLM-as-Soccer-VAR/data/complete_consistent_data.json"
TRAIN_FOUL_DEFAULT = "/Users/zhangxinyue/Desktop/BenchMark-VLM-as-Soccer-VAR/data/fouls_data.json"
TEST_FULL_DEFAULT  = "/Users/zhangxinyue/Desktop/BenchMark-VLM-as-Soccer-VAR/data/balanced_data.json"
TEST_FOUL_DEFAULT  = "/Users/zhangxinyue/Desktop/BenchMark-VLM-as-Soccer-VAR/data/balanced_fouls_data.json"

FRAMES_ROOT_DEFAULT = "/Users/zhangxinyue/Desktop/BenchMark-VLM-as-Soccer-VAR/data/frames_output"
IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".gif"}

# ---------------- Logging ----------------
def now() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")

def log(msg: str, *, level: str = "INFO"):
    print(f"[{now()}] [{level}] {msg}", flush=True)

def log_exc(msg: str):
    print(f"[{now()}] [ERROR] {msg}", flush=True)
    traceback.print_exc()

class Timer:
    def __init__(self, name: str):
        self.name, self.t0 = name, None
    def __enter__(self):
        self.t0 = time.time()
        log(f"{self.name} ...start")
        return self
    def __exit__(self, exc_type, exc, tb):
        dt = time.time() - self.t0
        if exc:
            log_exc(f"{self.name} failed after {dt:.2f}s")
        else:
            log(f"{self.name} ...done in {dt:.2f}s")

# ---------------- Label helpers ----------------
def reduce_card_label(field) -> str:
    if field is None:
        return "None"
    vals = field if isinstance(field, list) else [field]
    norm = []
    for v in vals:
        if v is None:
            norm.append("None")
            continue
        s = str(v).strip().lower()
        if s in {"no card", "none", "no_card"}:
            norm.append("None")
        elif "red" in s:
            norm.append("Red")
        elif "yellow" in s:
            norm.append("Yellow")
        else:
            norm.append("None")
    if "Red" in norm:
        return "Red"
    if "Yellow" in norm:
        return "Yellow"
    return "None"

def normalize_bool(field) -> int:
    if isinstance(field, list):
        for x in field:
            if isinstance(x, bool) and x:
                return 1
            if isinstance(x, str) and x.strip().lower() in {"yes", "true", "y", "1"}:
                return 1
        return 0
    if isinstance(field, bool):
        return int(field)
    if isinstance(field, (int, np.integer)):
        return int(field != 0)
    if isinstance(field, str):
        return int(field.strip().lower() in {"yes", "true", "y", "1"})
    return 0

def normalize_foul(field) -> int:
    return normalize_bool(field)

def normalize_advantage(field) -> int:
    return normalize_bool(field)

def extract_question(rec: Dict[str, Any]) -> str:
    q = rec.get("question")
    return q if isinstance(q, str) else ""

# ---------------- Data I/O ----------------
def load_dataset(path: str) -> Dict[str, Any]:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Data file not found: {path}")
    with Timer(f"Load dataset from {path}"):
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)

def build_df(data: Dict[str, Any]) -> pd.DataFrame:
    with Timer("Build DataFrame from JSON"):
        rows = []
        for aid, rec in data.items():
            rows.append({
                "action_id": str(aid),
                "y_foul": normalize_foul(rec.get("foul")),
                "y_card": reduce_card_label(rec.get("card")),
                "y_adv": normalize_advantage(rec.get("advantage")),
                "question": extract_question(rec),
                "video1": rec.get("video1"),
                "video2": rec.get("video2"),
            })
        df = pd.DataFrame(rows)
        if df.empty:
            raise ValueError("Empty dataframe from JSON.")
        log(f"DF shape: {df.shape}, positives (foul/adv): {df['y_foul'].sum()}/{df['y_adv'].sum()}")
        return df

# ---------------- Frames & Embeddings ----------------
def silent_extract(video_url: Optional[str], out_dir: str, fps: int = 10, verbose=False) -> None:
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    if not video_url:
        return
    if any(Path(out_dir).glob("*.jpg")):
        return
    try:
        if verbose:
            log(f"Extract frames: url={video_url} -> {out_dir} (fps={fps})")
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf), contextlib.redirect_stderr(buf):
            fx.extract_frames_from_video(video_url, out_dir, frames_per_second=fps)
    except Exception as e:
        log_exc(f"Frame extraction failed for {video_url}: {e}")

def ensure_frames_for_action(aid: str, v1: Optional[str], v2: Optional[str],
                             root: str, fps: int = 10, verbose=False) -> str:
    out_dir = str(Path(root) / Path(aid))
    if not any(Path(out_dir).glob("*.jpg")):
        silent_extract(v1, out_dir, fps=fps, verbose=verbose)
        silent_extract(v2, out_dir, fps=fps, verbose=verbose)
    return out_dir

def list_images(folder: str, max_imgs: int = 50) -> List[str]:
    p = Path(folder)
    if not p.exists():
        return []
    imgs = [str(x) for x in sorted(p.iterdir()) if x.suffix.lower() in IMG_EXTS]
    return imgs[:max_imgs]

def get_txt_dim() -> int:
    try:
        t = be.txt_embed("test")
        t = np.asarray(t).reshape(-1)
        return int(t.shape[0])
    except Exception:
        return 0

def compute_feature_vector(img_paths: List[str], question: str, use_question: bool,
                           txt_dim_hint: Optional[int] = None,
                           force_image_only_slice: bool = True) -> np.ndarray:
    q = question if use_question else ""
    txt_dim = txt_dim_hint if txt_dim_hint is not None else get_txt_dim()
    if img_paths:
        try:
            v = np.asarray(be.build_features(img_paths, q), dtype="float32").reshape(-1)
            if (not use_question) and force_image_only_slice and txt_dim > 0 and v.shape[0] > txt_dim:
                v = v[:-txt_dim]
            return v
        except Exception:
            log_exc(f"build_features failed (use_question={use_question}). Fallback embedding used.")
    IMG_DIM_DEFAULT = 2048
    img = np.zeros((IMG_DIM_DEFAULT,), dtype="float32")
    if use_question and txt_dim > 0:
        try:
            t = be.txt_embed(q).astype("float32").reshape(-1)
        except Exception:
            t = np.zeros((txt_dim,), dtype="float32")
        return np.concatenate([img, t], axis=0).astype("float32")
    else:
        return img.astype("float32")

def compute_embeddings(df: pd.DataFrame, frames_root: str, use_question: bool = True,
                       max_imgs_per_action: int = 64, fps: int = 10,
                       cache_path: Optional[str] = None,
                       log_every: int = 25, verbose: bool = False) -> np.ndarray:
    if cache_path and os.path.exists(cache_path):
        try:
            cache = np.load(cache_path, allow_pickle=True)
            if list(df["action_id"]) == cache["ids"].tolist() and bool(cache["use_q"]) == bool(use_question):
                E = cache["emb"]
                if isinstance(E, np.ndarray) and E.ndim == 2:
                    log(f"Use cached embeddings: {cache_path} shape={E.shape}")
                    return E
        except Exception:
            log("Cache load failed; will recompute.", level="WARN")

    txt_dim_hint = get_txt_dim()
    embs = []
    N = len(df)
    with Timer(f"Compute embeddings for {N} actions (use_question={use_question})"):
        for i, (_, row) in enumerate(df.iterrows(), 1):
            aid = row["action_id"]
            q = row.get("question", "") or ""
            out_dir = ensure_frames_for_action(aid, row.get("video1"), row.get("video2"),
                                               frames_root, fps=fps, verbose=verbose)
            img_paths = list_images(out_dir, max_imgs=max_imgs_per_action)
            v = compute_feature_vector(img_paths, q, use_question=use_question, txt_dim_hint=txt_dim_hint)
            embs.append(v.astype("float32"))
            if (i % log_every) == 0 or i == N:
                log(f"Embeddings progress: {i}/{N} (dir={Path(out_dir).name}, imgs={len(img_paths)})")

    max_dim = max(vec.shape[0] for vec in embs)
    E = np.zeros((len(embs), max_dim), dtype="float32")
    for i, vec in enumerate(embs):
        E[i, :vec.shape[0]] = vec

    if cache_path:
        with Timer(f"Save embeddings cache to {cache_path}"):
            np.savez_compressed(cache_path, emb=E, ids=df["action_id"].values, use_q=np.array(use_question))
    log(f"Embeddings shape: {E.shape}")
    return E

def pad_to_dim(E: np.ndarray, target_dim: int) -> np.ndarray:
    if E.shape[1] == target_dim:
        return E
    if E.shape[1] > target_dim:
        return E[:, :target_dim].copy()
    pad = np.zeros((E.shape[0], target_dim - E.shape[1]), dtype=E.dtype)
    return np.hstack([E, pad])

# ---------------- Models & thresholds ----------------
def make_rf(class_weight=None) -> RandomForestClassifier:
    return RandomForestClassifier(
        n_estimators=600, n_jobs=-1, random_state=42,
        class_weight=class_weight or "balanced_subsample",
        max_depth=12, min_samples_split=2, min_samples_leaf=3,
    )

def _spec_tpr_fpr(y_true: np.ndarray, y_pred: np.ndarray) -> Tuple[float, float, float]:
    y_true = y_true.astype(int)
    y_pred = y_pred.astype(int)
    tn = int(((y_true == 0) & (y_pred == 0)).sum())
    fp = int(((y_true == 0) & (y_pred == 1)).sum())
    fn = int(((y_true == 1) & (y_pred == 0)).sum())
    tp = int(((y_true == 1) & (y_pred == 1)).sum())
    spec = tn / max(1, tn + fp)
    tpr = tp / max(1, tp + fn)
    fpr = fp / max(1, tn + fp)
    return spec, tpr, fpr

def find_best_threshold_bacc(y_true: np.ndarray, y_proba: np.ndarray,
                             min_specificity: float = 0.0) -> Tuple[float, float, Tuple[float, float, float]]:
    """
    Search threshold in [0.05, 0.95] to maximize Balanced Accuracy,
    while enforcing specificity >= min_specificity if possible.
    Returns: (best_thr, best_bacc, (spec,tpr,fpr))
    """
    y_true = np.asarray(y_true).astype(int)
    y_proba = np.asarray(y_proba).astype(float)
    grid = np.linspace(0.05, 0.95, 19)
    best = (-1.0, 0.5, (0.0, 0.0, 0.0))
    best_any = (-1.0, 0.5, (0.0, 0.0, 0.0))

    for thr in grid:
        pred = (y_proba >= thr).astype(int)
        spec, tpr, fpr = _spec_tpr_fpr(y_true, pred)
        bacc = 0.5 * (spec + tpr)
        if bacc > best_any[0]:
            best_any = (bacc, thr, (spec, tpr, fpr))
        if spec + 1e-12 >= min_specificity and bacc > best[0]:
            best = (bacc, thr, (spec, tpr, fpr))

    chosen = best if best[0] >= 0 else best_any
    return chosen[1], chosen[0], chosen[2]

# ---------------- Save/Load ----------------
def save_bundle(save_dir: str, rf_foul, rf_card, rf_adv, scaler, le_card, thresholds: dict, config: dict):
    Path(save_dir).mkdir(parents=True, exist_ok=True)
    joblib.dump(rf_foul, os.path.join(save_dir, "rf_foul.pkl"))
    joblib.dump(rf_card, os.path.join(save_dir, "rf_card.pkl"))
    joblib.dump(rf_adv, os.path.join(save_dir, "rf_adv.pkl"))
    joblib.dump(scaler, os.path.join(save_dir, "scaler.pkl"))
    joblib.dump(le_card, os.path.join(save_dir, "label_card.pkl"))
    with open(os.path.join(save_dir, "thresholds.json"), "w") as f:
        json.dump(thresholds, f, indent=2)
    with open(os.path.join(save_dir, "config.json"), "w") as f:
        json.dump(config, f, indent=2)
    log(f"Models & config saved to: {save_dir}")

def load_bundle(load_dir: str):
    rf_foul = joblib.load(os.path.join(load_dir, "rf_foul.pkl"))
    rf_card = joblib.load(os.path.join(load_dir, "rf_card.pkl"))
    rf_adv = joblib.load(os.path.join(load_dir, "rf_adv.pkl"))
    scaler = joblib.load(os.path.join(load_dir, "scaler.pkl"))
    le_card = joblib.load(os.path.join(load_dir, "label_card.pkl"))
    with open(os.path.join(load_dir, "thresholds.json")) as f:
        thresholds = json.load(f)
    with open(os.path.join(load_dir, "config.json")) as f:
        config = json.load(f)
    log(f"Loaded models & config from: {load_dir}")
    return rf_foul, rf_card, rf_adv, scaler, le_card, thresholds, config

# ---------------- Main ----------------
def main():
    ap = argparse.ArgumentParser(description="Two-Stage RF (Stage-1 on FULL; Stage-2 on FOUL-ONLY)")
    # four datasets
    ap.add_argument("--train-full", type=str, default=TRAIN_FULL_DEFAULT,
                    help="FULL TRAIN dataset for Stage-1 (foul vs no-foul)")
    ap.add_argument("--train-foul", type=str, default=TRAIN_FOUL_DEFAULT,
                    help="FOUL-ONLY TRAIN dataset for Stage-2 (card + advantage)")
    ap.add_argument("--test-full", type=str, default=TEST_FULL_DEFAULT,
                    help="FULL TEST dataset for Stage-1 evaluation")
    ap.add_argument("--test-foul", type=str, default=TEST_FOUL_DEFAULT,
                    help="FOUL-ONLY TEST dataset for Stage-2 evaluation")
    # frames / feats
    ap.add_argument("--frames-root", type=str, default=FRAMES_ROOT_DEFAULT)
    ap.add_argument("--max-imgs", type=int, default=50)
    ap.add_argument("--fps", type=int, default=10)
    ap.add_argument("--use-question", type=str, default="false")
    ap.add_argument("--cache-full-train", type=str, default="emb_full_stage1_train.npz")
    ap.add_argument("--cache-full-test", type=str, default="emb_full_stage1_test.npz")
    ap.add_argument("--cache-foul-train", type=str, default="emb_foul_stage2_train.npz")
    ap.add_argument("--cache-foul-test", type=str, default="emb_foul_stage2_test.npz")
    ap.add_argument("--log-every", type=int, default=25)
    ap.add_argument("--verbose", action="store_true")
    # thresholds
    ap.add_argument("--foul-min-spec", type=float, default=0.0,
                    help="Minimum specificity constraint for Head-1 threshold search (0.0-1.0)")
    # save/load
    ap.add_argument("--save-dir", type=str, default="saved_models/two_stage_rf")
    ap.add_argument("--load-dir", type=str, default="")
    ap.add_argument("--test-only", action="store_true",
                    help="Load saved models and run 2-stage inference on --test-full")
    args = ap.parse_args()

    use_question = str(args.use_question).strip().lower() in {"1", "true", "t", "yes", "y"}
    log(f"Args parsed. use_question={use_question}, "
        f"max_imgs={args.max_imgs}, fps={args.fps}, foul_min_spec={args.foul_min_spec}")

    # ---------- TEST-ONLY ----------
    if args.test_only:
        if not args.load_dir:
            raise ValueError("--test-only requires --load-dir")
        rf_foul, rf_card, rf_adv, scaler, le_card, thr, cfg = load_bundle(args.load_dir)

        data_full = load_dataset(args.test_full)
        df_full = build_df(data_full)

        fps = args.fps or cfg.get("fps", 10)
        max_imgs = args.max_imgs or cfg.get("max_imgs", 50)
        E_full = compute_embeddings(df_full, frames_root=args.frames_root, use_question=use_question,
                                    max_imgs_per_action=max_imgs, fps=fps,
                                    cache_path=None, log_every=args.log_every, verbose=args.verbose)
        target_dim = scaler.mean_.shape[0]
        E_full = pad_to_dim(E_full, target_dim)
        X_full = scaler.transform(E_full)

        foul_thr = float(thr.get("foul", 0.5))
        foul_pred = (rf_foul.predict_proba(X_full)[:, 1] >= foul_thr).astype(int)

        out_card = {}
        out_adv = {}
        idxs = np.where(foul_pred == 1)[0]
        if len(idxs) > 0:
            X_foul = X_full[idxs]
            card_enc = rf_card.predict(X_foul)
            adv_thr = float(thr.get("adv", 0.5))
            adv = (rf_adv.predict_proba(X_foul)[:, 1] >= adv_thr).astype(int)
            for pos, c in zip(idxs, card_enc):
                out_card[pos] = le_card.classes_[c]
            for pos, a in zip(idxs, adv):
                out_adv[pos] = "Yes" if int(a) == 1 else "No"

        final = []
        for i, aid in enumerate(df_full["action_id"].values):
            if foul_pred[i] == 1:
                final.append({
                    "id": aid,
                    "decision": "Foul",
                    "card": out_card.get(i, "None"),
                    "advantage": out_adv.get(i, "No"),
                })
            else:
                final.append({"id": aid, "decision": "No Foul", "card": "None", "advantage": "No"})
        print("\n=== Inference on NEW FULL data (first 12) ===")
        try:
            print(json.dumps(final[:12], ensure_ascii=False, indent=2))
        except Exception:
            print(final[:12])
        return

    # ---------- TRAIN + EVAL + SAVE ----------
    # load four datasets
    data_full_train = load_dataset(args.train_full)
    data_foul_train = load_dataset(args.train_foul)
    data_full_test  = load_dataset(args.test_full)
    data_foul_test  = load_dataset(args.test_foul)

    df_full_train = build_df(data_full_train)    # Stage-1 train
    df_foul_train = build_df(data_foul_train)    # Stage-2 train
    df_full_test  = build_df(data_full_test)     # Stage-1 test
    df_foul_test  = build_df(data_foul_test)     # Stage-2 test
        # ---- 去掉 train 里和 test 重复的 action_id，避免数据泄漏 ----
    test_full_ids = set(df_full_test["action_id"])
    test_foul_ids = set(df_foul_test["action_id"])
    test_ids_union = test_full_ids | test_foul_ids

    log(f"Before dedup: full_train={len(df_full_train)}, foul_train={len(df_foul_train)}")
    df_full_train = df_full_train[~df_full_train["action_id"].isin(test_ids_union)].reset_index(drop=True)
    df_foul_train = df_foul_train[~df_foul_train["action_id"].isin(test_ids_union)].reset_index(drop=True)
    log(f"After  dedup: full_train={len(df_full_train)}, foul_train={len(df_foul_train)}")


    # sanity checks for FOUL-ONLY sets
    if (df_foul_train["y_foul"] != 1).any():
        neg = int((df_foul_train["y_foul"] != 1).sum())
        log(f"[WARN] TRAIN FOUL-ONLY file contains {neg} non-foul rows", level="WARN")
    if (df_foul_test["y_foul"] != 1).any():
        neg = int((df_foul_test["y_foul"] != 1).sum())
        log(f"[WARN] TEST FOUL-ONLY file contains {neg} non-foul rows", level="WARN")

    # embeddings (cache separately for each split)
    E_full_train = compute_embeddings(df_full_train, frames_root=args.frames_root, use_question=use_question,
                                      max_imgs_per_action=args.max_imgs, fps=args.fps,
                                      cache_path=args.cache_full_train, log_every=args.log_every, verbose=args.verbose)
    E_foul_train = compute_embeddings(df_foul_train, frames_root=args.frames_root, use_question=use_question,
                                      max_imgs_per_action=args.max_imgs, fps=args.fps,
                                      cache_path=args.cache_foul_train, log_every=args.log_every, verbose=args.verbose)
    E_full_test = compute_embeddings(df_full_test, frames_root=args.frames_root, use_question=use_question,
                                     max_imgs_per_action=args.max_imgs, fps=args.fps,
                                     cache_path=args.cache_full_test, log_every=args.log_every, verbose=args.verbose)
    E_foul_test = compute_embeddings(df_foul_test, frames_root=args.frames_root, use_question=use_question,
                                     max_imgs_per_action=args.max_imgs, fps=args.fps,
                                     cache_path=args.cache_foul_test, log_every=args.log_every, verbose=args.verbose)

    common_dim = max(E_full_train.shape[1], E_foul_train.shape[1],
                     E_full_test.shape[1], E_foul_test.shape[1])
    if len({E_full_train.shape[1], E_foul_train.shape[1],
            E_full_test.shape[1], E_foul_test.shape[1]}) > 1:
        log(f"Unifying feature dims to {common_dim} "
            f"(full_train={E_full_train.shape[1]}, foul_train={E_foul_train.shape[1]}, "
            f"full_test={E_full_test.shape[1]}, foul_test={E_foul_test.shape[1]})")

    E_full_train = pad_to_dim(E_full_train, common_dim)
    E_foul_train = pad_to_dim(E_foul_train, common_dim)
    E_full_test  = pad_to_dim(E_full_test, common_dim)
    E_foul_test  = pad_to_dim(E_foul_test, common_dim)

    # scaling (fit on FULL TRAIN only)
    with Timer("Fit StandardScaler (on FULL TRAIN) and transform all"):
        scaler = StandardScaler(with_mean=True, with_std=True)
        E_full_train_std = scaler.fit_transform(E_full_train)
        E_full_test_std  = scaler.transform(E_full_test)
        E_foul_train_std = scaler.transform(E_foul_train)
        E_foul_test_std  = scaler.transform(E_foul_test)
        log(f"E_full_train_std: {E_full_train_std.shape}, E_full_test_std: {E_full_test_std.shape}")
        log(f"E_foul_train_std: {E_foul_train_std.shape}, E_foul_test_std: {E_foul_test_std.shape}")

    # labels
    with Timer("Prepare labels"):
        # Stage-1 (FULL)
        y_foul_train = df_full_train["y_foul"].astype(int).values
        y_foul_test  = df_full_test["y_foul"].astype(int).values

        # Stage-2 (FOUL-ONLY)
        le_card = LabelEncoder()
        y_card_train = le_card.fit_transform(df_foul_train["y_card"].astype(str).values)
        card_classes = list(le_card.classes_)
        y_card_test  = le_card.transform(df_foul_test["y_card"].astype(str).values)

        y_adv_train = df_foul_train["y_adv"].astype(int).values
        y_adv_test  = df_foul_test["y_adv"].astype(int).values
        log(f"Card classes: {card_classes}")

        print("\n== FOUL-ONLY card counts (TRAIN) ==")
        print(pd.Series(df_foul_train["y_card"].values).value_counts())
        print("== FOUL-ONLY card counts (TEST) ==")
        print(pd.Series(df_foul_test["y_card"].values).value_counts())

    # Stage-1: Foul
    with Timer("Train Head-1 (Foul on FULL TRAIN)"):
        rf_foul = make_rf(class_weight="balanced_subsample")
        rf_foul.fit(E_full_train_std, y_foul_train)

    with Timer("Eval Head-1 (Foul on FULL TEST)"):
        y1_proba = rf_foul.predict_proba(E_full_test_std)[:, 1]
        thr1, bacc1, (spec1, tpr1, fpr1) = find_best_threshold_bacc(
            y_foul_test, y1_proba, min_specificity=float(args.foul_min_spec)
        )
        y1_pred = (y1_proba >= thr1).astype(int)
        print("\n=== Head-1 (Is Foul) ===")
        print(f"Chosen threshold = {thr1:.3f} | Balanced Acc = {bacc1:.4f} | "
              f"Specificity = {spec1:.4f} | TPR = {tpr1:.4f} | FPR = {fpr1:.4f}")
        print(classification_report(y_foul_test, y1_pred, digits=4))
        print("Confusion Matrix:\n", confusion_matrix(y_foul_test, y1_pred))

    # Stage-2: Card (FOUL-ONLY, multiclass)
    with Timer("Train Head-2 (Card multiclass on FOUL-ONLY TRAIN)"):
        cnt = np.bincount(y_card_train, minlength=len(card_classes))
        total = cnt.sum()
        cls_weight_card = {c: total / (len(cnt) * max(1, cnt[c])) for c in range(len(cnt))}
        rf_card = make_rf(class_weight=cls_weight_card)
        rf_card.fit(E_foul_train_std, y_card_train)

    with Timer("Eval Head-2 (Card on FOUL-ONLY TEST)"):
        y_c_pred = rf_card.predict(E_foul_test_std)
        print(f"\n=== Head-2 (Card: classes={card_classes}) ===")
        print(classification_report(y_card_test, y_c_pred, target_names=card_classes, digits=4))
        print("Confusion Matrix:\n", confusion_matrix(y_card_test, y_c_pred))

    # Stage-2: Advantage (FOUL-ONLY, binary)
    with Timer("Train Head-3 (Advantage on FOUL-ONLY TRAIN)"):
        rf_adv = make_rf(class_weight="balanced_subsample")
        rf_adv.fit(E_foul_train_std, y_adv_train)

    with Timer("Eval Head-3 (Advantage on FOUL-ONLY TEST)"):
        ya_proba = rf_adv.predict_proba(E_foul_test_std)[:, 1]
        thr_adv, bacc_adv, (spec_a, tpr_a, fpr_a) = find_best_threshold_bacc(
            y_adv_test, ya_proba, min_specificity=0.0
        )
        ya_pred = (ya_proba >= thr_adv).astype(int)
        print("\n=== Head-3 (Advantage) ===")
        print(f"Chosen threshold = {thr_adv:.3f} | Balanced Acc = {bacc_adv:.4f} | "
              f"Specificity = {spec_a:.4f} | TPR = {tpr_a:.4f} | FPR = {fpr_a:.4f}")
        print(classification_report(y_adv_test, ya_pred, digits=4))
        print("Confusion Matrix:\n", confusion_matrix(y_adv_test, ya_pred))

    # Save everything
    config = {
        "use_question": use_question,
        "feature_dim": int(E_full_train.shape[1]),
        "fps": int(args.fps),
        "max_imgs": int(args.max_imgs),
        "card_classes": card_classes,
    }
    thresholds = {"foul": float(thr1), "adv": float(thr_adv)}
    save_bundle(args.save_dir, rf_foul, rf_card, rf_adv, scaler, le_card, thresholds, config)

    # Sample 2-stage inference on FULL TEST set
    with Timer("Two-Stage inference on FULL TEST set"):
        foul_pred_test = (rf_foul.predict_proba(E_full_test_std)[:, 1] >= thr1).astype(int)
        card_fill = {}
        adv_fill = {}
        idxs = np.where(foul_pred_test == 1)[0]
        if len(idxs) > 0:
            X_foul = E_full_test_std[idxs]
            card_enc = rf_card.predict(X_foul)
            adv_pred = (rf_adv.predict_proba(X_foul)[:, 1] >= thr_adv).astype(int)
            for pos, c in zip(idxs, card_enc):
                card_fill[pos] = card_classes[c]
            for pos, a in zip(idxs, adv_pred):
                adv_fill[pos] = "Yes" if int(a) == 1 else "No"

        print("\n=== Sample outputs on FULL TEST (first 12) ===")
        assembled = []
        test_ids = df_full_test["action_id"].values
        for i in range(min(12, len(test_ids))):
            aid = test_ids[i]
            if foul_pred_test[i] == 1:
                assembled.append({
                    "id": aid,
                    "decision": "Foul",
                    "card": card_fill.get(i, "None"),
                    "advantage": adv_fill.get(i, "No"),
                })
            else:
                assembled.append({"id": aid, "decision": "No Foul", "card": "None", "advantage": "No"})
        try:
            print(json.dumps(assembled, ensure_ascii=False, indent=2))
        except Exception:
            print(assembled)

if __name__ == "__main__":
    main()
