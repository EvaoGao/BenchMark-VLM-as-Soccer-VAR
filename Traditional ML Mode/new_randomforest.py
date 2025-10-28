#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Three-head Random Forest (with progress logging)
Head-1: Is Foul?   (binary: 0/1)
Head-2: Card       (multiclass: None / Yellow / Red)
Head-3: Advantage  (binary: 0/1)

Input:
- Default: image frames + question text
- Image-only: set --use-question false (recommended when question is constant)

Logging:
- Timestamped progress lines at every major step to see where it might stall.
"""

import os, sys, io, json, argparse, contextlib, time, traceback
from typing import Dict, Any, List, Optional
from pathlib import Path
from collections import Counter
from datetime import datetime

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix

# --- Project imports ---
CUR_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(CUR_DIR, os.pardir))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)
import frame_extraction as fx
import build_embeddings as be

DATA_PATH   = "/Users/zhangxinyue/Desktop/BenchMark-VLM-as-Soccer-VAR/data/complete_consistent_data.json"
FRAMES_ROOT = "/Users/zhangxinyue/Desktop/BenchMark-VLM-as-Soccer-VAR/data/frames_output"
IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".gif"}

# ---------------- Logging helpers ----------------
def now() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")

def log(msg: str, *, level: str = "INFO"):
    print(f"[{now()}] [{level}] {msg}", flush=True)

def log_exc(msg: str):
    print(f"[{now()}] [ERROR] {msg}", flush=True)
    traceback.print_exc()

class Timer:
    def __init__(self, name: str):
        self.name = name
        self.t0 = None
    def __enter__(self):
        self.t0 = time.time(); log(f"{self.name} ...start")
        return self
    def __exit__(self, exc_type, exc, tb):
        dt = time.time() - self.t0
        if exc:
            log_exc(f"{self.name} failed after {dt:.2f}s")
        else:
            log(f"{self.name} ...done in {dt:.2f}s")

# ---------------- Label utilities ----------------
def reduce_card_label(field) -> str:
    if field is None: return "None"
    vals = field if isinstance(field, list) else [field]
    norm = []
    for v in vals:
        if v is None:
            norm.append("None")
        else:
            s = str(v).strip().lower()
            if s in {"no card", "none", "no_card"}: norm.append("None")
            elif "red" in s: norm.append("Red")
            elif "yellow" in s: norm.append("Yellow")
            else: norm.append("None")
    if "Red" in norm: return "Red"
    if "Yellow" in norm: return "Yellow"
    return "None"

def normalize_bool(field) -> int:
    if isinstance(field, list):
        for x in field:
            if isinstance(x, bool) and x: return 1
            if isinstance(x, str) and str(x).strip().lower() in {"yes","true","y","1"}: return 1
        return 0
    if isinstance(field, bool): return int(field)
    if isinstance(field, (int, np.integer)): return int(field != 0)
    if isinstance(field, str): return int(field.strip().lower() in {"yes","true","y","1"})
    return 0

def normalize_foul(field) -> int: return normalize_bool(field)
def normalize_advantage(field) -> int: return normalize_bool(field)

def extract_question(rec: Dict[str, Any]) -> str:
    q = rec.get("question")
    return q if isinstance(q, str) else ""

# ---------------- DataFrame build ----------------
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
                "y_adv":  normalize_advantage(rec.get("advantage")),
                "question": extract_question(rec),
                "video1": rec.get("video1"),
                "video2": rec.get("video2"),
            })
        df = pd.DataFrame(rows)
        if df.empty:
            raise ValueError("Empty dataframe from JSON.")
        log(f"DF shape: {df.shape}, positives (foul/adv): {df['y_foul'].sum()}/{df['y_adv'].sum()}")
        return df

# ---------------- Frame extraction ----------------
def silent_extract(video_url: Optional[str], out_dir: str, fps: int = 5, verbose=False) -> None:
    if not video_url:
        return
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    if any(Path(out_dir).glob("*.jpg")):
        return
    try:
        if verbose: log(f"Extract frames: url={video_url} -> {out_dir}")
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf), contextlib.redirect_stderr(buf):
            fx.extract_frames_from_video(video_url, out_dir, frames_per_second=fps)
    except Exception as e:
        log_exc(f"Frame extraction failed for {video_url}: {e}")

def ensure_frames_for_action(aid: str, v1: Optional[str], v2: Optional[str], root: str, fps: int = 5, verbose=False) -> str:
    out_dir = str(Path(root) / Path(aid))
    if not any(Path(out_dir).glob("*.jpg")):
        silent_extract(v1, out_dir, fps=fps, verbose=verbose)
        silent_extract(v2, out_dir, fps=fps, verbose=verbose)
    return out_dir

def list_images(folder: str, max_imgs: int = 8) -> List[str]:
    p = Path(folder)
    if not p.exists():
        return []
    imgs = [str(x) for x in sorted(p.iterdir()) if x.suffix.lower() in IMG_EXTS]
    return imgs[:max_imgs]

# ---------------- Embeddings ----------------
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
            if not use_question and force_image_only_slice and txt_dim > 0 and v.shape[0] > txt_dim:
                v = v[:-txt_dim]  # strip tail text part if present
            return v
        except Exception:
            log_exc(f"build_features failed (use_question={use_question}). Fallback to zeros/text.")

    # Fallbacks
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
                       max_imgs_per_action: int = 8, fps: int = 5, cache_path: Optional[str] = "emb_cache_three_heads.npz",
                       log_every: int = 25, verbose: bool = False) -> np.ndarray:
    # Attempt load cache
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
            out_dir = ensure_frames_for_action(aid, row.get("video1"), row.get("video2"), frames_root, fps=fps, verbose=verbose)
            img_paths = list_images(out_dir, max_imgs=max_imgs_per_action)
            v = compute_feature_vector(img_paths, q, use_question=use_question, txt_dim_hint=txt_dim_hint)
            embs.append(v.astype("float32"))
            if (i % log_every) == 0 or i == N:
                log(f"Embeddings progress: {i}/{N}  (dir={Path(out_dir).name}, imgs={len(img_paths)})")

    # Align dims
    max_dim = max(vec.shape[0] for vec in embs)
    E = np.zeros((len(embs), max_dim), dtype="float32")
    for i, vec in enumerate(embs):
        E[i, :vec.shape[0]] = vec

    if cache_path:
        with Timer(f"Save embeddings cache to {cache_path}"):
            np.savez_compressed(cache_path, emb=E, ids=df["action_id"].values, use_q=np.array(use_question))
    log(f"Embeddings shape: {E.shape}")
    return E

# ---------------- Model maker ----------------
def make_rf(class_weight=None) -> RandomForestClassifier:
    return RandomForestClassifier(
        n_estimators=600,
        n_jobs=-1,
        random_state=42,
        class_weight=class_weight or "balanced_subsample",
        max_depth=None,
        min_samples_split=2,
        min_samples_leaf=1,
    )

# ---------------- Main ----------------
def main():
    ap = argparse.ArgumentParser(description="Three-head RF (Foul / Card / Advantage) with logging")
    ap.add_argument("--data", type=str, default=DATA_PATH)
    ap.add_argument("--frames-root", type=str, default=FRAMES_ROOT)
    ap.add_argument("--max-imgs", type=int, default=8)
    ap.add_argument("--fps", type=int, default=5)
    ap.add_argument("--test-size", type=float, default=0.2)
    ap.add_argument("--random-state", type=int, default=42)
    ap.add_argument("--cache", type=str, default="emb_cache_three_heads.npz")
    ap.add_argument("--use-question", type=str, default="true",
                    help="true/false: include question text in features")
    ap.add_argument("--log-every", type=int, default=25, help="print per-sample progress every N items")
    ap.add_argument("--verbose", action="store_true", help="verbose logs (frame extraction URLs, etc.)")
    args = ap.parse_args()

    use_question = str(args.use_question).strip().lower() in {"1","true","t","yes","y"}
    log(f"Args parsed. use_question={use_question}, test_size={args.test_size}, max_imgs={args.max_imgs}, fps={args.fps}")

    # Load data & DF
    data = load_dataset(args.data)
    df = build_df(data)

    # Features
    E = compute_embeddings(
        df, frames_root=args.frames_root, use_question=use_question,
        max_imgs_per_action=args.max_imgs, fps=args.fps,
        cache_path=args.cache, log_every=args.log_every, verbose=args.verbose
    )

    # Scale
    with Timer("Fit StandardScaler"):
        scaler = StandardScaler(with_mean=True, with_std=True)
        E_std = scaler.fit_transform(E)
        log(f"E_std shape: {E_std.shape}")

    # Labels
    with Timer("Prepare labels"):
        y_foul = df["y_foul"].astype(int).values
        y_adv  = df["y_adv"].astype(int).values
        le_card = LabelEncoder()
        y_card_enc = le_card.fit_transform(df["y_card"].astype(str).values)
        card_classes = list(le_card.classes_)
        has_card = (df["y_card"].isin(["Yellow","Red"])).astype(int).values
        log(f"Card classes: {card_classes}")

    # Split
    with Timer("Stratified split"):
        combo = y_foul * 2 + has_card
        sss = StratifiedShuffleSplit(n_splits=1, test_size=args.test_size, random_state=args.random_state)
        (train_idx, test_idx), = sss.split(E_std, combo)
        X_tr, X_te = E_std[train_idx], E_std[test_idx]
        y_f_tr, y_f_te = y_foul[train_idx], y_foul[test_idx]
        y_c_tr, y_c_te = y_card_enc[train_idx], y_card_enc[test_idx]
        y_a_tr, y_a_te = y_adv[train_idx],  y_adv[test_idx]
        log(f"Train/Test sizes: {X_tr.shape}/{X_te.shape}")

    # Train Head-1: Foul
    with Timer("Train Head-1 (Foul)"):
        rf_foul = make_rf(class_weight="balanced_subsample")
        rf_foul.fit(X_tr, y_f_tr)
    with Timer("Eval Head-1 (Foul)"):
        y_f_pred = rf_foul.predict(X_te)
        print("\n=== Head-1 (Is Foul) ===")
        print(classification_report(y_f_te, y_f_pred, digits=4))
        print("Confusion Matrix:\n", confusion_matrix(y_f_te, y_f_pred))

    # Train Head-2: Card
    with Timer("Train Head-2 (Card)"):
        cnt = Counter(y_c_tr); total = sum(cnt.values())
        cls_weight_card = {c: total/(len(cnt)*cnt[c]) for c in cnt}
        rf_card = make_rf(class_weight=cls_weight_card)
        rf_card.fit(X_tr, y_c_tr)
    with Timer("Eval Head-2 (Card)"):
        y_c_pred = rf_card.predict(X_te)
        print(f"\n=== Head-2 (Card: classes={card_classes}) ===")
        print(classification_report(y_c_te, y_c_pred, target_names=card_classes, digits=4))
        print("Confusion Matrix:\n", confusion_matrix(y_c_te, y_c_pred))

    # Train Head-3: Advantage
    with Timer("Train Head-3 (Advantage)"):
        rf_adv = make_rf(class_weight="balanced_subsample")
        rf_adv.fit(X_tr, y_a_tr)
    with Timer("Eval Head-3 (Advantage)"):
        y_a_pred = rf_adv.predict(X_te)
        print("\n=== Head-3 (Advantage) ===")
        print(classification_report(y_a_te, y_a_pred, digits=4))
        print("Confusion Matrix:\n", confusion_matrix(y_a_te, y_a_pred))

    # Output sample JSON
    with Timer("Assemble & print sample JSON outputs"):
        out = []
        for i, idx in enumerate(test_idx):
            out.append({
                "id": df.iloc[idx]["action_id"],
                "decision": "Foul" if y_f_pred[i] == 1 else "No Foul",
                "card": card_classes[y_c_pred[i]],
                "advantage": "Yes" if y_a_pred[i] == 1 else "No"
            })
        try:
            print("\n=== Sample outputs (first 10) ===")
            print(json.dumps(out[:10], ensure_ascii=False, indent=2))
        except Exception:
            print(out[:10])

if __name__ == "__main__":
    main()
