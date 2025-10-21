#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Two-head Random Forest
Head-1: Is Foul? (binary: 0/1)
Head-2: Card (multiclass: None / Yellow / Red)
"""
# 让父目录可被 import（父目录里有 frame_extraction.py）
import os, sys
CUR_DIR = os.path.dirname(os.path.abspath(__file__))            # .../random_forest
PROJECT_ROOT = os.path.abspath(os.path.join(CUR_DIR, os.pardir))# .../
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

# 现在真正导入：父目录的 frame_extraction.py
import frame_extraction as fx

import io, json, argparse, contextlib
from typing import Dict, Any, List, Optional
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
from collections import Counter

# 你的路径
DATA_PATH   = "/Users/zhangxinyue/Desktop/BenchMark-VLM-as-Soccer-VAR/data/complete_consistent_data.json"
FRAMES_ROOT = "/Users/zhangxinyue/Desktop/BenchMark-VLM-as-Soccer-VAR/data/frames_output"

# 同目录下的 build_embeddings.py
import build_embeddings as be

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".gif"}

def load_dataset(path: str) -> Dict[str, Any]:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Data file not found: {path}")
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)
# ---------- 标签规约 ----------
def reduce_card_label(field) -> str:
    """
    归并到 {None, Yellow, Red}，其中 "No card"/None -> None
    如果是列表，优先级：Red > Yellow > None
    """
    if field is None:
        return "None"
    vals = field if isinstance(field, list) else [field]
    norm = []
    for v in vals:
        if v is None:
            norm.append("None")
        else:
            s = str(v).strip().lower()
            if s in {"no card", "none", "no_card"}:
                norm.append("None")
            elif "red" in s:
                norm.append("Red")
            elif "yellow" in s:
                norm.append("Yellow")
            else:
                norm.append("None")
    if "Red" in norm: return "Red"
    if "Yellow" in norm: return "Yellow"
    return "None"

def normalize_foul(field) -> int:
    """
    将 foul 转成 0/1：
    - 若是列表：只要有 True 就 1；全 False 则 0；其余默认 0
    - 若是标量 bool：True->1, False->0
    - 其他 -> 0
    """
    if isinstance(field, list):
        return int(any(True is x for x in field))
    if isinstance(field, bool):
        return int(field)
    return 0

# ---------- question 取值（只用 question，不用 explanation） ----------
def extract_question(rec: Dict[str, Any]) -> str:
    q = rec.get("question")
    return q if isinstance(q, str) else ""

# ---------- 构造 DataFrame ----------
def build_df(data: Dict[str, Any]) -> pd.DataFrame:
    rows = []
    for aid, rec in data.items():
        rows.append({
            "action_id": str(aid),
            "y_foul": normalize_foul(rec.get("foul")),
            "y_card": reduce_card_label(rec.get("card")),
            "question": extract_question(rec),
            "video1": rec.get("video1"),
            "video2": rec.get("video2"),
        })
    df = pd.DataFrame(rows)
    if df.empty:
        raise ValueError("Empty dataframe from JSON.")
    return df

# ---------- 抽帧：静默调用你的 frame_extraction ----------
def silent_extract(video_url: Optional[str], out_dir: str, fps: int = 5) -> None:
    if not video_url:
        return
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    # 若已有帧则跳过
    if any(Path(out_dir).glob("*.jpg")):
        return
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf), contextlib.redirect_stderr(buf):
        try:
            fx.extract_frames_from_video(video_url, out_dir, frames_per_second=fps)
        except Exception:
            pass  # 失败则后续用文本兜底

def ensure_frames_for_action(aid: str, v1: Optional[str], v2: Optional[str], root: str, fps: int = 5) -> str:
    out_dir = str(Path(root) / Path(aid))
    if not any(Path(out_dir).glob("*.jpg")):
        silent_extract(v1, out_dir, fps=fps)
        silent_extract(v2, out_dir, fps=fps)
    return out_dir

def list_images(folder: str, max_imgs: int = 8) -> List[str]:
    p = Path(folder)
    if not p.exists():
        return []
    imgs = [str(x) for x in sorted(p.iterdir()) if x.suffix.lower() in IMG_EXTS]
    return imgs[:max_imgs]

# ---------- Embeddings：图+文；无图则文本兜底 ----------
def compute_embeddings(df: pd.DataFrame,
                       frames_root: str,
                       max_imgs_per_action: int = 8,
                       fps: int = 5,
                       cache_path: Optional[str] = "emb_cache_two_heads.npz") -> np.ndarray:
    if cache_path and os.path.exists(cache_path):
        cache = np.load(cache_path, allow_pickle=True)
        if list(df["action_id"]) == cache["ids"].tolist():
            return cache["emb"]

    embs = []
    for _, row in df.iterrows():
        aid = row["action_id"]
        q = row.get("question", "") or ""
        out_dir = ensure_frames_for_action(aid, row.get("video1"), row.get("video2"), frames_root, fps=fps)
        img_paths = list_images(out_dir, max_imgs=max_imgs_per_action)
        try:
            if img_paths:
                v = be.build_features(img_paths, q)
                v = np.asarray(v, dtype="float32").reshape(-1)
            else:
                # 文本兜底：图像 2048 维全零 + 文本 384 维
                txt = be.txt_embed(q).astype("float32").reshape(-1)
                img = np.zeros(2048, dtype="float32")
                v = np.concatenate([img, txt], 0).astype("float32")
        except Exception:
            txt = be.txt_embed(q).astype("float32").reshape(-1)
            img = np.zeros(2048, dtype="float32")
            v = np.concatenate([img, txt], 0).astype("float32")
        embs.append(v)

    E = np.stack(embs, axis=0)
    if cache_path:
        np.savez_compressed(cache_path, emb=E, ids=df["action_id"].values)
    return E

# ---------- 模型 ----------
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

# ---------- 主流程 ----------
def main():
    ap = argparse.ArgumentParser(description="Two-head RF (IsFoul / Card) with image+question embeddings (silent)")
    ap.add_argument("--data", type=str, default=DATA_PATH)
    ap.add_argument("--frames-root", type=str, default=FRAMES_ROOT)
    ap.add_argument("--max-imgs", type=int, default=8)
    ap.add_argument("--fps", type=int, default=5)
    ap.add_argument("--test-size", type=float, default=0.2)
    ap.add_argument("--random-state", type=int, default=42)
    ap.add_argument("--cache", type=str, default="emb_cache_two_heads.npz")
    args = ap.parse_args()

    data = load_dataset(args.data)
    df = build_df(data)

    # 特征：图 + question
    E = compute_embeddings(
        df, frames_root=args.frames_root,
        max_imgs_per_action=args.max_imgs, fps=args.fps,
        cache_path=args.cache
    )  # [N, 2432]

    # 标准化
    scaler = StandardScaler(with_mean=True, with_std=True)
    E_std = scaler.fit_transform(E)

    # 标签
    y_foul = df["y_foul"].astype(int).values
    le_card = LabelEncoder()
    y_card_enc = le_card.fit_transform(df["y_card"].astype(str).values)
    card_classes = list(le_card.classes_)
    has_card = (df["y_card"].isin(["Yellow", "Red"])).astype(int).values

    # 分层切分（兼顾 是否犯规 × 是否有牌）
    combo = y_foul * 2 + has_card
    sss = StratifiedShuffleSplit(n_splits=1, test_size=args.test_size, random_state=args.random_state)
    (train_idx, test_idx), = sss.split(E_std, combo)

    X_tr, X_te = E_std[train_idx], E_std[test_idx]
    yA_tr, yA_te = y_foul[train_idx], y_foul[test_idx]
    yB_tr, yB_te = y_card_enc[train_idx], y_card_enc[test_idx]

    # ===== Head-1: 是否犯规 =====
    rfA = make_rf(class_weight="balanced_subsample")
    rfA.fit(X_tr, yA_tr)
    yA_pred = rfA.predict(X_te)
    print("\n=== Head-1 (Is Foul) ===")
    print(classification_report(yA_te, yA_pred, digits=4))
    print("Confusion Matrix:\n", confusion_matrix(yA_te, yA_pred))

    # ===== Head-2: 牌（多类，类权重缓解不平衡） =====
    cnt = Counter(yB_tr); total = sum(cnt.values())
    cls_weight = {c: total/(len(cnt)*cnt[c]) for c in cnt}
    rfB = make_rf(class_weight=cls_weight)
    rfB.fit(X_tr, yB_tr)
    yB_pred = rfB.predict(X_te)
    print("\n=== Head-2 (Card: classes={}) ===".format(card_classes))
    print(classification_report(yB_te, yB_pred, target_names=card_classes, digits=4))
    print("Confusion Matrix:\n", confusion_matrix(yB_te, yB_pred))

if __name__ == "__main__":
    main()
