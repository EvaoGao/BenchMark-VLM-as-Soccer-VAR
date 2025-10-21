#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Two-head Random Forest with text-fallback embeddings
"""

import os, json, argparse
from typing import Dict, Any, List, Optional
from pathlib import Path
from glob import glob
from collections import Counter

import numpy as np
import pandas as pd

from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix

DATA_PATH   = "/Users/zhangxinyue/Desktop/BenchMark-VLM-as-Soccer-VAR/data/auto_labeled_data.json"
FRAMES_ROOT = "/Users/zhangxinyue/Desktop/BenchMark-VLM-as-Soccer-VAR/data/frames_output"

import build_embeddings as be

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".gif"}

# ---------- 数据读取 & 构造 ----------
def load_dataset(path: str) -> Dict[str, Any]:
    tried = [path, DATA_PATH]
    for p in tried:
        if p and os.path.exists(p):
            with open(p, "r", encoding="utf-8") as f:
                return json.load(f)
    raise FileNotFoundError(f"Data file not found. Tried: {tried}. CWD={os.getcwd()}")

def reduce_card_label(card_field) -> str:
    """规约成 3 类：None / Yellow / Red（优先级：Red > Yellow > None）"""
    if card_field is None: return "None"
    if isinstance(card_field, list):
        vals = [str(x).strip().lower() for x in card_field if x is not None]
    else:
        vals = [str(card_field).strip().lower()]
    if any(v == "red" for v in vals):    return "Red"
    if any(v == "yellow" for v in vals): return "Yellow"
    return "None"

def pick_question_text(rec: Dict[str, Any]) -> str:
    exps = rec.get("foul_explanation") or rec.get("explanation") or []
    if isinstance(exps, list) and len(exps) > 0:
        return " ".join(exps[:1])
    if isinstance(exps, str):
        return exps
    league = rec.get("league") or ""
    return f"League: {league}"

def build_df(data: Dict[str, Any]) -> pd.DataFrame:
    rows = []
    for aid, rec in data.items():
        fouls = rec.get("foul", [])
        if isinstance(fouls, list):
            y_foul = int(any(x is True for x in fouls))
        elif isinstance(fouls, bool):
            y_foul = int(fouls)
        else:
            y_foul = 0
        rows.append({
            "action_id": str(aid),
            "y_foul": y_foul,
            "y_card": reduce_card_label(rec.get("card")),
            "question": pick_question_text(rec),
        })
    df = pd.DataFrame(rows)
    if df.empty:
        raise ValueError("Empty dataframe from JSON.")
    return df

# ---------- 取图 ----------
def get_image_paths_for_action(aid: str, root: str, max_imgs: int = 8) -> List[str]:
    d = Path(root) / Path(aid)
    if d.exists() and d.is_dir():
        imgs = [str(p) for p in sorted(d.iterdir()) if p.suffix.lower() in IMG_EXTS]
        if imgs:
            return imgs[:max_imgs]

    flat = []
    aid_alt = str(aid).replace("/", "_")
    for pat in (f"*{aid}*", f"*{aid_alt}*"):
        for ext in IMG_EXTS:
            flat.extend(glob(str(Path(root) / f"{pat}{ext}")))
    flat = sorted(set(flat))
    return flat[:max_imgs]

# ---------- embeddings ----------
def compute_embeddings(df: pd.DataFrame,
                       frames_root: str,
                       max_imgs_per_action: int = 8,
                       cache_path: Optional[str] = "emb_cache.npz") -> np.ndarray:
    # 读取缓存
    if cache_path and os.path.exists(cache_path):
        cache = np.load(cache_path, allow_pickle=True)
        if list(df["action_id"]) == cache["ids"].tolist():
            return cache["emb"]

    embs = []
    for _, row in df.iterrows():
        aid = row["action_id"]; q = row.get("question", "") or ""
        img_paths = get_image_paths_for_action(aid, frames_root, max_imgs=max_imgs_per_action)
        try:
            if img_paths:
                v = be.build_features(img_paths, q)
                v = np.asarray(v, dtype="float32").reshape(-1)
            else:
                # 文本兜底：图像=全零(2048) + 文本(384) -> 2432
                txt = be.txt_embed(q).astype("float32").reshape(-1)      # 384
                img = np.zeros(2048, dtype="float32")                    # 2048
                v = np.concatenate([img, txt], axis=0).astype("float32") # 2432
        except Exception:
            txt = be.txt_embed(q).astype("float32").reshape(-1)
            img = np.zeros(2048, dtype="float32")
            v = np.concatenate([img, txt], axis=0).astype("float32")
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
        min_samples_leaf=1
    )

# ---------- 主流程 ----------
def main():
    ap = argparse.ArgumentParser(description="Two-head RF with text-fallback embeddings (terminal-only, silent)")
    ap.add_argument("--data", type=str, default=DATA_PATH)
    ap.add_argument("--frames-root", type=str, default=FRAMES_ROOT)
    ap.add_argument("--max-imgs", type=int, default=8)
    ap.add_argument("--test-size", type=float, default=0.2)
    ap.add_argument("--random-state", type=int, default=42)
    ap.add_argument("--cache", type=str, default="emb_cache.npz")
    args = ap.parse_args()

    data = load_dataset(args.data)
    df = build_df(data)

    # Embedding
    E = compute_embeddings(
        df, frames_root=args.frames_root,
        max_imgs_per_action=args.max_imgs,
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

    # 统一分层切分
    combo = y_foul * 2 + has_card
    sss = StratifiedShuffleSplit(n_splits=1, test_size=args.test_size, random_state=args.random_state)
    (train_idx, test_idx), = sss.split(E_std, combo)

    X_tr, X_te = E_std[train_idx], E_std[test_idx]
    yA_tr, yA_te = y_foul[train_idx], y_foul[test_idx]
    yB_tr, yB_te = y_card_enc[train_idx], y_card_enc[test_idx]

    # ===== Head-1: 是否犯规 =====
    rfA = make_rf(class_weight="balanced_subsample")
    rfA.fit(X_tr, yA_tr)
    yA_pred  = rfA.predict(X_te)
    print("\n=== Head-1 (Is Foul) ===")
    print(classification_report(yA_te, yA_pred, digits=4))
    print("Confusion Matrix:\n", confusion_matrix(yA_te, yA_pred))

    # ===== Head-2: 牌 =====
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
