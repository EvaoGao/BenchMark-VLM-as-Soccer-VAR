#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Two-Stage VideoBERT (VideoMAE backbone) with RF heads
- Stage-1 on FULL (Head-1: Is Foul?)
- Stage-2 on FOUL-ONLY (Head-2: Card multiclass; Head-3: Advantage binary)

CLI example (train + eval + save):
python video_bert_two_stage.py \
  --train-full "/Users/wangshuyi/Desktop/BenchMark-VLM-as-Soccer-VAR-1/train_foul.json" \
  --train-foul "/Users/wangshuyi/Desktop/BenchMark-VLM-as-Soccer-VAR-1/train_card_adv.json" \
  --test-full  "/Users/wangshuyi/Desktop/BenchMark-VLM-as-Soccer-VAR-1/balanced_data.json" \
  --test-foul  "/Users/wangshuyi/Desktop/BenchMark-VLM-as-Soccer-VAR-1/balanced_fouls_data.json" \
  --frames-root "/Users/wangshuyi/Desktop/BenchMark-VLM-as-Soccer-VAR-1/frames_output" \
  --pretrained "MCG-NJU/videomae-base" --num-frames 16 --max-imgs 64 \
  --cache-full-train emb_full_train_vmae_16.npz \
  --cache-foul-train emb_foul_train_vmae_16.npz \
  --cache-full-test  emb_full_test_vmae_16.npz \
  --cache-foul-test  emb_foul_test_vmae_16.npz \
  --foul-min-spec 0.0 \
  --save-dir "saved_models/video_bert_two_stage"

Test-only on NEW FULL:
python video_bert_two_stage.py \
  --test-full "/path/to/new_full.json" \
  --frames-root "/Users/wangshuyi/Desktop/BenchMark-VLM-as-Soccer-VAR-1/frames_output" \
  --pretrained "MCG-NJU/videomae-base" --num-frames 16 --max-imgs 64 \
  --load-dir "saved_models/video_bert_two_stage" --test-only
"""

import os, io, json, argparse, time, contextlib, traceback
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple

import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
import joblib

import torch
from transformers import AutoImageProcessor, VideoMAEModel

# ---------------- Logging ----------------
from datetime import datetime
def now() -> str: return datetime.now().strftime("%Y-%m-%d %H:%M:%S")
def log(msg: str, level: str="INFO"): print(f"[{now()}] [{level}] {msg}", flush=True)
def log_exc(msg: str): print(f"[{now()}] [ERROR] {msg}", flush=True); traceback.print_exc()

class Timer:
    def __init__(self, name: str): self.name, self.t0 = name, None
    def __enter__(self): self.t0 = time.time(); log(f"{self.name} ...start"); return self
    def __exit__(self, et, ev, tb):
        dt = time.time() - self.t0
        if et: log_exc(f"{self.name} failed after {dt:.2f}s")
        else:  log(f"{self.name} ...done in {dt:.2f}s")

# ---------------- Label helpers ----------------
def reduce_card_label(field) -> str:
    if field is None: return "None"
    vals = field if isinstance(field, list) else [field]
    norm = []
    for v in vals:
        if v is None: norm.append("None"); continue
        s = str(v).strip().lower()
        if s in {"no card","none","no_card","0"}: norm.append("None")
        elif "red" in s or s=="2": norm.append("Red")
        elif "yellow" in s or s=="1": norm.append("Yellow")
        else: norm.append("None")
    if "Red" in norm: return "Red"
    if "Yellow" in norm: return "Yellow"
    return "None"

def normalize_bool(field) -> int:
    if isinstance(field, list):
        for x in field:
            if isinstance(x, bool) and x: return 1
            if isinstance(x, str) and x.strip().lower() in {"yes","true","y","1"}: return 1
        return 0
    if isinstance(field, bool): return int(field)
    if isinstance(field, (int, np.integer)): return int(field != 0)
    if isinstance(field, str): return int(field.strip().lower() in {"yes","true","y","1"})
    return 0

def normalize_foul(field) -> int: return normalize_bool(field)
def normalize_advantage(field) -> int: return normalize_bool(field)

# ---------------- I/O ----------------
def load_dataset(path: str) -> Dict[str, Any]:
    if not os.path.exists(path): raise FileNotFoundError(path)
    with Timer(f"Load dataset: {path}"):
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)

def build_df(data: Dict[str, Any]) -> pd.DataFrame:
    rows = []
    for aid, rec in (data.items() if isinstance(data, dict) else enumerate(data)):
        rec = dict(rec)
        rows.append({
            "action_id": str(aid),
            "y_foul": normalize_foul(rec.get("foul")),
            "y_card": reduce_card_label(rec.get("card")),
            "y_adv" : normalize_advantage(rec.get("advantage")),
            "video1" : rec.get("video1"), "video2": rec.get("video2")
        })
    df = pd.DataFrame(rows)
    if df.empty: raise ValueError("Empty dataframe from JSON.")
    log(f"DF shape: {df.shape} | positives foul/adv = {int(df.y_foul.sum())}/{int(df.y_adv.sum())}")
    return df

# ---------------- Frames ----------------
IMG_EXTS = {".jpg",".jpeg",".png",".bmp",".webp"}
def list_images(folder: Path, limit: int) -> List[str]:
    if not folder.exists(): return []
    imgs = [str(p) for p in sorted(folder.iterdir()) if p.suffix.lower() in IMG_EXTS]
    return imgs if limit<=0 else imgs[:limit]

def ensure_frames_for_action(aid: str, v1: Optional[str], v2: Optional[str], root: str) -> Path:
    # 假设帧已存在 frames_root/<aid>；若没有，你可以在这里接上你自己的提帧逻辑
    out = Path(root) / aid
    out.mkdir(parents=True, exist_ok=True)
    return out

def uniform_sample(paths: List[str], n: int) -> List[str]:
    if not paths: raise ValueError("No frames found")
    if len(paths) >= n:
        idx = np.linspace(0, len(paths)-1, n, dtype=int)
        return [paths[i] for i in idx]
    return paths + [paths[-1]]*(n-len(paths))

def load_rgb(paths: List[str]):
    from PIL import Image
    return [Image.open(p).convert("RGB") for p in paths]

# ---------------- VideoMAE embeddings ----------------
@torch.no_grad()
def videomae_embed(img_paths: List[str], processor, model, device: str, n_frames: int) -> np.ndarray:
    # VideoMAE 要求 num_frames 为 tubelet_size 的倍数
    tube = getattr(model.config, "tubelet_size", 2)
    n_req = getattr(model.config, "num_frames", n_frames or 16)
    n_req = ((max(n_req, tube) + tube - 1)//tube)*tube
    sel = uniform_sample(img_paths, n_req)
    imgs = load_rgb(sel)
    inputs = processor([imgs], return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}
    out = model(**inputs)
    if getattr(out, "pooler_output", None) is not None:
        vec = out.pooler_output[0]
    else:
        vec = out.last_hidden_state.mean(1)[0]
    return vec.detach().cpu().numpy().astype("float32")

def compute_embeddings(df: pd.DataFrame, frames_root: str, cache_path: Optional[str],
                       model_name: str, device: str, num_frames: int, max_imgs: int,
                       log_every: int=25) -> np.ndarray:
    if cache_path and os.path.exists(cache_path):
        try:
            cache = np.load(cache_path, allow_pickle=True)
            if cache.get("model")==model_name and cache.get("num_frames")==num_frames \
               and cache.get("ids").tolist()==list(df.action_id):
                E = cache["emb"]
                log(f"Use cached embeddings: {cache_path} shape={E.shape}")
                return E
        except Exception: log("Cache load failed; recompute...", level="WARN")

    processor = AutoImageProcessor.from_pretrained(model_name)
    backbone  = VideoMAEModel.from_pretrained(model_name).to(device).eval()

    embs=[]; N=len(df)
    with Timer(f"Compute embeddings (N={N}, frames={num_frames})"):
        for i, (_, row) in enumerate(df.iterrows(), 1):
            folder = ensure_frames_for_action(row.action_id, row.video1, row.video2, frames_root)
            img_paths = list_images(folder, max_imgs)
            if not img_paths:
                # 空帧：用 0 向量占位
                dim = getattr(backbone.config, "hidden_size", 768)
                embs.append(np.zeros((dim,), dtype="float32"))
            else:
                embs.append(videomae_embed(img_paths, processor, backbone, device, num_frames))
            if (i%log_every)==0 or i==N:
                log(f"Embeddings {i}/{N} (id={row.action_id}, imgs={len(img_paths)})")

    E = np.stack(embs, axis=0).astype("float32")
    if cache_path:
        with Timer(f"Save cache: {cache_path}"):
            np.savez_compressed(cache_path, emb=E, ids=df.action_id.values,
                                model=model_name, num_frames=num_frames)
    return E

# ---------------- Utils ----------------
def pick_device() -> str:
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available(): return "mps"
    if torch.cuda.is_available(): return "cuda"
    return "cpu"

def pad_to_dim(E: np.ndarray, target_dim: int) -> np.ndarray:
    if E.shape[1] == target_dim: return E
    if E.shape[1] > target_dim:  return E[:, :target_dim].copy()
    pad = np.zeros((E.shape[0], target_dim - E.shape[1]), dtype=E.dtype)
    return np.hstack([E, pad])

def make_rf(class_weight=None) -> RandomForestClassifier:
    return RandomForestClassifier(
        n_estimators=600, n_jobs=-1, random_state=42,
        class_weight=class_weight or "balanced_subsample",
        max_depth=12, min_samples_split=2, min_samples_leaf=3,
    )

def _spec_tpr_fpr(y_true: np.ndarray, y_pred: np.ndarray) -> Tuple[float,float,float]:
    y_true = y_true.astype(int); y_pred = y_pred.astype(int)
    tn = int(((y_true==0)&(y_pred==0)).sum())
    fp = int(((y_true==0)&(y_pred==1)).sum())
    fn = int(((y_true==1)&(y_pred==0)).sum())
    tp = int(((y_true==1)&(y_pred==1)).sum())
    spec = tn/max(1,tn+fp); tpr = tp/max(1,tp+fn); fpr = fp/max(1,tn+fp)
    return spec,tpr,fpr

def find_best_threshold_bacc(y_true: np.ndarray, y_proba: np.ndarray, min_specificity: float=0.0):
    grid = np.linspace(0.05,0.95,19)
    best=(-1.0,0.5,(0.0,0.0,0.0)); best_any=(-1.0,0.5,(0.0,0.0,0.0))
    for thr in grid:
        pred = (y_proba >= thr).astype(int)
        spec,tpr,fpr = _spec_tpr_fpr(y_true, pred)
        bacc = 0.5*(spec+tpr)
        if bacc>best_any[0]: best_any=(bacc,thr,(spec,tpr,fpr))
        if spec+1e-12>=min_specificity and bacc>best[0]: best=(bacc,thr,(spec,tpr,fpr))
    chosen = best if best[0]>=0 else best_any
    return chosen[1], chosen[0], chosen[2]  # thr, bacc, (spec,tpr,fpr)

# ---------------- Save / Load ----------------
def save_bundle(save_dir: str, rf_foul, rf_card, rf_adv, scaler, le_card, thresholds: dict, config: dict):
    Path(save_dir).mkdir(parents=True, exist_ok=True)
    joblib.dump(rf_foul, os.path.join(save_dir, "rf_foul.pkl"))
    joblib.dump(rf_card, os.path.join(save_dir, "rf_card.pkl"))
    joblib.dump(rf_adv, os.path.join(save_dir, "rf_adv.pkl"))
    joblib.dump(scaler,  os.path.join(save_dir, "scaler.pkl"))
    joblib.dump(le_card, os.path.join(save_dir, "label_card.pkl"))
    with open(os.path.join(save_dir, "thresholds.json"), "w") as f: json.dump(thresholds, f, indent=2)
    with open(os.path.join(save_dir, "config.json"), "w") as f: json.dump(config, f, indent=2)
    log(f"Models & config saved to: {save_dir}")

def load_bundle(load_dir: str):
    rf_foul = joblib.load(os.path.join(load_dir, "rf_foul.pkl"))
    rf_card = joblib.load(os.path.join(load_dir, "rf_card.pkl"))
    rf_adv  = joblib.load(os.path.join(load_dir, "rf_adv.pkl"))
    scaler  = joblib.load(os.path.join(load_dir, "scaler.pkl"))
    le_card = joblib.load(os.path.join(load_dir, "label_card.pkl"))
    with open(os.path.join(load_dir, "thresholds.json")) as f: thresholds = json.load(f)
    with open(os.path.join(load_dir, "config.json")) as f:     config     = json.load(f)
    log(f"Loaded models & config from: {load_dir}")
    return rf_foul, rf_card, rf_adv, scaler, le_card, thresholds, config

# ---------------- Main ----------------
def main():
    ap = argparse.ArgumentParser(description="Two-Stage VideoBERT (VideoMAE backbone) with RF heads")
    ap.add_argument("--train-full", type=str, required=False, default="train_foul.json")
    ap.add_argument("--train-foul", type=str, required=False, default="train_card_adv.json")
    ap.add_argument("--test-full",  type=str, required=True)
    ap.add_argument("--test-foul",  type=str, required=True)

    ap.add_argument("--frames-root", type=str, required=True)
    ap.add_argument("--pretrained",  type=str, default="MCG-NJU/videomae-base")
    ap.add_argument("--num-frames",  type=int, default=16)
    ap.add_argument("--max-imgs",    type=int, default=64)
    ap.add_argument("--log-every",   type=int, default=25)

    ap.add_argument("--cache-full-train", type=str, default="emb_full_train_vmae.npz")
    ap.add_argument("--cache-foul-train", type=str, default="emb_foul_train_vmae.npz")
    ap.add_argument("--cache-full-test",  type=str, default="emb_full_test_vmae.npz")
    ap.add_argument("--cache-foul-test",  type=str, default="emb_foul_test_vmae.npz")

    ap.add_argument("--foul-min-spec", type=float, default=0.0)
    ap.add_argument("--save-dir", type=str, default="saved_models/video_bert_two_stage")
    ap.add_argument("--load-dir", type=str, default="")
    ap.add_argument("--test-only", action="store_true")
    args = ap.parse_args()

    device = pick_device()
    log(f"Device: {device} | backbone={args.pretrained}")

    # ---------- TEST-ONLY ----------
    if args.test_only:
        if not args.load_dir: raise ValueError("--test-only requires --load-dir")
        rf_foul, rf_card, rf_adv, scaler, le_card, thr, cfg = load_bundle(args.load_dir)

        df_full = build_df(load_dataset(args.test_full))
        E_full  = compute_embeddings(df_full, args.frames_root, None,
                                     cfg.get("pretrained", args.pretrained), device,
                                     cfg.get("num_frames", args.num_frames), args.max_imgs, args.log_every)
        E_full  = pad_to_dim(E_full, scaler.mean_.shape[0])
        X_full  = scaler.transform(E_full)

        foul_thr = float(thr.get("foul", 0.5))
        foul_pred = (rf_foul.predict_proba(X_full)[:,1] >= foul_thr).astype(int)

        out_card, out_adv = {}, {}
        idxs = np.where(foul_pred==1)[0]
        if len(idxs)>0:
            X_foul = X_full[idxs]
            card_enc = rf_card.predict(X_foul)
            adv_thr = float(thr.get("adv", 0.5))
            adv = (rf_adv.predict_proba(X_foul)[:,1] >= adv_thr).astype(int)
            for pos, c in zip(idxs, card_enc): out_card[pos] = le_card.classes_[c]
            for pos, a in zip(idxs, adv):      out_adv[pos]  = "Yes" if int(a)==1 else "No"

        final=[]
        for i, aid in enumerate(df_full.action_id.values):
            if foul_pred[i]==1:
                final.append({"id": aid, "decision":"Foul",
                              "card": out_card.get(i,"None"), "advantage": out_adv.get(i,"No")})
            else:
                final.append({"id": aid, "decision":"No Foul", "card":"None", "advantage":"No"})
        print("\n=== Inference on NEW FULL data (first 12) ===")
        print(json.dumps(final[:12], ensure_ascii=False, indent=2))
        return

    # ---------- TRAIN + EVAL + SAVE ----------
    df_full_train = build_df(load_dataset(args.train_full))
    df_foul_train = build_df(load_dataset(args.train_foul))
    df_full_test  = build_df(load_dataset(args.test_full))
    df_foul_test  = build_df(load_dataset(args.test_foul))

    # 去掉 train 与 test 交集，避免数据泄漏
    test_ids = set(df_full_test.action_id) | set(df_foul_test.action_id)
    log(f"Before dedup: full_train={len(df_full_train)}, foul_train={len(df_foul_train)}")
    df_full_train = df_full_train[~df_full_train.action_id.isin(test_ids)].reset_index(drop=True)
    df_foul_train = df_foul_train[~df_foul_train.action_id.isin(test_ids)].reset_index(drop=True)
    log(f"After  dedup: full_train={len(df_full_train)}, foul_train={len(df_foul_train)}")

    # FOUL-ONLY sanity
    if (df_foul_train.y_foul!=1).any():
        log(f"[WARN] TRAIN FOUL-ONLY contains {(df_foul_train.y_foul!=1).sum()} non-foul rows", level="WARN")
    if (df_foul_test.y_foul!=1).any():
        log(f"[WARN] TEST FOUL-ONLY contains {(df_foul_test.y_foul!=1).sum()} non-foul rows", level="WARN")

    # embeddings (+cache)
    E_full_train = compute_embeddings(df_full_train, args.frames_root, args.cache_full_train,
                                      args.pretrained, device, args.num_frames, args.max_imgs, args.log_every)
    E_foul_train = compute_embeddings(df_foul_train, args.frames_root, args.cache_foul_train,
                                      args.pretrained, device, args.num_frames, args.max_imgs, args.log_every)
    E_full_test  = compute_embeddings(df_full_test,  args.frames_root, args.cache_full_test,
                                      args.pretrained, device, args.num_frames, args.max_imgs, args.log_every)
    E_foul_test  = compute_embeddings(df_foul_test,  args.frames_root, args.cache_foul_test,
                                      args.pretrained, device, args.num_frames, args.max_imgs, args.log_every)

    # 统一特征维度（正常应等于 hidden_size=768）
    common_dim = max(E_full_train.shape[1], E_foul_train.shape[1], E_full_test.shape[1], E_foul_test.shape[1])
    if len({E_full_train.shape[1], E_foul_train.shape[1], E_full_test.shape[1], E_foul_test.shape[1]})>1:
        log(f"Unifying dims to {common_dim}")
    E_full_train = pad_to_dim(E_full_train, common_dim)
    E_foul_train = pad_to_dim(E_foul_train, common_dim)
    E_full_test  = pad_to_dim(E_full_test,  common_dim)
    E_foul_test  = pad_to_dim(E_foul_test,  common_dim)

    # 标准化（只在 FULL TRAIN 上 fit）
    with Timer("Fit StandardScaler (FULL TRAIN) & transform"):
        scaler = StandardScaler(with_mean=True, with_std=True)
        X_full_train = scaler.fit_transform(E_full_train)
        X_full_test  = scaler.transform(E_full_test)
        X_foul_train = scaler.transform(E_foul_train)
        X_foul_test  = scaler.transform(E_foul_test)

    # 准备标签
    y_foul_train = df_full_train.y_foul.astype(int).values
    y_foul_test  = df_full_test.y_foul.astype(int).values

    le_card = LabelEncoder()
    y_card_train = le_card.fit_transform(df_foul_train.y_card.astype(str).values)
    y_card_test  = le_card.transform(df_foul_test.y_card.astype(str).values)
    card_classes = list(le_card.classes_)

    y_adv_train = df_foul_train.y_adv.astype(int).values
    y_adv_test  = df_foul_test.y_adv.astype(int).values

    print("\n== FOUL-ONLY card counts (TRAIN) ==")
    print(pd.Series(df_foul_train.y_card.values).value_counts())
    print("== FOUL-ONLY card counts (TEST) ==")
    print(pd.Series(df_foul_test.y_card.values).value_counts())

    # Head-1: Foul (FULL)
    with Timer("Train Head-1 (Foul on FULL TRAIN)"):
        rf_foul = make_rf(class_weight="balanced_subsample")
        rf_foul.fit(X_full_train, y_foul_train)

    with Timer("Eval Head-1 (Foul on FULL TEST)"):
        p1 = rf_foul.predict_proba(X_full_test)[:,1]
        thr1, bacc1, (spec1, tpr1, fpr1) = find_best_threshold_bacc(y_foul_test, p1, args.foul_min_spec)
        y1 = (p1 >= thr1).astype(int)
        print("\n=== Head-1 (Is Foul) ===")
        print(f"Chosen threshold = {thr1:.3f} | Balanced Acc = {bacc1:.4f} | "
              f"Specificity = {spec1:.4f} | TPR = {tpr1:.4f} | FPR = {fpr1:.4f}")
        print(classification_report(y_foul_test, y1, digits=4))
        print("Confusion Matrix:\n", confusion_matrix(y_foul_test, y1))

    # Head-2: Card (FOUL-ONLY)
    with Timer("Train Head-2 (Card on FOUL-ONLY TRAIN)"):
        cnt = np.bincount(y_card_train, minlength=len(card_classes))
        tot = cnt.sum()
        cls_weight_card = {c: tot/(len(cnt)*max(1,cnt[c])) for c in range(len(cnt))}
        rf_card = make_rf(class_weight=cls_weight_card)
        rf_card.fit(X_foul_train, y_card_train)

    with Timer("Eval Head-2 (Card on FOUL-ONLY TEST)"):
        y2_pred = rf_card.predict(X_foul_test)
        print(f"\n=== Head-2 (Card: classes={card_classes}) ===")
        print(classification_report(y_card_test, y2_pred, target_names=card_classes, digits=4))
        print("Confusion Matrix:\n", confusion_matrix(y_card_test, y2_pred))

    # Head-3: Advantage (FOUL-ONLY)
    with Timer("Train Head-3 (Advantage on FOUL-ONLY TRAIN)"):
        rf_adv = make_rf(class_weight="balanced_subsample")
        rf_adv.fit(X_foul_train, y_adv_train)

    with Timer("Eval Head-3 (Advantage on FOUL-ONLY TEST)"):
        pa = rf_adv.predict_proba(X_foul_test)[:,1]
        thr_a, bacc_a, (spec_a, tpr_a, fpr_a) = find_best_threshold_bacc(y_adv_test, pa, 0.0)
        y3 = (pa >= thr_a).astype(int)
        print("\n=== Head-3 (Advantage) ===")
        print(f"Chosen threshold = {thr_a:.3f} | Balanced Acc = {bacc_a:.4f} | "
              f"Specificity = {spec_a:.4f} | TPR = {tpr_a:.4f} | FPR = {fpr_a:.4f}")
        print(classification_report(y_adv_test, y3, digits=4))
        print("Confusion Matrix:\n", confusion_matrix(y_adv_test, y3))

    # Save
    config = {
        "pretrained": args.pretrained,
        "feature_dim": int(common_dim),
        "num_frames": int(args.num_frames),
        "max_imgs": int(args.max_imgs),
        "card_classes": card_classes,
    }
    thresholds = {"foul": float(thr1), "adv": float(thr_a)}
    save_bundle(args.save_dir, rf_foul, rf_card, rf_adv, scaler, le_card, thresholds, config)

    # Sample 2-stage inference on FULL TEST
    with Timer("Two-Stage inference on FULL TEST"):
        foul_pred = (rf_foul.predict_proba(X_full_test)[:,1] >= thr1).astype(int)
        card_fill, adv_fill = {}, {}
        idxs = np.where(foul_pred==1)[0]
        if len(idxs)>0:
            X_foul = X_full_test[idxs]
            card_enc = rf_card.predict(X_foul)
            adv_pred = (rf_adv.predict_proba(X_foul)[:,1] >= thr_a).astype(int)
            for pos, c in zip(idxs, card_enc): card_fill[pos] = card_classes[c]
            for pos, a in zip(idxs, adv_pred): adv_fill[pos] = "Yes" if int(a)==1 else "No"

        assembled=[]
        for i, aid in enumerate(df_full_test.action_id.values):
            if foul_pred[i]==1:
                assembled.append({"id": aid, "decision":"Foul",
                                  "card": card_fill.get(i,"None"), "advantage": adv_fill.get(i,"No")})
            else:
                assembled.append({"id": aid, "decision":"No Foul", "card":"None", "advantage":"No"})
        print("\n=== Sample outputs on FULL TEST (first 12) ===")
        print(json.dumps(assembled[:12], ensure_ascii=False, indent=2))

if __name__ == "__main__":
    main()

