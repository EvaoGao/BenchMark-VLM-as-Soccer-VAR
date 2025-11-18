"""
Three-head VideoBERT (VideoMAE backbone) — pure Python runner
- Edit CONFIG only, then run this file.
- Prints precision/recall/F1 & confusion matrices for Foul / Card / Advantage.
"""

import os, sys, io, json, time, traceback
from typing import Dict, Any, List, Optional
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd

from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import classification_report, confusion_matrix

import torch
import torch.nn as nn
from transformers import AutoImageProcessor, VideoMAEModel
from PIL import Image

REPO_ROOT  = Path(os.path.expanduser("~/Desktop/BenchMark-VLM-as-Soccer-VAR-1"))

DATA_JSON   = REPO_ROOT / "balanced_data_2.json"      
FRAMES_ROOT = REPO_ROOT / "frames_output"            
CACHE_PATH  = REPO_ROOT / "data" / "videomae_emb_cache.npz"

PRETRAINED  = "MCG-NJU/videomae-base"
NUM_FRAMES  = 16
MAX_IMGS    = 64

AUTO_EXTRACT_FPS = 2.0                             
TEST_SIZE   = 0.2
EPOCHS      = 6
LR          = 1e-3
RANDOM_STATE= 42
LOG_EVERY   = 25
VERBOSE     = False


try:
    sys.path.append(str(REPO_ROOT))
    import frame_extraction as fx
except Exception:
    fx = None  


def now() -> str: return datetime.now().strftime("%Y-%m-%d %H:%M:%S")
def log(msg: str, level="INFO"): print(f"[{now()}] [{level}] {msg}", flush=True)
def log_exc(msg: str): print(f"[{now()}] [ERROR] {msg}", flush=True); traceback.print_exc()
class Timer:
    def __init__(self, name): self.name=name; self.t0=None
    def __enter__(self): self.t0=time.time(); log(f"{self.name} ...start"); return self
    def __exit__(self, et, ev, tb):
        dt=time.time()-self.t0
        if et: log_exc(f"{self.name} failed after {dt:.2f}s")
        else:  log(f"{self.name} ...done in {dt:.2f}s")

def _truthy(x) -> bool:
    if isinstance(x, bool): return x
    if isinstance(x, (int, float, np.integer, np.floating)): return x != 0
    if isinstance(x, str): return x.strip().lower() in {"1","true","yes","y"}
    if isinstance(x, (list, tuple, set)): return any(_truthy(v) for v in x)
    if isinstance(x, dict): return any(_truthy(v) for v in x.values())
    return False

def reduce_card_label(x: Any) -> str:
    vals = x if isinstance(x, list) else [x]
    norm = []
    for v in vals:
        s = "" if v is None else str(v).strip().lower()
        if "red" in s: norm.append("Red")
        elif "yellow" in s or "yell" in s: norm.append("Yellow")
        elif s in {"", "none", "no card", "no_card", "n/a"}: norm.append("None")
    if "Red" in norm: return "Red"
    if "Yellow" in norm: return "Yellow"
    return "None"

def normalize_advantage(x: Any) -> int:
    return int(_truthy(x))

def normalize_foul_rec(rec: Dict[str, Any]) -> int:

    if "foul" in rec:     return int(_truthy(rec["foul"]))
    if "no_foul" in rec:  return int(not _truthy(rec["no_foul"]))
    txt = " ".join([
        str(rec.get("decision","")),
        str(rec.get("question","")),
        str(rec.get("answer","")),
    ]).lower()
    if any(k in txt for k in ["no foul","not a foul","non-foul","clean"]): return 0
    if "foul" in txt: return 1
    return 0

def load_dataset(path: Path):
    if not path.exists(): raise FileNotFoundError(path)
    with Timer(f"Load dataset: {path}"):
        with open(path,"r",encoding="utf-8") as f:
            return json.load(f)

def _to_items(obj)->List[Dict[str,Any]]:
    rows=[]
    if isinstance(obj,dict):
        for k,v in obj.items():
            r=dict(v); r["action_id"]=str(k); rows.append(r)
    else:
        for v in obj:
            r=dict(v)
            r.setdefault("action_id", str(r.get("video_id") or r.get("id") or r.get("path")))
            rows.append(r)
    return rows

def build_df(data)->pd.DataFrame:
    with Timer("Build DataFrame"):
        rows=[]
        for rec in _to_items(data):
            rows.append({
                "action_id": str(rec["action_id"]),
                "y_foul": normalize_foul_rec(rec),
                "y_card": reduce_card_label(rec.get("card")),
                "y_adv" : normalize_advantage(rec.get("advantage")),
                "video1": rec.get("video1"),
                "video2": rec.get("video2"),
            })
        df=pd.DataFrame(rows).drop_duplicates(subset=["action_id"])
        log(f"DF shape: {df.shape} | positives foul/adv = {int(df.y_foul.sum())}/{int(df.y_adv.sum())}")
        return df


IMG_EXTS={".jpg",".jpeg",".png",".bmp",".webp"}
def list_images(folder: Path)->List[str]:
    if not folder.exists(): return []
    return [str(p) for p in sorted(folder.iterdir()) if p.suffix.lower() in IMG_EXTS]

def _uniform_sample_paths(paths: List[str], n: int) -> List[str]:
    if not paths: return []
    if len(paths) >= n:
        idx = np.linspace(0, len(paths) - 1, n, dtype=int)
        return [paths[i] for i in idx]
    return paths + [paths[-1]] * (n - len(paths))

def _load_rgb(paths: List[str]) -> List[Image.Image]:
    return [Image.open(p).convert("RGB") for p in paths]

def silent_extract(url: Optional[str], out_dir: Path, fps: Optional[float]):
    if not url or fx is None: return
    out_dir.mkdir(parents=True, exist_ok=True)
    if any(out_dir.glob("*.jpg")): return
    try:
        if VERBOSE: log(f"Extract frames {url} -> {out_dir}")
        import contextlib
        buf=io.StringIO()
        with contextlib.redirect_stdout(buf), contextlib.redirect_stderr(buf):
            if fps is None:
                fx.extract_frames_from_video(url, output_dir=str(out_dir))
            else:
                fx.extract_frames_from_video(url, output_dir=str(out_dir), frames_per_second=fps)
    except Exception as e:
        log_exc(f"Frame extraction failed for {url}: {e}")

def ensure_frames(action_id:str, v1:Optional[str], v2:Optional[str], root:Path, fps:Optional[float])->Path:
    od = root / action_id
    if not any(od.glob("*.jpg")):
        silent_extract(v1, od, fps); silent_extract(v2, od, fps)
    return od


@torch.no_grad()
def videomae_embed(img_paths, processor, backbone, device, num_frames: int):
    n_req = getattr(backbone.config, "num_frames", num_frames or 16)
    tube  = getattr(backbone.config, "tubelet_size", 2)
    n_req = max(n_req, tube)
    n_req = ((n_req + tube - 1) // tube) * tube

    frame_paths = _uniform_sample_paths(img_paths, n_req)
    if not frame_paths:
        hidden = int(getattr(backbone.config, "hidden_size", 768))
        return np.zeros((hidden,), dtype="float32")

    imgs  = _load_rgb(frame_paths)
    batch = processor([imgs], return_tensors="pt")  
    batch = {k: v.to(device) for k, v in batch.items()}
    out   = backbone(**batch)

    if hasattr(out, "pooler_output") and out.pooler_output is not None:
        vec = out.pooler_output[0]
    else:
        vec = out.last_hidden_state.mean(1)[0]
    return vec.detach().cpu().numpy().astype("float32")

def compute_embeddings(df: pd.DataFrame, frames_root: Path, cache_path: Path,
                       model_name:str, device:str, num_frames:int, max_imgs:int,
                       auto_fps: Optional[float])->np.ndarray:
    if cache_path.exists():
        try:
            cache=np.load(cache_path,allow_pickle=True)
            if (cache.get("model")==model_name and cache.get("num_frames")==num_frames
                and list(df.action_id)==cache["ids"].tolist()):
                E=cache["emb"]
                if isinstance(E,np.ndarray) and E.ndim==2:
                    log(f"Use cached embeddings: {cache_path} shape={E.shape}")
                    return E
        except Exception:
            log("Cache load failed; recompute...", level="WARN")

    proc=AutoImageProcessor.from_pretrained(model_name)
    back=VideoMAEModel.from_pretrained(model_name).to(device).eval()

    embs=[]
    with Timer(f"Compute embeddings (N={len(df)}, frames={num_frames})"):
        for i,(_,row) in enumerate(df.iterrows(),1):
            od=ensure_frames(row.action_id, row.video1, row.video2, frames_root,
                             auto_fps if auto_fps and auto_fps>0 else None)
            img_paths = list_images(od)
            if max_imgs and max_imgs>0 and len(img_paths)>max_imgs:
                # 先裁上限，再均匀采样到 NUM_FRAMES
                idx = np.linspace(0, len(img_paths) - 1, max_imgs, dtype=int)
                img_paths = [img_paths[j] for j in idx]
            vec = videomae_embed(img_paths, proc, back, device, num_frames)
            embs.append(vec)
            if (i%LOG_EVERY)==0 or i==len(df):
                log(f"Embeddings {i}/{len(df)} (id={row.action_id}, imgs={len(img_paths)})")

    E=np.stack(embs,axis=0).astype("float32")
    cache_path.parent.mkdir(parents=True,exist_ok=True)
    with Timer(f"Save cache {cache_path}"):
        np.savez_compressed(cache_path, emb=E, ids=df.action_id.values,
                            model=model_name, num_frames=num_frames)
    return E

class Heads(nn.Module):
    def __init__(self, in_dim:int, hid:int=512, p:float=0.2):
        super().__init__()
        self.mlp=nn.Sequential(nn.Linear(in_dim,hid), nn.ReLU(True), nn.Dropout(p))
        self.hf=nn.Linear(hid,2)   # foul
        self.hc=nn.Linear(hid,3)   # card: None/Red/Yellow
        self.ha=nn.Linear(hid,2)   # advantage
    def forward(self,x):
        z=self.mlp(x)
        return self.hf(z), self.hc(z), self.ha(z)

def warn_single_class(name, y):
    cls = np.unique(y)
    if cls.size < 2:
        log(f"[WARN] {name}: test set contains a single class {cls.tolist()}. Metrics can be misleading.", level="WARN")

def train_and_eval(E: np.ndarray, df: pd.DataFrame, device:str):
    
    with Timer("Standardize features"):
        scaler=StandardScaler(with_mean=True,with_std=True)
        X=scaler.fit_transform(E).astype("float32")

    with Timer("Prepare labels"):
        y_f=df.y_foul.astype(int).values
        y_a=df.y_adv.astype(int).values
        le=LabelEncoder()
        y_c=le.fit_transform(df.y_card.astype(str).values)
        has_card=(df.y_card.isin(["Yellow","Red"])).astype(int).values
        card_classes=list(le.classes_)
        log(f"Card classes: {card_classes}")

    with Timer("Stratified split"):
        combo=y_f*2+has_card
        sss=StratifiedShuffleSplit(n_splits=1,test_size=TEST_SIZE,random_state=RANDOM_STATE)
        (tr,te),=sss.split(X,combo)
        Xtr,Xte=X[tr],X[te]
        y_f_tr,y_f_te=y_f[tr],y_f[te]
        y_c_tr,y_c_te=y_c[tr],y_c[te]
        y_a_tr,y_a_te=y_a[tr],y_a[te]
        log(f"Train/Test sizes: {Xtr.shape}/{Xte.shape}")

    in_dim=X.shape[1]
    net=Heads(in_dim).to(device)
    opt=torch.optim.AdamW(net.parameters(), lr=LR)
    ce =nn.CrossEntropyLoss()

    Xt=torch.from_numpy(Xtr).to(device)
    yft=torch.from_numpy(y_f_tr).long().to(device)
    yct=torch.from_numpy(y_c_tr).long().to(device)
    yat=torch.from_numpy(y_a_tr).long().to(device)

    with Timer(f"Train heads (epochs={EPOCHS})"):
        net.train()
        for ep in range(1,EPOCHS+1):
            opt.zero_grad()
            lf,lc,la = net(Xt)
            loss = ce(lf,yft) + ce(lc,yct) + ce(la,yat)
            loss.backward(); opt.step()
            log(f"epoch {ep}/{EPOCHS}  loss={loss.item():.4f}")

    net.eval()
    Xte_t = torch.from_numpy(Xte).to(device)
    with torch.no_grad():
        pf, pc, pa = net(Xte_t)
        y_f_pred = pf.argmax(1).cpu().numpy()
        y_c_pred = pc.argmax(1).cpu().numpy()
        y_a_pred = pa.argmax(1).cpu().numpy()

    print("\n=== Head-1 (Is Foul) ===")
    labels_bin = [0,1]
    warn_single_class("Foul", y_f_te)
    print(classification_report(y_f_te, y_f_pred, labels=labels_bin,
                                target_names=["No Foul","Foul"], digits=4, zero_division=0))
    print("Confusion Matrix:\n", confusion_matrix(y_f_te, y_f_pred, labels=labels_bin))

    print(f"\n=== Head-2 (Card: classes={card_classes}) ===")
    labels_card = np.arange(len(card_classes))
    print(classification_report(y_c_te, y_c_pred, labels=labels_card,
                                target_names=card_classes, digits=4, zero_division=0))
    print("Confusion Matrix:\n", confusion_matrix(y_c_te, y_c_pred, labels=labels_card))

    print("\n=== Head-3 (Advantage) ===")
    warn_single_class("Advantage", y_a_te)
    print(classification_report(y_a_te, y_a_pred, labels=labels_bin,
                                target_names=["No","Yes"], digits=4, zero_division=0))
    print("Confusion Matrix:\n", confusion_matrix(y_a_te, y_a_pred, labels=labels_bin))

def pick_device()->str:
    if hasattr(torch.backends,"mps") and torch.backends.mps.is_available(): return "mps"
    if torch.cuda.is_available(): return "cuda"
    return "cpu"

def run():
    device = pick_device()
    log(f"Device: {device}")
    data = load_dataset(DATA_JSON)
    df   = build_df(data)
    E    = compute_embeddings(df, FRAMES_ROOT, CACHE_PATH, PRETRAINED, device,
                              NUM_FRAMES, MAX_IMGS,
                              AUTO_EXTRACT_FPS if AUTO_EXTRACT_FPS and AUTO_EXTRACT_FPS>0 else None)
    train_and_eval(E, df, device)

if __name__ == "__main__":
    run()
