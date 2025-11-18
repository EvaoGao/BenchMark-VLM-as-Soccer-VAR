import os, io, json, time, traceback
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm

import torch
import torch.nn as nn

from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix

from transformers import AutoImageProcessor, VideoMAEModel

REPO_ROOT = Path("/Users/wangshuyi/Desktop/BenchMark-VLM-as-Soccer-VAR-1")
CACHE_DIR = REPO_ROOT / "data"
CACHE_DIR.mkdir(parents=True, exist_ok=True)


from pathlib import Path
import json
def load_dataset(path: Path):
    with Timer(f"Load dataset: {path}"):
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)

JSON_H1 = REPO_ROOT / "balanced_data.json"  
JSON_H23 = REPO_ROOT / "balanced_fouls_data.json" 

FRAMES_ROOT = REPO_ROOT / "frames_output"
CACHE_DIR   = REPO_ROOT / "data"

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

# 选择要训练的 head: "foul" / "card" / "adv" / "all"
HEAD_SELECT = "all"


from datetime import datetime
def now(): return datetime.now().strftime("%Y-%m-%d %H:%M:%S")
def log(msg, level="INFO"): print(f"[{now()}] [{level}] {msg}", flush=True)
def log_exc(msg): print(f"[{now()}] [ERROR] {msg}", flush=True); traceback.print_exc()

class Timer:
    def __init__(self, name): self.name, self.t0 = name, None
    def __enter__(self): self.t0 = time.time(); log(f"{self.name} ...start"); return self
    def __exit__(self, et, ev, tb):
        dt = time.time() - self.t0
        if et: log_exc(f"{self.name} failed after {dt:.2f}s")
        else:  log(f"{self.name} ...done in {dt:.2f}s")

from pathlib import Path
import json

def load_dataset(path: Path):
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"{p}")
    with Timer(f"Load dataset: {p}"):
        with open(p, "r", encoding="utf-8") as f:
            return json.load(f)


try:
    import sys
    sys.path.append(str(REPO_ROOT))
    import frame_extraction as fx
except Exception:
    fx = None

from typing import Any, Dict, List
import numpy as np
import pandas as pd

TRUE_SET  = {"1","true","yes","y","t","foul","advantage","has","on"}
FALSE_SET = {"0","false","no","n","f","none","no_foul","no foul","off","na","","null","none."}

def _as_str(x: Any) -> str:
    if x is None: return ""
    return str(x).strip().lower()

def _coerce_bool(x: Any) -> int:
    """最宽松：bool/int/str/list 都能判；遇到列表按“出现正就算正”"""
    if isinstance(x, (bool, np.bool_)): return int(bool(x))
    if isinstance(x, (int, np.integer)): return int(x != 0)
    if isinstance(x, list):
        for v in x:
            if _coerce_bool(v) == 1: return 1
        return 0
    s = _as_str(x)
    if s in TRUE_SET:  return 1
    if s in FALSE_SET: return 0
    return 0

def normalize_foul(rec: Dict[str, Any]) -> int:
    """兼容: foul / is_foul / has_foul / no_foul / decision / label ..."""
    
    for k in ["foul","is_foul","has_foul"]:
        if k in rec: return _coerce_bool(rec[k])
    if "no_foul" in rec: return int(not _coerce_bool(rec["no_foul"]))

    text_keys = ["decision","call","label","foul_explanation","description"]
    s = " ".join([_as_str(rec.get(k, "")) for k in text_keys])
    if "no foul" in s or "not a foul" in s or "no_foul" in s:
        return 0
    if "foul" in s:
        return 1
    return 0

def normalize_advantage(rec: Dict[str, Any]) -> int:
    """兼容: advantage / is_adv / has_advantage / no_advantage / explanation ..."""
    for k in ["advantage","is_adv","has_advantage"]:
        if k in rec: return _coerce_bool(rec[k])
    if "no_advantage" in rec: return int(not _coerce_bool(rec["no_advantage"]))

    text_keys = ["advantage_explanation","decision","label","description"]
    s = " ".join([_as_str(rec.get(k, "")) for k in text_keys])
    if "no advantage" in s or "no_adv" in s:
        return 0
    if "advantage" in s:
        return 1
    return 0

def reduce_card_label(x: Any) -> str:
    """
    统一到: 'None' / 'Yellow' / 'Red'
    兼容: 'no card','none','yellow card','yc','red','rc', 数字(0/1/2)，列表混合。
    """
    vals = x if isinstance(x, list) else [x]
    got = set()
    for v in vals:
        s = _as_str(v)
        if s in {"", "none", "no", "no card", "no_card", "na", "n/a"}:
            got.add("None")
        elif s in {"y", "yellow", "yellow card", "yc", "caution"} or s == "1":
            got.add("Yellow")
        elif s in {"r", "red", "red card", "rc", "send off", "sending-off", "dismissal"} or s == "2":
            got.add("Red")
        elif s.isdigit():
            if s == "0": got.add("None")
            elif s == "1": got.add("Yellow")
            elif s == "2": got.add("Red")

    if "Red" in got: return "Red"
    if "Yellow" in got: return "Yellow"
    return "None"

def _pick(rec: Dict[str, Any], keys: List[str], default=None):
    for k in keys:
        if k in rec and rec[k]: return rec[k]
    return default

def build_df(data) -> pd.DataFrame:
    """把 JSON（dict 或 list）变成统一 DataFrame：y_foul / y_card / y_adv / video1 / video2"""
    rows: List[Dict[str, Any]] = []
    if isinstance(data, dict):
        items = [(k, v) for k, v in data.items()]
    else:
        items = []
        for v in data:
            k = str(v.get("action_id") or v.get("video_id") or v.get("id") or v.get("path") or len(items))
            items.append((k, v))

    for aid, rec in items:
        rec = dict(rec)
        v1 = _pick(rec, ["video1","clip_0","url1"])
        v2 = _pick(rec, ["video2","clip_1","url2"])

        rows.append({
            "action_id": str(aid),
            "y_foul": normalize_foul(rec),
            "y_card": reduce_card_label(rec.get("card")),
            "y_adv" : normalize_advantage(rec),
            "video1": v1,
            "video2": v2,
        })

    df = pd.DataFrame(rows).drop_duplicates(subset=["action_id"])
    pos_foul = int(df["y_foul"].sum())
    pos_adv  = int(df["y_adv"].sum())
    print(f"[INFO] DF shape: {df.shape} | positives foul/adv = {pos_foul}/{pos_adv}")
    return df



IMG_EXTS = {".jpg",".jpeg",".png",".bmp",".webp"}
def list_images(folder: Path, limit: int)->List[str]:
    if not folder.exists(): return []
    imgs = [str(p) for p in sorted(folder.iterdir()) if p.suffix.lower() in IMG_EXTS]
    return imgs if limit<=0 else imgs[:limit]

def _uniform_sample_paths(paths: List[str], n: int)->List[str]:
    if not paths: raise ValueError("No frames found for this video")
    if len(paths) >= n:
        idx = np.linspace(0, len(paths) - 1, n, dtype=int)
        return [paths[i] for i in idx]
    else:
        return paths + [paths[-1]] * (n - len(paths))

def _load_rgb(paths: List[str]):
    return [Image.open(p).convert("RGB") for p in paths]

def silent_extract(url: Optional[str], out_dir: Path, fps: Optional[float]):
    if not url or fx is None: return
    out_dir.mkdir(parents=True, exist_ok=True)
    if any(out_dir.glob("*.jpg")): return
    try:
        import contextlib
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf), contextlib.redirect_stderr(buf):
            if fps is None: fx.extract_frames_from_video(url, output_dir=str(out_dir))
            else:           fx.extract_frames_from_video(url, output_dir=str(out_dir), frames_per_second=fps)
    except Exception as e:
        log_exc(f"Frame extraction failed for {url}: {e}")

def ensure_frames(action_id: str, v1: Optional[str], v2: Optional[str], root: Path, fps: Optional[float])->Path:
    od = root / action_id
    if not any(od.glob("*.jpg")):
        silent_extract(v1, od, fps); silent_extract(v2, od, fps)
    return od


@torch.no_grad()
def videomae_embed(img_paths, processor, backbone, device, num_frames: int):
    n_req = getattr(backbone.config, "num_frames", num_frames or 16)
    tube  = getattr(backbone.config, "tubelet_size", 2)
    n_req = max(n_req, tube)
    n_req = ((n_req + tube - 1)//tube)*tube  # 对齐 tubelet

    frame_paths = _uniform_sample_paths(img_paths, n_req)
    imgs = _load_rgb(frame_paths)

    batch = processor([imgs], return_tensors="pt")
    batch = {k: v.to(device) for k,v in batch.items()}

    out = backbone(**batch)
    if hasattr(out, "pooler_output") and out.pooler_output is not None:
        vec = out.pooler_output[0]
    else:
        vec = out.last_hidden_state.mean(1)[0]
    return vec.detach().cpu().numpy().astype("float32")

def compute_embeddings(df: pd.DataFrame, frames_root: Path, cache_path: Path,
                       model_name: str, device: str, num_frames: int, max_imgs: int,
                       auto_fps: Optional[float])->np.ndarray:
    if cache_path.exists():
        try:
            cache = np.load(cache_path, allow_pickle=True)
            if (cache.get("model")==model_name and cache.get("num_frames")==num_frames
                and list(df.action_id)==cache["ids"].tolist()):
                E = cache["emb"]
                if isinstance(E, np.ndarray) and E.ndim==2:
                    log(f"Use cached embeddings: {cache_path} shape={E.shape}")
                    return E
        except Exception:
            log("Cache load failed; recompute...", level="WARN")

    proc = AutoImageProcessor.from_pretrained(model_name)
    back = VideoMAEModel.from_pretrained(model_name).to(device).eval()

    embs=[]
    with Timer(f"Compute embeddings (N={len(df)}, frames={num_frames})"):
        for i,(_,row) in enumerate(df.iterrows(), 1):
            od = ensure_frames(row.action_id, row.video1, row.video2, frames_root,
                               auto_fps if auto_fps and auto_fps>0 else None)
            img_paths = list_images(od, max_imgs)
            if not img_paths:
                embs.append(np.zeros((back.config.hidden_size,), dtype="float32"))
            else:
                vec = videomae_embed(img_paths, proc, back, device, num_frames)
                embs.append(vec)
            if (i%LOG_EVERY)==0 or i==len(df):
                log(f"Embeddings {i}/{len(df)} (id={row.action_id}, imgs={len(img_paths)})")

    E = np.stack(embs, axis=0).astype("float32")
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    with Timer(f"Save cache {cache_path}"):
        np.savez_compressed(cache_path, emb=E, ids=df.action_id.values,
                            model=model_name, num_frames=num_frames)
    return E


class SingleHead(nn.Module):
    def __init__(self, in_dim:int, out_dim:int, hid:int=512, p:float=0.2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hid), nn.ReLU(True), nn.Dropout(p),
            nn.Linear(hid, out_dim)
        )
    def forward(self, x): return self.net(x)


def pick_device()->str:
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available(): return "mps"
    if torch.cuda.is_available(): return "cuda"
    return "cpu"


def train_one_head(E: np.ndarray, df: pd.DataFrame, head: str, device: str):
    
    scaler = StandardScaler(with_mean=True, with_std=True)
    X = scaler.fit_transform(E).astype("float32")

    if head == "foul":
        y = df.y_foul.astype(int).values
        labels = [0, 1]
        target_names = ["No Foul", "Foul"]
        title = "Head-1 (Is Foul)"
        out_dim = 2
    elif head == "card":
        le = LabelEncoder()
        y = le.fit_transform(df.y_card.astype(str).values)
        labels = list(range(len(le.classes_)))
        target_names = list(le.classes_)
        title = f"Head-2 (Card: classes={target_names})"
        out_dim = len(labels)
    else:  # adv
        y = df.y_adv.astype(int).values
        labels = [0, 1]
        target_names = ["No", "Yes"]
        title = "Head-3 (Advantage)"
        out_dim = 2

    sss = StratifiedShuffleSplit(n_splits=1, test_size=TEST_SIZE, random_state=RANDOM_STATE)
    (tr, te), = sss.split(X, y)
    Xtr, Xte = X[tr], X[te]
    ytr, yte = y[tr], y[te]

    net = nn.Sequential(
        nn.Linear(X.shape[1], 512),
        nn.ReLU(True),
        nn.Dropout(0.2),
        nn.Linear(512, out_dim),
    ).to(device)
    opt = torch.optim.AdamW(net.parameters(), lr=LR)
    ce = nn.CrossEntropyLoss()

    Xt = torch.from_numpy(Xtr).to(device)
    Yt = torch.from_numpy(ytr).long().to(device)

    log(f"Train {title} (epochs={EPOCHS}) ...start")
    net.train()
    for ep in range(1, EPOCHS + 1):
        opt.zero_grad()
        loss = ce(net(Xt), Yt)
        loss.backward()
        opt.step()
        log(f"epoch {ep}/{EPOCHS}  loss={loss.item():.4f}")
    log(f"Train {title} (epochs={EPOCHS}) ...done")

    net.eval()
    with torch.no_grad():
        pred = net(torch.from_numpy(Xte).to(device)).argmax(1).cpu().numpy()

    print(f"\n=== {title} ===")
    print(classification_report(yte, pred, labels=labels, target_names=target_names,
                                digits=4, zero_division=0))
    print("Confusion Matrix:\n", confusion_matrix(yte, pred, labels=labels))

def _run_one_head(head_name: str, data_path: Path, device: str):
    log(f"===== Run head: {head_name} | data: {data_path.name} =====")
    data = load_dataset(data_path)
    df = build_df(data)

    from pathlib import Path
    tag = Path(data_path).stem  
    cache_path = CACHE_DIR / f"videomae_emb_{tag}_{head_name}_{NUM_FRAMES}.npz"



    train_one_head(E, df, head=head_name, device=device)

def run():
    device = pick_device()
    log(f"Device: {device}")

    HEAD_TO_DATA = {
        "foul": REPO_ROOT / "complete_consistent_data.json",
        "card": REPO_ROOT / "fouls_data.json",
        "adv" : REPO_ROOT / "fouls_data.json",
    }

    for head in ["foul", "card", "adv"]:
        _run_one_head(head, HEAD_TO_DATA[head], device)

if __name__ == "__main__":
    run()
