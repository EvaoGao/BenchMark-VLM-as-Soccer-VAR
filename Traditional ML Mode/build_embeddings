# pip install torch torchvision sentence-transformers
import torch, numpy as np
from PIL import Image
from glob import glob
import torchvision.transforms as T
import torchvision.models as models
from sentence_transformers import SentenceTransformer

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# 1) Image embedding：ResNet50 (2048 dim)
resnet = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2).to(DEVICE).eval()
backbone = torch.nn.Sequential(*list(resnet.children())[:-1]).to(DEVICE)
img_tf = models.ResNet50_Weights.IMAGENET1K_V2.transforms()

@torch.no_grad()
def img_embed(paths):
    if not paths: 
        return np.zeros((2048,), dtype="float32")
    feats = []
    for p in paths:
        x = img_tf(Image.open(p).convert("RGB")).unsqueeze(0).to(DEVICE)  # [1,3,224,224]
        f = backbone(x).squeeze().flatten()  # [2048]
        f = f / (f.norm() + 1e-8)
        feats.append(f.cpu().numpy())
    return np.mean(np.stack(feats, 0), 0).astype("float32")  # mean-pool

# 2) Text embedding：SBERT (all-MiniLM-L6-v2, 384 dim)
txt_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2", device=DEVICE)

def txt_embed(question: str):
    v = txt_model.encode([question], normalize_embeddings=True)
    return v[0].astype("float32")  # [384]

# 3) 拼接成RF
def build_features(image_paths, question):
    v = img_embed(image_paths)        # 2048
    q = txt_embed(question)           # 384
    return np.concatenate([v, q], 0)  # 2432 dim
