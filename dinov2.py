import os, json
from pathlib import Path
import joblib
import numpy as np
import torch
from PIL import Image
import matplotlib.pyplot as plt
import timm
import torchvision.transforms as T
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, accuracy_score
from sklearn.utils.class_weight import compute_sample_weight

from frame_extraction import extract_frames_from_video
def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def plot_confusion_matrix(matrix, labels, title):
    fig, ax = plt.subplots(figsize=(4,4))
    cax = ax.matshow(matrix, cmap="Blues")
    fig.colorbar(cax)

    ax.set_xticks(np.arange(len(labels)))
    ax.set_yticks(np.arange(len(labels)))
    ax.set_xticklabels(labels)
    ax.set_yticklabels(labels)

    total = matrix.sum()

    for i in range(len(matrix)):
        for j in range(len(matrix[i])):
            count = matrix[i][j]
            perc = 100 * count / total if total > 0 else 0
            ax.text(
                j, i,
                f"{count}\n({perc:.1f}%)",
                ha="center", va="center",
                fontsize=10, fontweight="bold",  color="red"
            )

    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title(title, fontsize=11, fontweight="bold")
    plt.tight_layout()
    plt.show()

def embed_action(action_id, record):
    num_segments = 5
    video_url = record["video2"]
    extract_frames_from_video(video_url, output_dir="frames_output", frames_per_second=10)
    frames_dir = Path("frames_output")
    image_paths = []

    for pattern in ("*.jpg","*.jpeg","*.png","*.bmp","*.webp"):
        image_paths += sorted(frames_dir.glob(pattern))
    images = [transform(Image.open(p).convert("RGB")) for p in image_paths]
    images = torch.stack(images).to(device)

    with torch.no_grad():
        feats = model(images)  # [N, D]
    feats_np = feats.cpu().numpy()
    T_len, D = feats_np.shape
    segments = np.array_split(feats_np, num_segments, axis=0)
    segment_means = []

    for segment in segments:
        if len(segment) > 0:
            segment_means.append(segment.mean(axis=0))
        else:
            segment_means.append(np.zeros(D, dtype="float32"))
    segment_means = np.concatenate(segment_means, axis=0)  # (num_segments * D,)

    global_mean = feats_np.mean(axis=0)        # (D,)
    global_std = feats_np.std(axis=0)          # (D,)

    # [seg1_mean, ..., segK_mean, global_mean, global_std]
    return np.concatenate([segment_means, global_mean, global_std], axis=0)

def load_or_build_cache(cache_path, json_path):
    if os.path.exists(cache_path):
        print(f"Loading cached dataset: {cache_path}")
        return joblib.load(cache_path)

    print(f"Building dataset from: {json_path}")
    data = build_dataset(json_path)
    joblib.dump(data, cache_path)
    print(f"Saved cache to: {cache_path}")
    return data

def build_dataset(json_path):
    data = load_json(json_path)
    features, fouls, cards, advantages = [], [], [], []
    count = 1

    for action_id, record in data.items():
        features.append(embed_action(action_id, record))
        fouls.append(bool(record["foul"][0]))
        cards.append(str(record["card"][0]))
        advantages.append(bool(record["advantage"][0]))
        print(f"Processed {count}/{len(data)} actions", end="\r")
        count += 1
    return np.vstack(features), np.array(fouls, dtype=int), np.array(cards), np.array(advantages, dtype=int)

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = timm.create_model("vit_small_patch14_dinov2.lvd142m", pretrained=True, num_classes=0).to(device)
    model.eval()

    transform = T.Compose([
        T.Resize(518),
        T.CenterCrop(518),
        T.ToTensor(),
        T.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])
    ])

    X_foul_train, y_foul_train, _, _ = load_or_build_cache(
        "C:/Users/Evang/Desktop/Work/DS440/cached_X_foul_train.pkl",
        "codes/train_foul.json"
    )

    X_foul_test,  y_foul_test,  _, _ = load_or_build_cache(
        "C:/Users/Evang/Desktop/Work/DS440/cached_X_foul_test.pkl",
        "codes/balanced_data.json"
    )

    X_ca_train, _, y_card_train, y_adv_train = load_or_build_cache(
        "C:/Users/Evang/Desktop/Work/DS440/cached_X_ca_train.pkl",
        "codes/train_card_adv.json"
    )

    X_ca_test,  _, y_card_test,  y_adv_test  = load_or_build_cache(
        "C:/Users/Evang/Desktop/Work/DS440/cached_X_ca_test.pkl",
        "codes/balanced_fouls_data_new.json"
    )

    # Foul model
    foul_mlp = MLPClassifier(
        hidden_layer_sizes=(256, 128),
        activation="relu",
        solver="adam",
        alpha=1e-4,
        early_stopping=True,
        n_iter_no_change=30,
        max_iter=500,
        random_state=42
    )
    foul_mlp.fit(X_foul_train, y_foul_train)
    y_foul_pred = foul_mlp.predict(X_foul_test)
    acc_foul = accuracy_score(y_foul_test, y_foul_pred)

    # Card model (multi-class)
    card_encoder = LabelEncoder()
    y_card_train_enc = card_encoder.fit_transform(y_card_train)
    y_card_test_enc = card_encoder.transform(y_card_test)

    card_mlp = MLPClassifier(
        hidden_layer_sizes=(256, 128),
        activation="relu",
        solver="adam",
        alpha=1e-4,
        early_stopping=True,
        n_iter_no_change=30,
        max_iter=500,
        random_state=42
    )
    card_mlp.fit(X_ca_train, y_card_train_enc)
    y_card_pred_enc = card_mlp.predict(X_ca_test)
    acc_card = accuracy_score(y_card_test_enc, y_card_pred_enc)

    # Advantage model
    adv_sample_weight = compute_sample_weight(class_weight="balanced", y=y_adv_train)

    adv_mlp = MLPClassifier(
        hidden_layer_sizes=(128, 64),
        activation="relu",
        solver="adam",
        alpha=1e-4,
        early_stopping=True,
        n_iter_no_change=30,
        max_iter=500,
        random_state=42
    )

    adv_mlp.fit(X_ca_train, y_adv_train, sample_weight=adv_sample_weight)
    y_adv_pred = adv_mlp.predict(X_ca_test)
    acc_adv = accuracy_score(y_adv_test, y_adv_pred)

    os.makedirs("saved_models", exist_ok=True)

    joblib.dump(foul_mlp, "saved_models/foul_mlp.pkl")
    joblib.dump(card_mlp, "saved_models/card_mlp.pkl")
    joblib.dump(adv_mlp,  "saved_models/adv_mlp.pkl")
    joblib.dump(card_encoder, "saved_models/card_label_encoder.pkl")

    print("Models saved to ./saved_models/")



    # Confusion matrices
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    cm_foul = confusion_matrix(y_foul_test, y_foul_pred)
    plot_confusion_matrix(
        cm_foul,
        ["No Foul", "Foul"],
        f"Foul Confusion Matrix (Acc = {acc_foul:.2f})"
    )
    axes[0].set_title(f"Foul (acc={acc_foul:.2f})")

    cm_card = confusion_matrix(y_card_test_enc, y_card_pred_enc)
    plot_confusion_matrix(
        cm_card,
        card_encoder.classes_,
        f"Card Confusion Matrix (Acc = {acc_card:.2f})"
    )
    axes[1].set_title(f"Card (acc={acc_card:.2f})")

    cm_adv = confusion_matrix(y_adv_test, y_adv_pred)
    plot_confusion_matrix(
        cm_adv,
        ["No Advantage", "Advantage"],
        f"Advantage Confusion Matrix (Acc = {acc_adv:.2f})"
    )
    axes[2].set_title(f"Advantage (acc={acc_adv:.2f})")

    plt.tight_layout()

    # Accuracy bar plot
    plt.figure(figsize=(5,4))
    tasks = ["Foul","Card","Advantage"]
    accuracies = [acc_foul, acc_card, acc_adv]
    plt.bar(tasks, accuracies)
    for i, v in enumerate(accuracies):
        plt.text(i, v + 0.01, f"{v:.2f}", ha="center")
    plt.ylim(0,1)
    plt.ylabel("Accuracy")
    plt.title("Model Accuracy per Task")
    plt.tight_layout()
    plt.show()
