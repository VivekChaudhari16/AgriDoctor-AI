# =====================================================================
# 🌾 AgriDoctor AI - FIELD MODEL TRAINER  (portable script)
# Crops: MAIZE + SOYBEAN + COTTON, trained on REAL FIELD PHOTOS
# ---------------------------------------------------------------------
# BEFORE YOU RUN:  pip install -r requirements.txt   (a GPU is strongly recommended)
#   Run from the repository root:   python training/train_field_model.py
# What it does:
#   1. downloads 8 real-photo datasets (Hugging Face, Project-AgML)
#   2. maps them to ONE label scheme:  Crop___Condition
#   3. removes duplicate photos, balances classes, splits 70/15/15
#   4. trains ResNet-50 with strong augmentation (2 + 12 epochs)
#   5. calibrates the confidence (temperature scaling)
#   6. tests on photos the model never saw, per dataset source
#   7. saves the model + a full report to Google Drive
#   8. (optional) predicts your own photos from  AgriDoctor_AI/my_test_photos
# =====================================================================
import copy
import json
import os
import random
import re
import shutil
import time
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps

# ---------------------------------------------------------------------
# Settings
# ---------------------------------------------------------------------
PROJECT_DIR = Path(os.environ.get("AGRIDOCTOR_HOME", "."))   # models/ and the report are created here
MODEL_DIR = PROJECT_DIR / "models"
REPORT_DIR = PROJECT_DIR / "field_model_report"
TEST_PHOTOS = PROJECT_DIR / "my_test_photos"
WORK = Path(os.environ.get("AGRIDOCTOR_WORK", "agri_field_data"))
MODEL_FILE = MODEL_DIR / "agridoctor_field_model.pth"

SEED = 42
ARCH = "resnet50"
IMG = 224
BATCH = 32
MAX_PER_CLASS = 1200      # cap so no class dominates
MIN_PER_CLASS = 60        # drop classes with too few photos
EPOCHS_HEAD = 2           # train only the new output layer first
EPOCHS_FINE = 12          # then fine-tune the whole network
PATIENCE = 4

random.seed(SEED)
np.random.seed(SEED)

# ---------------------------------------------------------------------
# Datasets. Each rule is (regex on the cleaned label, crop or None, condition or None).
#   crop None      -> use the source's default crop
#   condition None -> skip that class (not a leaf disease / not useful)
# Labels that match no rule are skipped and printed, so nothing is lost silently.
# ---------------------------------------------------------------------
SOURCES = [
    ("Project-AgML/corn_maize_leaf_disease", "Maize", [
        ("health", None, "Healthy"), ("rust", None, "Common rust"),
        ("gr[ae]y", None, "Gray leaf spot"), ("blight", None, "Northern leaf blight")]),
    ("Project-AgML/corn_leaf_disease_classification", "Maize", [
        ("bipolaris", None, "Southern leaf blight"), ("rust", None, "Common rust"),
        ("curvularia", None, "Curvularia leaf spot"), ("northern", None, "Northern leaf blight"),
        ("sheath", None, "Sheath blight"), ("own", None, None)]),
    ("Project-AgML/maize_disease_classification", "Maize", [
        ("health", None, "Healthy"), ("mln", None, "Maize lethal necrosis (MLN)"),
        ("msv", None, "Maize streak virus (MSV)")]),
    ("Project-AgML/plant_doc_classification", "Maize", [
        ("corn.*gr[ae]y", "Maize", "Gray leaf spot"), ("corn.*blight", "Maize", "Northern leaf blight"),
        ("corn.*rust", "Maize", "Common rust"), ("soy", "Soybean", "Healthy")]),
    ("Project-AgML/soybean_leaf_disease_classification", "Soybean", [
        ("dry", None, None), ("root", None, None), ("bacterial", None, "Bacterial blight"),
        ("septoria|brown", None, "Septoria brown spot"), ("vein", None, "Vein necrosis"),
        ("health", None, "Healthy")]),
    ("Project-AgML/cotton_leaf_disease_classification", "Cotton", [
        ("alternaria", None, "Alternaria leaf spot"), ("bacterial", None, "Bacterial blight"),
        ("fusarium", None, "Fusarium wilt"), ("verticillium", None, "Verticillium wilt"),
        ("health", None, "Healthy")]),
    ("Project-AgML/cotton_leaf_disease_classification_bangladesh_2", "Cotton", [
        ("health", None, "Healthy"), ("hopper|jassid", None, "Leaf hopper (jassid) damage"),
        ("redd", None, "Leaf reddening"), ("variegat", None, "Leaf variegation")]),
    ("Project-AgML/cotton_disease_classification", "Cotton", [
        ("alternaria", None, "Alternaria leaf spot"), ("bacterial", None, "Bacterial blight"),
        ("cercospora", None, "Cercospora leaf spot"), ("mildew", None, "Grey mildew"),
        ("health", None, "Healthy")]),
]
# To add more crops (rice, wheat, sugarcane ...): add one line here with a Hugging Face
# dataset that has an image column and a class-label column, then run the cell again.


# ---------------------------------------------------------------------
# Data helpers
# ---------------------------------------------------------------------
def clean_label(name):
    return re.sub(r"[^a-z0-9]+", "_", str(name).lower()).strip("_")


def resolve(label, default_crop, rules):
    """Return (crop, condition), 'SKIP' (deliberately ignored) or None (no rule matched)."""
    text = clean_label(label)
    if text.startswith("aug"):             # augmented copies would leak into the test set
        return "SKIP"
    for pattern, crop, condition in rules:
        if re.search(pattern, text):
            return (crop or default_crop, condition) if condition else "SKIP"
    return None


def dhash(image):
    """64-bit difference hash: identical or re-saved copies of a photo get the same hash."""
    gray = np.asarray(image.convert("L").resize((9, 8), Image.LANCZOS), dtype=np.int16)
    bits = (gray[:, 1:] > gray[:, :-1]).flatten()
    return int("".join("1" if b else "0" for b in bits), 2)


def find_columns(ds):
    image_col = label_col = None
    for name, feature in ds.features.items():
        kind = type(feature).__name__
        if kind == "Image" and image_col is None:
            image_col = name
        if kind == "ClassLabel" and label_col is None:
            label_col = name
    return image_col, label_col


def load_repo(repo):
    from datasets import get_dataset_config_names, load_dataset
    try:
        configs = get_dataset_config_names(repo)
    except Exception:
        configs = []
    keep = [c for c in configs if "aug" not in c.lower()] or configs or [None]
    parts = []
    for config in keep:
        data = load_dataset(repo, config) if config else load_dataset(repo)
        parts.extend(data.values() if hasattr(data, "values") else [data])
    return parts


def ingest(repo, default_crop, rules, rows, seen_hashes, stats, loader=load_repo):
    short = repo.split("/")[-1][:24]
    try:
        parts = loader(repo)
    except Exception as error:
        print(f"  !! could not load {repo}: {type(error).__name__}: {str(error)[:120]}")
        stats["failed"].append(repo)
        return
    taken = Counter()
    for pi, ds in enumerate(parts):
        image_col, label_col = find_columns(ds)
        if image_col is None or label_col is None:
            print(f"  !! {repo}: no image/label columns (found {list(ds.features)})")
            stats["failed"].append(repo)
            continue
        names = ds.features[label_col].names
        mapping = {i: resolve(n, default_crop, rules) for i, n in enumerate(names)}
        for i, n in enumerate(names):
            if mapping[i] is None:
                stats["unmapped"].add((repo, n))
        labels = ds[label_col]
        order = list(range(len(labels)))
        random.shuffle(order)
        for idx in order:
            target = mapping[labels[idx]]
            if target is None or target == "SKIP":
                continue
            cls = f"{target[0]}___{target[1]}"
            if taken[cls] >= MAX_PER_CLASS:
                continue
            image = ds[idx][image_col]
            image = ImageOps.exif_transpose(image).convert("RGB")
            digest = dhash(image)
            if digest in seen_hashes:
                stats["duplicates"] += 1
                continue
            seen_hashes.add(digest)
            image.thumbnail((640, 640))
            folder = WORK / "all" / cls
            folder.mkdir(parents=True, exist_ok=True)
            path = folder / f"{short}_{pi}_{idx}.jpg"
            image.save(path, quality=92)
            rows.append({"path": str(path), "cls": cls, "source": short})
            taken[cls] += 1
    print(f"  {repo.split('/')[-1]:55s} {sum(taken.values()):5d} photos  {dict(taken)}")


def balance_and_split(rows):
    from sklearn.model_selection import train_test_split
    by_class = defaultdict(list)
    for r in rows:
        by_class[r["cls"]].append(r)
    kept = []
    for cls, items in sorted(by_class.items()):
        if len(items) < MIN_PER_CLASS:
            print(f"  dropped (only {len(items)} photos): {cls}")
            for r in items:
                Path(r["path"]).unlink(missing_ok=True)
            continue
        random.shuffle(items)
        for r in items[MAX_PER_CLASS:]:
            Path(r["path"]).unlink(missing_ok=True)
        kept.extend(items[:MAX_PER_CLASS])
    classes = sorted({r["cls"] for r in kept})
    idx = list(range(len(kept)))
    y = [kept[i]["cls"] for i in idx]
    train_idx, rest_idx = train_test_split(idx, test_size=0.30, stratify=y, random_state=SEED)
    y_rest = [kept[i]["cls"] for i in rest_idx]
    val_idx, test_idx = train_test_split(rest_idx, test_size=0.50, stratify=y_rest, random_state=SEED)
    for split, ids in (("train", train_idx), ("val", val_idx), ("test", test_idx)):
        for i in ids:
            kept[i]["split"] = split
    return kept, classes


def fit_temperature(logits, labels):
    """Pick T that makes the confidence honest (lowest log-loss on the validation set)."""
    logits = np.asarray(logits, dtype=np.float64)
    labels = np.asarray(labels)
    best_t, best_nll = 1.0, float("inf")
    for t in np.arange(0.5, 5.01, 0.05):
        z = logits / t
        z = z - z.max(axis=1, keepdims=True)
        log_probs = z - np.log(np.exp(z).sum(axis=1, keepdims=True))
        nll = -log_probs[np.arange(len(labels)), labels].mean()
        if nll < best_nll:
            best_t, best_nll = float(t), float(nll)
    return round(best_t, 2)


# ---------------------------------------------------------------------
# 1-3. Build the dataset
# ---------------------------------------------------------------------
print("\n=== 1. Downloading and mapping datasets ===")
if WORK.exists():
    shutil.rmtree(WORK)
rows, seen_hashes = [], set()
stats = {"failed": [], "unmapped": set(), "duplicates": 0}
for repo, crop, rules in SOURCES:
    ingest(repo, crop, rules, rows, seen_hashes, stats)

print(f"\nDuplicate photos removed: {stats['duplicates']}")
if stats["failed"]:
    print("Datasets that could not be loaded:", stats["failed"])
if stats["unmapped"]:
    print("Labels with no rule (skipped). Add a rule in SOURCES if you want them:")
    for repo, label in sorted(stats["unmapped"]):
        if "plant_doc" not in repo:         # PlantDoc has many other crops on purpose
            print("   ", repo.split("/")[-1], "->", label)
if not rows:
    raise RuntimeError("No images were downloaded. Check the internet / dataset names above.")

print("\n=== 2. Balancing and splitting (70/15/15) ===")
data, CLASSES = balance_and_split(rows)
counts = Counter((r["cls"], r["split"]) for r in data)
print(f"{'class':55s} train   val  test")
for cls in CLASSES:
    print(f"{cls:55s} {counts[(cls, 'train')]:5d} {counts[(cls, 'val')]:5d} {counts[(cls, 'test')]:5d}")
print(f"\nTotal photos: {len(data)}   Classes: {len(CLASSES)}   Crops: "
      f"{sorted({c.split('___')[0] for c in CLASSES})}")
crops_in_model = {c.split("___")[0] for c in CLASSES}

# ---------------------------------------------------------------------
# 4. Train
# ---------------------------------------------------------------------
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from torchvision import models, transforms

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if device.type != "cuda":
    print("\n!! No GPU found. Training will be very slow. Runtime > Change runtime type > T4 GPU")
torch.manual_seed(SEED)

MEAN, STD = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
CLASS_INDEX = {c: i for i, c in enumerate(CLASSES)}


class CenterSquare:
    def __call__(self, image):
        w, h = image.size
        side = min(w, h)
        left, top = (w - side) // 2, (h - side) // 2
        return image.crop((left, top, left + side, top + side))


train_tf = transforms.Compose([
    transforms.RandomResizedCrop(IMG, scale=(0.45, 1.0), ratio=(0.75, 1.33)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(0.2),
    transforms.RandomRotation(25),
    transforms.ColorJitter(0.35, 0.35, 0.3, 0.05),
    transforms.RandomGrayscale(0.05),
    transforms.RandomApply([transforms.GaussianBlur(5, (0.1, 1.5))], p=0.2),
    transforms.ToTensor(),
    transforms.Normalize(MEAN, STD),
    transforms.RandomErasing(0.25),
])
eval_tf = transforms.Compose([
    CenterSquare(), transforms.Resize((IMG, IMG)),
    transforms.ToTensor(), transforms.Normalize(MEAN, STD),
])


class Photos(Dataset):
    def __init__(self, items, transform):
        self.items, self.transform = items, transform

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        item = self.items[i]
        with Image.open(item["path"]) as im:
            image = im.convert("RGB")
        return self.transform(image), CLASS_INDEX[item["cls"]]


split_items = {s: [r for r in data if r["split"] == s] for s in ("train", "val", "test")}
class_counts = Counter(r["cls"] for r in split_items["train"])
sample_weights = [1.0 / class_counts[r["cls"]] for r in split_items["train"]]
sampler = WeightedRandomSampler(sample_weights, num_samples=len(sample_weights), replacement=True)

loaders = {
    "train": DataLoader(Photos(split_items["train"], train_tf), batch_size=BATCH, sampler=sampler,
                        num_workers=2, pin_memory=True, drop_last=True),
    "val": DataLoader(Photos(split_items["val"], eval_tf), batch_size=64, num_workers=2),
    "test": DataLoader(Photos(split_items["test"], eval_tf), batch_size=64, num_workers=2),
}

model = getattr(models, ARCH)(weights="DEFAULT")
model.fc = nn.Linear(model.fc.in_features, len(CLASSES))
model.to(device)
criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
use_amp = device.type == "cuda"
scaler = torch.amp.GradScaler("cuda", enabled=use_amp)


def set_trainable(backbone):
    for name, p in model.named_parameters():
        p.requires_grad = backbone or name.startswith("fc.")


def run_epoch(loader, optimizer=None):
    training = optimizer is not None
    model.train(training)
    total_loss, n = 0.0, 0
    all_logits, all_labels = [], []
    for x, y in loader:
        x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
        with torch.set_grad_enabled(training):
            with torch.autocast(device_type=device.type, enabled=use_amp):
                logits = model(x)
                loss = criterion(logits, y)
        if training:
            optimizer.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        total_loss += loss.item() * len(y)
        n += len(y)
        all_logits.append(logits.detach().float().cpu())
        all_labels.append(y.cpu())
    return total_loss / n, torch.cat(all_logits).numpy(), torch.cat(all_labels).numpy()


def macro_f1(logits, labels):
    from sklearn.metrics import f1_score
    return f1_score(labels, logits.argmax(1), average="macro")


print("\n=== 3. Training ===")
best_f1, best_state, history = -1.0, None, []
start = time.time()
plan = [("head", EPOCHS_HEAD)] + [("fine", EPOCHS_FINE)]
for phase, epochs in plan:
    set_trainable(phase == "fine")
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(params, lr=1e-3 if phase == "head" else 2e-4, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    stale = 0
    for epoch in range(1, epochs + 1):
        train_loss, _, _ = run_epoch(loaders["train"], optimizer)
        val_loss, val_logits, val_labels = run_epoch(loaders["val"])
        scheduler.step()
        f1 = macro_f1(val_logits, val_labels)
        acc = float((val_logits.argmax(1) == val_labels).mean())
        history.append({"phase": phase, "epoch": epoch, "train_loss": train_loss,
                        "val_loss": val_loss, "val_acc": acc, "val_macro_f1": float(f1)})
        mark = ""
        if f1 > best_f1:
            best_f1, best_state, stale, mark = f1, copy.deepcopy(model.state_dict()), 0, "  <- best"
        else:
            stale += 1
        print(f"[{phase} {epoch:2d}/{epochs}] train loss {train_loss:.3f} | val loss {val_loss:.3f} "
              f"| val acc {acc * 100:5.1f}% | val macro-F1 {f1:.3f}{mark}   ({(time.time() - start) / 60:.1f} min)")
        if stale >= PATIENCE:
            print("  early stop: no improvement")
            break

# ---------------------------------------------------------------------
# 5-6. Calibrate and test on unseen photos
# ---------------------------------------------------------------------
print("\n=== 4. Calibration and final test ===")
model.load_state_dict(best_state)
_, val_logits, val_labels = run_epoch(loaders["val"])
temperature = fit_temperature(val_logits, val_labels)
_, test_logits, test_labels = run_epoch(loaders["test"])

from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score

test_pred = test_logits.argmax(1)
test_acc = float(accuracy_score(test_labels, test_pred))
test_f1 = float(f1_score(test_labels, test_pred, average="macro"))
print(f"Temperature T = {temperature}")
print(f"TEST accuracy {test_acc * 100:.2f}%   macro-F1 {test_f1:.3f}   ({len(test_labels)} unseen photos)")
report_text = classification_report(test_labels, test_pred, target_names=CLASSES, digits=3, zero_division=0)
print(report_text)

per_source = {}
for source in sorted({r["source"] for r in split_items["test"]}):
    ids = [i for i, r in enumerate(split_items["test"]) if r["source"] == source]
    per_source[source] = {
        "photos": len(ids),
        "accuracy": float((test_pred[ids] == test_labels[ids]).mean()),
    }
print("Accuracy per dataset source (does it work on every photo style?):")
for source, v in per_source.items():
    print(f"  {source:26s} {v['accuracy'] * 100:5.1f}%  ({v['photos']} photos)")

# ---------------------------------------------------------------------
# 7. Save model + report
# ---------------------------------------------------------------------
MODEL_DIR.mkdir(parents=True, exist_ok=True)
REPORT_DIR.mkdir(parents=True, exist_ok=True)
checkpoint = {
    "arch": ARCH,
    "state_dict": {k: v.cpu() for k, v in best_state.items()},
    "class_names": list(CLASSES),
    "img_size": IMG,
    "mean": MEAN,
    "std": STD,
    "temperature": temperature,
    "test_accuracy": test_acc,
    "test_macro_f1": test_f1,
}
torch.save(checkpoint, MODEL_FILE)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

cm = confusion_matrix(test_labels, test_pred, labels=list(range(len(CLASSES))))
cm_norm = cm / np.maximum(cm.sum(axis=1, keepdims=True), 1)
size = max(8, len(CLASSES) * 0.55)
fig, ax = plt.subplots(figsize=(size, size))
ax.imshow(cm_norm, cmap="Greens", vmin=0, vmax=1)
ax.set_xticks(range(len(CLASSES)))
ax.set_yticks(range(len(CLASSES)))
short_names = [c.replace("___", ": ") for c in CLASSES]
ax.set_xticklabels(short_names, rotation=90, fontsize=7)
ax.set_yticklabels(short_names, fontsize=7)
ax.set_xlabel("Predicted")
ax.set_ylabel("True")
ax.set_title(f"Confusion matrix (test set, accuracy {test_acc * 100:.1f}%)")
fig.tight_layout()
fig.savefig(REPORT_DIR / "confusion_matrix.png", dpi=150)
plt.close(fig)

fig, ax = plt.subplots(figsize=(7, 4))
ax.plot([h["train_loss"] for h in history], label="train loss")
ax.plot([h["val_loss"] for h in history], label="val loss")
ax.set_xlabel("epoch")
ax.legend()
fig.tight_layout()
fig.savefig(REPORT_DIR / "training_curves.png", dpi=150)
plt.close(fig)

metrics = {
    "test_accuracy": test_acc, "test_macro_f1": test_f1, "temperature": temperature,
    "classes": CLASSES, "per_source_accuracy": per_source, "history": history,
    "photos": {s: len(v) for s, v in split_items.items()}, "duplicates_removed": stats["duplicates"],
}
(REPORT_DIR / "metrics.json").write_text(json.dumps(metrics, indent=2))
(REPORT_DIR / "classification_report.txt").write_text(report_text)

card = f"""# AgriDoctor AI - Field Model Card

**Task:** classify leaf photos of {', '.join(sorted(crops_in_model))} into {len(CLASSES)} crop-condition classes.
**Model:** {ARCH} (ImageNet pre-trained), fine-tuned with strong augmentation, label smoothing,
class-balanced sampling and temperature-scaled confidence (T = {temperature}).

## Data
{len(data)} real field/leaf photos from {len(SOURCES)} public datasets (Hugging Face, Project-AgML), split 70/15/15
per class after removing {stats['duplicates']} duplicate photos. Train / val / test: {metrics['photos']}.

## Results on unseen test photos
- Accuracy: **{test_acc * 100:.2f}%**
- Macro F1: **{test_f1:.3f}**
- Per dataset source: {', '.join(f"{k} {v['accuracy'] * 100:.0f}%" for k, v in per_source.items())}

Files: confusion_matrix.png, training_curves.png, classification_report.txt, metrics.json

## Limitations
- Test photos come from the same datasets as training photos, so real-world accuracy on a farmer's
  phone photo can be lower. Test with your own photos (my_test_photos folder).
- Only the classes listed in metrics.json are supported. Results are a decision aid, not a diagnosis.
"""
(REPORT_DIR / "MODEL_CARD.md").write_text(card)
print(f"\nSaved model : {MODEL_FILE}")
print(f"Saved report: {REPORT_DIR}")

# ---------------------------------------------------------------------
# 8. Try your own photos (optional): put photos in AgriDoctor_AI/my_test_photos
# ---------------------------------------------------------------------
if TEST_PHOTOS.exists():
    files = [p for p in sorted(TEST_PHOTOS.iterdir()) if p.suffix.lower() in (".jpg", ".jpeg", ".png", ".webp")]
    model.eval()
    print(f"\n=== Your own photos ({len(files)}) ===")
    for path in files:
        with Image.open(path) as im:
            image = ImageOps.exif_transpose(im).convert("RGB")
        w, h = image.size
        side = max(w, h)
        views = [CenterSquare()(image), ImageOps.pad(image, (side, side), color=(128, 128, 128))]
        batch = torch.stack([eval_tf(v) for v in views]).to(device)
        with torch.no_grad():
            probs = torch.softmax(model(batch).float() / temperature, dim=1).mean(0).cpu().numpy()
        top = probs.argsort()[::-1][:3]
        print(f"{path.name:35s} " + " | ".join(f"{CLASSES[i].replace('___', ': ')} {probs[i] * 100:.0f}%" for i in top))
else:
    print(f"\nTip: create the folder {TEST_PHOTOS} and put your own leaf photos there, then run again.")

print("\nDONE. Now run the AgriDoctor app cell: it will use this model automatically for Maize, Soybean and Cotton.")
