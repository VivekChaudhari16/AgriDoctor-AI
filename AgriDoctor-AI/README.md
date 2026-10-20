# 🌾 AgriDoctor AI — Voice-Enabled Crop Leaf Disease Assistant for Farmers

An AI assistant that looks at a photo of a crop leaf, tells the farmer what the problem probably is, and explains **what to do, which medicine is commonly advised, and how to prevent it** — in **English, Marathi and Hindi**, with a **voice assistant** the farmer can talk to.

> ⚠️ **Decision aid, not a diagnosis.** Predictions can be wrong. The app always tells the farmer to confirm with an agriculture officer before spraying anything.

<!-- Add 3-4 screenshots here: home screen, result with confidence ring, voice assistant -->
<!-- ![Home](docs/home.png) ![Result](docs/result.png) -->

---

## ✨ Features

| Area | What it does |
|---|---|
| **Disease detection** | Banana, Maize, Soybean, Cotton (field-photo model) + 9 more crops (PlantVillage model) |
| **Honest confidence** | Calibrated confidence ring; "likely" vs "not sure" states; no disease is named when the AI is unsure |
| **Wrong-crop guard** | If the AI believes the leaf is a *different* crop, it warns instead of guessing |
| **Farmer solutions** | Numbered *What to do*, *Medicine*, *Prevention*, *About* tabs + spray-safety warning |
| **Voice assistant** | Reads the full solution aloud; farmer can **ask by voice** ("Which medicine?", "कोणते औषध?", "कौन सी दवा?") or type |
| **3 languages** | English, मराठी, हिंदी — interface, advice and speech |
| **Farmer-friendly UI** | Camera or upload, ox-plough farm theme, large buttons, works on a phone browser |

---

## 🧠 How it works

```
Leaf photo ──► choose crop ──► model for that crop ──► probabilities (only that crop's classes)
                                    │
                 square views + flips (no squashing) ──► averaged ──► temperature-scaled confidence
                                    │
         wrong-crop check ──► confident / likely / unsure ──► advice (KB) ──► text + voice (gTTS)
```

### Models

| Model | Crops | Architecture | Trained on |
|---|---|---|---|
| **Banana model** | Banana (healthy, Sigatoka, Xanthomonas wilt) | ResNet-18 | Banana leaf dataset (Hugging Face) |
| **Field model** | Maize, Soybean, Cotton | ResNet-50 (ImageNet pre-trained) | Real field photos from 8 public datasets (see below) |
| **PlantVillage model** | Tomato, Potato, Bell pepper, Grape, Apple, Cherry, Peach, Strawberry (+ maize fallback) | ResNet-18 | PlantVillage (38 classes) |

**Field-model training recipe** (`train_field_model.py`)
- 8 datasets mapped to one label scheme `Crop___Condition`
- Duplicate photos removed with a perceptual hash (so train and test don't share the same image)
- Class balancing (cap per class + weighted sampling), stratified 70 / 15 / 15 split
- Strong augmentation: random resized crop, rotation, colour jitter, blur, random erasing
- Two-phase fine-tuning (head only, then whole network), AdamW, cosine schedule, label smoothing 0.1, early stopping on validation macro-F1
- **Temperature scaling** on the validation set so the confidence is honest
- Evaluation per class **and per dataset source**; confusion matrix + model card generated automatically

### Why a separate field model?
The PlantVillage model scores very high on its own studio-style photos but fails on ordinary phone photos (e.g. a maize leaf predicted as strawberry). Training on real field photos, restricting predictions to the farmer's chosen crop, and adding a wrong-crop guard address this.

---

## 📊 Results

> Fill these in from `AgriDoctor_AI/field_model_report/MODEL_CARD.md` after training. Do not publish numbers you have not measured.

| Model | Test accuracy | Macro-F1 | Notes |
|---|---|---|---|
| Field model (Maize / Soybean / Cotton) | _fill in_ | _fill in_ | per-source accuracy: _fill in_ |
| Banana model | _fill in_ | _fill in_ | from the training notebook |
| PlantVillage model | _fill in_ | – | validated on the same studio-photo distribution |

Also add: `confusion_matrix.png`, `training_curves.png`, and results on **your own phone photos** (`my_test_photos`).

---

## 🗂️ Project structure

```
AgriDoctor-AI/
├── app.py                          # Streamlit app (farmer UI, voice assistant, advice)
├── training/
│   └── train_field_model.py        # builds the dataset + trains the Maize/Soybean/Cotton model
├── notebooks/
│   └── Project.ipynb               # exploration + banana and PlantVillage model training
├── colab/
│   ├── agridoctor_app_single_cell.py   # whole app in ONE Colab cell (+ ngrok link)
│   └── train_field_model_colab.py      # field-model trainer as ONE Colab cell
├── models/                         # put trained .pth weights here (not tracked by Git)
├── docs/screenshots/               # add your screenshots for this README
├── .streamlit/config.toml          # app theme
├── .vscode/                        # VS Code run/debug settings
├── requirements.txt
├── LICENSE
└── README.md
```

---

## 🚀 Run it

### Option A - On your computer (VS Code)
```bash
git clone https://github.com/<VivekChaudhari16>/AgriDoctor-AI.git
cd AgriDoctor-AI
python -m venv .venv
.venv\Scripts\activate            # Windows      (Mac/Linux: source .venv/bin/activate)
pip install -r requirements.txt
# install ffmpeg too (needed for voice input), then:
python training/train_field_model.py     # once; GPU recommended -> creates models/agridoctor_field_model.pth
streamlit run app.py
```
In VS Code you can also press **F5** and choose *Streamlit: AgriDoctor app*.

The banana and PlantVillage weights come from `notebooks/Project.ipynb`; copy them into `models/` (see `models/README.md`).

### Option B - Google Colab (easiest, free GPU)
1. *Runtime → Change runtime type → T4 GPU*, paste `colab/train_field_model_colab.py` into one cell and run it.
2. Paste `colab/agridoctor_app_single_cell.py` into one cell, run it, enter your free [ngrok](https://ngrok.com) token (or save it in Colab Secrets as `NGROK_AUTH_TOKEN`) and open the printed `https://` link on your phone. Camera and microphone need `https`.

**Tech stack:** Python · PyTorch · torchvision · Streamlit · Hugging Face `datasets` · scikit-learn · gTTS · SpeechRecognition · ffmpeg · ngrok

---

## 🗣️ Voice assistant

- **Output:** gTTS in English / Marathi / Hindi; long answers are converted in parallel so they start quickly.
- **Input:** browser microphone → converted to 16 kHz mono WAV with ffmpeg → silence check → Google speech recognition (`en-IN`, `mr-IN`, `hi-IN`) → keyword intent matching (*treatment, medicine, prevention, about, everything*).
- **Fallbacks:** answer buttons and a typed question box, plus clear error messages (no sound / no internet / not understood).

---

## 🛡️ Safety design

- No disease is named when confidence is low, or when the AI thinks the leaf is another crop.
- Medicine advice lists commonly recommended products with **"confirm with your agriculture officer and follow the label"**; specific doses are given only where taken from a cited source (banana Sigatoka: Tamil Nadu Agricultural University).
- Spray-safety warning (gloves, mask, timing, waiting period) on every treatment.
- Serious diseases with no cure (e.g. Xanthomonas wilt, citrus greening, wilts) say so clearly and tell the farmer to report them.
- Kisan Call Centre (India) toll-free number **1800-180-1551** shown in the app.

---

## ⚠️ Limitations

- Test photos come from the same datasets as training photos, so accuracy on a farmer's phone photo can be **lower** than the reported test accuracy.
- Best results: **one leaf, filling the frame, plain light background, daylight.** Busy backgrounds reduce accuracy.
- Only the crops and diseases listed above are supported; the PlantVillage crops with a single class are not offered.
- Treatment advice is general guidance written for this project and has **not** been reviewed by an agronomist — have it reviewed before real-world use.
- Voice recognition needs internet (Google speech service).

---

## 🔭 Future work

- Add rice, wheat, sugarcane and groundnut (one line per dataset in `SOURCES`)
- Grad-CAM heat-maps to show *where* the AI looked
- Collect and label real farmer photos for a field-validated test set
- Offline mode (on-device model + offline speech)
- Agronomist review of the advice knowledge base; district-wise weather alerts

---

## 📚 Data & credits

- **PlantVillage** — Hughes & Salathé, *An open access repository of images on plant health…*
- **PlantDoc** — Singh et al., *PlantDoc: A Dataset for Visual Plant Disease Detection*, CoDS-COMAD 2020
- **Project-AgML** datasets on Hugging Face: `corn_maize_leaf_disease`, `corn_leaf_disease_classification` (Anhui), `maize_disease_classification` (Tanzania, Mduma & Mayo), `plant_doc_classification`, `soybean_leaf_disease_classification`, `cotton_leaf_disease_classification` (Ripon et al., East West University), `cotton_leaf_disease_classification_bangladesh_2` (SAR-CLD-2024), `cotton_disease_classification`
- Banana leaf dataset from Hugging Face (add the exact dataset name and citation from your notebook)
- Banana Sigatoka spray guidance: Tamil Nadu Agricultural University (TNAU Agritech Portal)

This repository contains **code only, no datasets**. Check each dataset's licence before redistributing any data or model weights.

---

## 📄 Licence

The code is released under the [MIT Licence](LICENSE).
Datasets and pre-trained weights used in this project have their own licences; this repository contains no data, so check each dataset's licence before redistributing any data or trained weights.

## 👤 Author

**Vivek Chaudhari** · [LinkedIn](https://www.linkedin.com/in/vivek-chaudhari-9653b1371)
