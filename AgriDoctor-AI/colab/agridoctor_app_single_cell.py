# =====================================================================
# 🌾 AgriDoctor AI - FARMER EDITION  (ONE Google Colab cell)
# ---------------------------------------------------------------------
# Run this single cell. It will:
#   1. install the libraries      2. mount Google Drive
#   3. check your saved models    4. write the farm-style app.py
#   5. start Streamlit            6. print a public https URL (ngrok)
#
# The app has:
#   * Farmer background: ox plough, farmer, banana, maize, wheat, sugarcane
#     (put your own photo named background.jpg in the AgriDoctor_AI folder to use it)
#   * Friendly farmer guide who speaks to you
#   * Camera or upload. Models: Banana, FIELD model (Maize, Soybean, Cotton - made by
#     train_field_model.py, used automatically when agridoctor_field_model.pth exists)
#     and the 38-class PlantVillage model for other crops
#   * English / Marathi / Hindi
#   * Full solution: What to do, Medicine, Prevention, About + spray safety
#   * VOICE ASSISTANT: reads the full solution aloud, and you can ASK BY VOICE
#     ("what should I do?", "बचाव कसा करावा?", "क्या करें?")
#   * Safety: no disease is named when the AI is not sure
# =====================================================================
import ast
import subprocess
import sys
import time
from pathlib import Path

# ---------- 1. Install libraries ----------
subprocess.run(
    [sys.executable, "-m", "pip", "install", "-q", "-U",
     "streamlit", "pyngrok", "gTTS", "SpeechRecognition", "requests"],
    check=True,
)

# ---------- 2. Mount Google Drive ----------
from google.colab import drive
drive.mount("/content/drive")

PROJECT_DIR = Path("/content/drive/MyDrive/AgriDoctor_AI")
MODEL_DIR = PROJECT_DIR / "models"
APP_PATH = PROJECT_DIR / "app.py"
LOG_PATH = PROJECT_DIR / "streamlit.log"
PORT = 8501

# ---------- 3. Check that the trained models exist ----------
required = ["banana_resnet18_finetuned_best.pth", "resnet18_finetuned.pth"]
missing = [f for f in required if not (MODEL_DIR / f).exists()]
for f in required:
    print(("OK       " if f not in missing else "MISSING  ") + str(MODEL_DIR / f))
if missing:
    raise FileNotFoundError(
        f"Missing model file(s): {missing}. Run the training cells first."
    )

# ---------- 4. Write the Streamlit app ----------
APP_CODE = r'''
import base64
import hashlib
import html
import math
import re
from concurrent.futures import ThreadPoolExecutor
from io import BytesIO
from pathlib import Path

import streamlit as st
import torch
import torch.nn as nn
from gtts import gTTS
from PIL import Image, ImageOps, ImageStat
from torchvision import models, transforms

try:
    import speech_recognition as sr
except Exception:
    sr = None

st.set_page_config(page_title="AgriDoctor AI", page_icon="🌾", layout="centered")

# ======================================================================
# Settings
# ======================================================================
MODEL_DIR = Path("/content/drive/MyDrive/AgriDoctor_AI/models")
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MIN_SCORE = 0.70       # "sure": top prediction reaches this score ...
LIKELY_SCORE = 0.50    # "likely": shown with a warning when between 50% and 70% ...
MIN_MARGIN = 0.15      # ... and it must lead the second prediction by this much

MODEL_OPTIONS = {
    "banana": {
        "file": "banana_resnet18_finetuned_best.pth",
        "default_classes": ("healthy", "sigatoka", "xanthomonas"),
    },
    "plantvillage": {
        "file": "resnet18_finetuned.pth",
        "default_classes": None,   # class names are stored in the checkpoint
    },
    "field": {                      # trained by train_field_model.py on real field photos
        "file": "agridoctor_field_model.pth",
        "default_classes": None,
    },
}

LANGS = ["English", "मराठी", "हिंदी"]
LI = {name: i for i, name in enumerate(LANGS)}
# Crops the 38-class PlantVillage model can tell apart (crops with at least 2 classes).
# key: (class-name prefix, emoji, (English, मराठी, हिंदी))
PV_CROPS = {
    "corn": ("Corn_(maize)", "🌽", ("Maize (corn)", "मका", "मक्का")),
    "tomato": ("Tomato", "🍅", ("Tomato", "टोमॅटो", "टमाटर")),
    "potato": ("Potato", "🥔", ("Potato", "बटाटा", "आलू")),
    "pepper": ("Pepper,_bell", "🫑", ("Bell pepper", "ढोबळी मिरची", "शिमला मिर्च")),
    "grape": ("Grape", "🍇", ("Grape", "द्राक्ष", "अंगूर")),
    "apple": ("Apple", "🍎", ("Apple", "सफरचंद", "सेब")),
    "cherry": ("Cherry_(including_sour)", "🍒", ("Cherry", "चेरी", "चेरी")),
    "peach": ("Peach", "🍑", ("Peach", "पीच", "आड़ू")),
    "strawberry": ("Strawberry", "🍓", ("Strawberry", "स्ट्रॉबेरी", "स्ट्रॉबेरी")),
}
# Crops handled by the field-photo model (shown only when agridoctor_field_model.pth exists)
FIELD_CROPS = {
    "maize": ("Maize", "🌽", ("Maize (corn)", "मका", "मक्का")),
    "soybean": ("Soybean", "🌱", ("Soybean", "सोयाबीन", "सोयाबीन")),
    "cotton": ("Cotton", "☁️", ("Cotton", "कापूस", "कपास")),
}
PREFIX_NAMES = {v[0]: v[2] for v in PV_CROPS.values()}
PREFIX_NAMES.update({
    "Blueberry": ("Blueberry", "ब्लूबेरी", "ब्लूबेरी"),
    "Orange": ("Orange", "संत्रा", "संतरा"),
    "Raspberry": ("Raspberry", "रास्पबेरी", "रास्पबेरी"),
    "Soybean": ("Soybean", "सोयाबीन", "सोयाबीन"),
    "Squash": ("Squash", "स्क्वॅश", "स्क्वैश"),
})
PREFIX_NAMES.update({v[0]: v[2] for v in FIELD_CROPS.values()})

TTS_CODES = {"English": "en", "मराठी": "mr", "हिंदी": "hi"}
SR_CODES = {"English": "en-IN", "मराठी": "mr-IN", "हिंदी": "hi-IN"}

TRANSFORM = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])

# ======================================================================
# Farm background: ox plough + crops (SVG, no internet needed).
# To use your OWN photo instead, save it as background.jpg (or .png / .webp)
# inside the AgriDoctor_AI folder on Google Drive.
# ======================================================================
def ox(body, dark, patch):
    return f"""
<path d="M24 62 Q2 80 10 116" stroke="{dark}" stroke-width="5" fill="none" stroke-linecap="round"/>
<ellipse cx="10" cy="120" rx="6" ry="9" fill="#3e2a1a"/>
<rect x="34" y="88" width="15" height="44" rx="6" fill="{dark}"/>
<rect x="64" y="92" width="15" height="40" rx="6" fill="{body}"/>
<rect x="150" y="92" width="15" height="40" rx="6" fill="{body}"/>
<rect x="176" y="88" width="15" height="44" rx="6" fill="{dark}"/>
<rect x="33" y="129" width="17" height="9" rx="3" fill="#2d2013"/>
<rect x="63" y="129" width="17" height="9" rx="3" fill="#2d2013"/>
<rect x="149" y="129" width="17" height="9" rx="3" fill="#2d2013"/>
<rect x="175" y="129" width="17" height="9" rx="3" fill="#2d2013"/>
<ellipse cx="105" cy="68" rx="88" ry="38" fill="{body}"/>
<ellipse cx="84" cy="76" rx="32" ry="17" fill="{patch}" opacity=".85"/>
<ellipse cx="152" cy="36" rx="30" ry="20" fill="{dark}"/>
<path d="M170 52 Q214 40 232 70 L238 96 Q214 112 188 102Z" fill="{body}"/>
<ellipse cx="238" cy="92" rx="14" ry="11" fill="#efd3b5"/>
<path d="M176 100 Q196 130 214 100Z" fill="{dark}"/>
<path d="M205 52 Q198 22 174 14 Q206 20 216 50Z" fill="#f6efd6"/>
<ellipse cx="195" cy="62" rx="14" ry="6" transform="rotate(-30 195 62)" fill="{dark}"/>
<circle cx="219" cy="72" r="3.4" fill="#1b1b1b"/>
<circle cx="241" cy="90" r="2" fill="#5d4037"/>
"""

DEFS = f"""
<g id="oxA">{ox('#c98f5a', '#9a6538', '#f3e5d2')}</g>
<g id="oxB">{ox('#ece0cf', '#c9ab88', '#c98f5a')}</g>
<g id="banana">
 <rect x="-5" y="-80" width="10" height="80" rx="4" fill="#8bc34a"/>
 <path d="M0 -80 Q-55 -120 -105 -70 Q-55 -92 0 -72Z" fill="#43a047"/>
 <path d="M0 -80 Q55 -120 105 -70 Q55 -92 0 -72Z" fill="#4caf50"/>
 <path d="M0 -80 Q-30 -150 -70 -150 Q-35 -118 0 -76Z" fill="#388e3c"/>
 <path d="M0 -80 Q30 -150 70 -150 Q35 -118 0 -76Z" fill="#66bb6a"/>
 <path d="M0 -80 Q-5 -140 0 -168 Q6 -140 0 -80Z" fill="#2e7d32"/>
 <path d="M8 -70 Q24 -62 18 -38" stroke="#6d4c41" stroke-width="3" fill="none"/>
 <ellipse cx="9" cy="-36" rx="5" ry="11" fill="#fdd835"/>
 <ellipse cx="18" cy="-34" rx="6" ry="13" fill="#fbc02d"/>
 <ellipse cx="28" cy="-38" rx="6" ry="12" fill="#fdd835"/>
</g>
<g id="maize">
 <rect x="-3" y="-95" width="6" height="95" fill="#7cb342"/>
 <path d="M0 -30 Q-50 -55 -66 -20 Q-34 -38 0 -22Z" fill="#558b2f"/>
 <path d="M0 -55 Q50 -80 70 -42 Q36 -62 0 -46Z" fill="#689f38"/>
 <path d="M0 -78 Q-44 -100 -58 -64 Q-30 -82 0 -68Z" fill="#558b2f"/>
 <ellipse cx="10" cy="-52" rx="7" ry="19" fill="#fbc02d"/>
 <path d="M10 -70 Q23 -52 12 -33 Q1 -52 10 -70Z" fill="#9ccc65" opacity=".85"/>
 <path d="M0 -95 l-10 -20 M0 -95 l0 -26 M0 -95 l10 -20" stroke="#d4b44a" stroke-width="3" stroke-linecap="round"/>
</g>
<g id="wheat">
 <path d="M0 0 Q2 -50 0 -90" stroke="#c9a227" stroke-width="3" fill="none"/>
 <ellipse cx="0" cy="-100" rx="5" ry="15" fill="#e6b422"/>
 <ellipse cx="-6" cy="-92" rx="4" ry="11" fill="#d9a521" transform="rotate(-25 -6 -92)"/>
 <ellipse cx="6" cy="-92" rx="4" ry="11" fill="#d9a521" transform="rotate(25 6 -92)"/>
 <path d="M0 -112 l0 -14 M-3 -108 l-4 -12 M3 -108 l4 -12" stroke="#c9a227" stroke-width="1.6"/>
</g>
<g id="cane">
 <rect x="-5" y="-130" width="10" height="130" fill="#b5cc5a"/>
 <path d="M-5 -30 H5 M-5 -60 H5 M-5 -90 H5 M-5 -118 H5" stroke="#6d8f2b" stroke-width="2.5"/>
 <path d="M0 -130 Q-55 -150 -84 -112 Q-40 -132 0 -120Z" fill="#7cb342"/>
 <path d="M0 -130 Q55 -150 84 -112 Q40 -132 0 -120Z" fill="#8bc34a"/>
 <path d="M0 -130 Q-6 -160 -2 -180 Q8 -158 0 -128Z" fill="#689f38"/>
</g>
<g id="hut">
 <rect x="0" y="30" width="84" height="52" fill="#d7b98c"/>
 <polygon points="-14,34 42,-14 98,34" fill="#a1887f"/>
 <rect x="34" y="48" width="18" height="34" fill="#6d4c41"/>
</g>
<g id="hay">
 <path d="M0 0 Q5 -55 40 -60 Q75 -55 80 0Z" fill="#e0b84c"/>
 <path d="M10 -12 Q40 -34 70 -12" stroke="#c99a2e" fill="none" stroke-width="3"/>
</g>
<g id="tree">
 <rect x="-6" y="-50" width="12" height="50" fill="#6d4c41"/>
 <circle cx="0" cy="-72" r="38" fill="#4e9a2f"/>
 <circle cx="-26" cy="-52" r="26" fill="#43882a"/>
 <circle cx="26" cy="-52" r="26" fill="#43882a"/>
</g>
<g id="team">
 <path d="M-40 4 Q150 -4 330 4" stroke="#6d4a2b" stroke-width="10" fill="none" opacity=".35"/>
 <g transform="translate(228 -146) scale(.97)"><use href="#oxB"/></g>
 <g transform="translate(262 -138)"><use href="#oxA"/></g>
 <rect x="372" y="-122" width="86" height="9" rx="4.5" fill="#7b5a2e" transform="rotate(-5 372 -122)"/>
 <path d="M386 -116 L84 -52" stroke="#7b5a2e" stroke-width="9" stroke-linecap="round"/>
 <path d="M82 -58 L62 -14 L24 -3 L74 -3 L92 -12Z" fill="#5d4037"/>
 <path d="M22 0 L72 0 L58 -11Z" fill="#9e9e9e"/>
 <path d="M70 -48 L14 -112" stroke="#7b5a2e" stroke-width="7" stroke-linecap="round"/>
 <rect x="-34" y="-62" width="12" height="62" rx="5" fill="#f5f0e6"/>
 <rect x="-16" y="-62" width="12" height="62" rx="5" fill="#ebe4d3"/>
 <rect x="-40" y="-122" width="38" height="66" rx="10" fill="#fffdf6"/>
 <path d="M-8 -108 L12 -112" stroke="#8d5524" stroke-width="7" stroke-linecap="round"/>
 <circle cx="-21" cy="-139" r="14" fill="#8d5524"/>
 <ellipse cx="-21" cy="-151" rx="19" ry="10" fill="#e53935"/>
 <rect x="-40" y="-4" width="16" height="6" rx="3" fill="#4e342e"/>
 <rect x="-20" y="-4" width="16" height="6" rx="3" fill="#4e342e"/>
</g>
"""

def hero_svg():
    stalks = "".join(
        f'<use href="#wheat" transform="translate({30 + i*20} {208 + (i%3)*5}) scale(.8)" />' for i in range(7))
    crops = (
        '<use href="#banana" transform="translate(615 218) scale(.5)"/>'
        '<use href="#maize" transform="translate(690 220) scale(.55)"/>'
        '<use href="#maize" transform="translate(722 214) scale(.5)"/>'
        '<use href="#cane" transform="translate(770 220) scale(.5)"/>'
        '<use href="#cane" transform="translate(796 214) scale(.46)"/>'
        '<use href="#banana" transform="translate(838 224) scale(.5)"/>'
    )
    return f"""<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 860 240" preserveAspectRatio="xMidYMid slice">
<defs>
<linearGradient id="hs" x1="0" y1="0" x2="0" y2="1"><stop offset="0" stop-color="#8fd3f4"/><stop offset=".6" stop-color="#d6f0fb"/><stop offset="1" stop-color="#fff0bd"/></linearGradient>
<radialGradient id="hsun"><stop offset="0" stop-color="#fffde7"/><stop offset=".4" stop-color="#ffe066"/><stop offset="1" stop-color="#ffe066" stop-opacity="0"/></radialGradient>
{DEFS}
</defs>
<style>.walk{{animation:walk 12s ease-in-out infinite alternate}}@keyframes walk{{from{{transform:translateX(0)}}to{{transform:translateX(30px)}}}}</style>
<rect width="860" height="240" fill="url(#hs)"/>
<circle cx="110" cy="62" r="62" fill="url(#hsun)"/>
<g fill="#fff" opacity=".9"><ellipse cx="420" cy="40" rx="48" ry="13"/><ellipse cx="450" cy="30" rx="30" ry="14"/><ellipse cx="620" cy="62" rx="40" ry="11"/></g>
<path d="M0 130 Q200 78 420 122 T860 112 V240 H0Z" fill="#a5d36b"/>
<path d="M0 158 Q260 112 520 150 T860 140 V240 H0Z" fill="#84bd46"/>
<use href="#hut" x="690" y="96" transform="translate(0 0) scale(.7)"/>
<use href="#tree" x="620" y="140" transform="translate(0 0) scale(.6)"/>
<use href="#hay" x="440" y="170" transform="translate(0 0) scale(.6)"/>
<path d="M0 196 Q430 172 860 196 V240 H0Z" fill="#8d6238"/>
<path d="M0 214 Q430 192 860 214 M0 228 Q430 210 860 228" stroke="#6d4a2b" stroke-width="3" fill="none" opacity=".6"/>
{stalks}
{crops}
<g class="walk"><use href="#team" transform="translate(150 226) scale(.72)"/></g>
<path d="M300 40 q8 -8 16 0 q8 -8 16 0 M340 62 q6 -6 12 0 q6 -6 12 0" stroke="#37474f" stroke-width="2" fill="none"/>
</svg>"""

def bg_svg(animate=True):
    # wheat band + crops
    wheat = "".join(f'<use href="#wheat" x="{x}" y="{660 + (x//40)%3*6}" transform="translate(0 0)"/>' for x in range(300, 1300, 38))
    right = ""
    for i, x in enumerate(range(1150, 1600, 95)):
        right += f'<use href="#banana" x="{x}" y="{760 + (i%2)*20}" transform="translate(0 0)"/>'
    for x in range(1120, 1600, 48):
        right += f'<use href="#maize" x="{x}" y="{845 + (x//48)%2*8}"/>'
    cane = "".join(f'<use href="#cane" x="{x}" y="{700 + (x//40)%2*10}"/>' for x in range(1180, 1600, 42))
    left_crops = "".join(f'<use href="#maize" x="{x}" y="872"/>' for x in range(860, 1100, 52))
    rows = "".join(f'<path d="M800 640 L{x} 900" stroke="#2e6b1a" stroke-width="6" opacity=".35"/>' for x in range(-600, 2300, 300))
    style = """<style>
.c1{animation:drift 70s linear infinite}.c2{animation:drift 105s linear infinite;animation-delay:-45s}
@keyframes drift{from{transform:translateX(-350px)}to{transform:translateX(1950px)}}
.sun{animation:pulse 7s ease-in-out infinite;transform-origin:260px 150px}
@keyframes pulse{50%{transform:scale(1.1)}}
.walk{animation:walk 22s ease-in-out infinite alternate}
@keyframes walk{from{transform:translateX(0)}to{transform:translateX(70px)}}
</style>""" if animate else ""
    return f"""<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 1600 900" preserveAspectRatio="xMidYMax slice">
<defs>
<linearGradient id="sky" x1="0" y1="0" x2="0" y2="1"><stop offset="0" stop-color="#78c6ef"/><stop offset=".55" stop-color="#cfeefb"/><stop offset="1" stop-color="#fff0bd"/></linearGradient>
<radialGradient id="sun"><stop offset="0" stop-color="#fffde7"/><stop offset=".35" stop-color="#ffe066"/><stop offset="1" stop-color="#ffe066" stop-opacity="0"/></radialGradient>
{DEFS}
</defs>
{style}
<rect width="1600" height="900" fill="url(#sky)"/>
<circle class="sun" cx="260" cy="150" r="170" fill="url(#sun)"/>
<g class="c1" fill="#fff" opacity=".92"><ellipse cx="0" cy="150" rx="95" ry="28"/><ellipse cx="55" cy="128" rx="62" ry="30"/><ellipse cx="-55" cy="132" rx="52" ry="22"/></g>
<g class="c2" fill="#fff" opacity=".85"><ellipse cx="0" cy="290" rx="80" ry="22"/><ellipse cx="45" cy="272" rx="52" ry="24"/></g>
<path d="M0 520 Q300 400 620 500 T1200 470 T1600 510 V900 H0Z" fill="#a5d36b"/>
<use href="#hut" x="130" y="470"/><use href="#tree" x="260" y="540"/><use href="#tree" x="1330" y="520" transform="scale(.9)"/>
<path d="M0 600 Q420 490 820 590 T1600 570 V900 H0Z" fill="#82bb45"/>
<use href="#hay" x="1000" y="640"/><use href="#hay" x="1090" y="650" transform="scale(.8)"/>
<path d="M0 668 Q800 612 1600 668 V900 H0Z" fill="#6aa334"/>
{wheat}
{rows}
{cane}
<path d="M0 760 Q800 722 1600 760 V900 H0Z" fill="#8d6238"/>
<path d="M0 790 Q800 756 1600 790 M0 826 Q800 794 1600 826 M0 862 Q800 832 1600 862" stroke="#6d4a2b" stroke-width="4" fill="none" opacity=".55"/>
{left_crops}
{right}
<g class="walk"><use href="#team" transform="translate(70 842) scale(1.15)"/></g>
<path d="M700 120 q8 -8 16 0 q8 -8 16 0 M760 150 q6 -6 12 0 q6 -6 12 0 M1000 90 q8 -8 16 0 q8 -8 16 0" stroke="#37474f" stroke-width="2.5" fill="none"/>
</svg>"""


HERO_B64 = base64.b64encode(hero_svg().encode("utf-8")).decode("ascii")


def background_url():
    mimes = {"jpg": "jpeg", "jpeg": "jpeg", "png": "png", "webp": "webp"}
    for ext, mime in mimes.items():
        path = MODEL_DIR.parent / f"background.{ext}"
        if path.exists():
            data = base64.b64encode(path.read_bytes()).decode("ascii")
            return f"data:image/{mime};base64,{data}"
    svg = base64.b64encode(bg_svg().encode("utf-8")).decode("ascii")
    return f"data:image/svg+xml;base64,{svg}"


BG_URL = background_url()

CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Poppins:wght@400;500;600;700&family=Noto+Sans+Devanagari:wght@400;600;700&display=swap');
html, body, [class*="css"], .stMarkdown, button, label, input, textarea {
    font-family: 'Poppins', 'Noto Sans Devanagari', 'Segoe UI', sans-serif;
}
.stApp { background: url("__BG__") center bottom / cover no-repeat fixed; }
@media (max-width: 800px) { .stApp { background-position: 14% bottom; } }
[data-testid="stHeader"] { background: transparent; }
.block-container, [data-testid="stMainBlockContainer"] {
    max-width: 820px; padding: 1rem 1rem 3rem; background: transparent;
}
/* every section is its own white card, so the farm scene stays bright between them */
[class*="st-key-card"] {
    background: rgba(255, 255, 255, .94); border-radius: 22px; padding: 1.1rem 1.25rem;
    margin-bottom: 1rem; border: 1px solid rgba(255, 255, 255, .8);
    box-shadow: 0 10px 30px rgba(20, 60, 20, .25);
}
.st-key-card_head { padding: 0 !important; overflow: hidden; }
.brand {
    display: flex; align-items: center; gap: .9rem; padding: .95rem 1.25rem; color: #fff;
    background: linear-gradient(135deg, #1b5e20, #2e7d32 55%, #66bb6a);
}
.brand .logo { font-size: 2.3rem; line-height: 1; }
.brand h1 { margin: 0; padding: 0; font-size: 1.35rem; font-weight: 700; color: #fff; line-height: 1.25; }
.brand p { margin: .1rem 0 0; font-size: .85rem; opacity: .92; color: #fff; }
.hero { width: 100%; display: block; }
.farmer-row { display: flex; gap: 14px; align-items: center; }
.farmer { font-size: 3.4rem; line-height: 1; animation: bob 3s ease-in-out infinite; }
@keyframes bob { 50% { transform: translateY(-5px); } }
.bubble {
    position: relative; background: #f1f8e9; border: 2px solid #aed581; border-radius: 16px;
    padding: .7rem .95rem; color: #2f3e1f; font-size: .98rem; line-height: 1.5;
}
.bubble:before {
    content: ""; position: absolute; left: -12px; top: 50%; margin-top: -8px;
    border: 8px solid transparent; border-right: 12px solid #aed581;
}
.sec { display: flex; align-items: center; gap: .6rem; margin: .2rem 0 .55rem; }
.sec .num {
    width: 1.9rem; height: 1.9rem; border-radius: 50%; display: grid; place-items: center;
    background: #2e7d32; color: #fff; font-weight: 700; font-size: .95rem; flex: none;
}
.sec .ttl { font-size: 1.15rem; font-weight: 600; color: #1b3d1c; }
.res { display: flex; align-items: center; gap: 1.1rem; border-radius: 18px; padding: .9rem 1rem;
       border: 2px solid; margin-bottom: .6rem; }
.res.ok { background: #e8f5e9; border-color: #66bb6a; }
.res.bad { background: #ffebee; border-color: #ef5350; }
.res.unsure { background: #fff8e1; border-color: #ffb74d; }
.ring { --p: 0; --c: #2e7d32; width: 88px; height: 88px; border-radius: 50%; flex: none;
        display: grid; place-items: center;
        background: conic-gradient(var(--c) calc(var(--p) * 1%), #dfe6dc 0); }
.ring span { width: 68px; height: 68px; border-radius: 50%; background: #fff; display: grid;
             place-items: center; font-weight: 700; font-size: 1.15rem; color: #263238; }
.pill { display: inline-block; font-size: .8rem; font-weight: 600; padding: .15rem .7rem;
        border-radius: 999px; color: #fff; margin: 0 .35rem .3rem 0; }
.pill.ok { background: #2e7d32; } .pill.bad { background: #d32f2f; }
.pill.unsure { background: #ef6c00; } .pill.likely { background: #f9a825; color: #3b2a00; }
.rname { font-size: 1.35rem; font-weight: 700; color: #1f2d1f; line-height: 1.3; }
.rsub { font-size: .85rem; color: #546e7a; margin-top: .15rem; }
.bars { margin: .3rem 0 .6rem; }
.bars-title { font-size: .85rem; font-weight: 600; color: #455a64; margin-bottom: .35rem; }
.bar-row { display: grid; grid-template-columns: minmax(90px, 38%) 1fr 3rem; gap: .6rem;
           align-items: center; margin: .3rem 0; font-size: .9rem; color: #263238; }
.bar { height: 12px; background: #e3e8e1; border-radius: 999px; overflow: hidden; }
.fill { height: 100%; border-radius: 999px; }
.bar-val { text-align: right; font-weight: 600; }
.also { font-size: .88rem; color: #6d4c00; background: #fff3cd; border-radius: 10px;
        padding: .4rem .7rem; margin: .2rem 0 .6rem; }
.answer {
    background: #e8f5e9; border-left: 8px solid #2e7d32; border-radius: 12px;
    padding: .8rem 1rem; color: #1b3d1c; font-size: 1.02rem; line-height: 1.55; margin: .6rem 0;
}
.stButton > button { border-radius: 999px; font-weight: 600; border: 1.5px solid #2e7d32;
                     color: #1b5e20; background: #fff; padding: .45rem 1.1rem; }
.stButton > button:hover { background: #e8f5e9; border-color: #1b5e20; color: #1b5e20; }
.stButton > button[kind="primary"] {
    background: linear-gradient(135deg, #2e7d32, #43a047); color: #fff; border: none;
    font-size: 1.1rem; padding: .6rem 1.2rem; box-shadow: 0 6px 16px rgba(46, 125, 50, .4);
}
.stButton > button[kind="primary"]:hover { filter: brightness(1.08); color: #fff; }
div[role="radiogroup"] { gap: .5rem; flex-wrap: wrap; }
div[role="radiogroup"] > label {
    background: #f4f8ee; border: 1.5px solid #c5d9a8; border-radius: 999px; padding: .2rem .95rem;
}
div[role="radiogroup"] > label:has(input:checked) { background: #2e7d32; border-color: #1b5e20; }
div[role="radiogroup"] > label:has(input:checked) * { color: #fff !important; }
button[data-baseweb="tab"] { font-weight: 600; }
[data-baseweb="tab-highlight"] { background-color: #2e7d32 !important; }
.st-key-card_lang { padding: .5rem 1rem !important; }
</style>
""".replace("__BG__", BG_URL)

st.markdown(CSS, unsafe_allow_html=True)

# ======================================================================
# Text in three languages: (English, मराठी, हिंदी)
# ======================================================================
UI = {
    "title": ("AgriDoctor AI - Smart Farm Helper",
              "ॲग्रीडॉक्टर एआय - स्मार्ट शेती मदतनीस",
              "एग्रीडॉक्टर एआई - स्मार्ट खेती सहायक"),
    "greet": ("Namaste, farmer friend! Show me a leaf of your crop and I will tell you "
              "what is wrong and what to do. You can also talk to me - just press the "
              "microphone.",
              "नमस्कार शेतकरी मित्रा! मला तुमच्या पिकाचे पान दाखवा, काय चुकले आहे आणि काय "
              "करायचे ते मी सांगतो. तुम्ही माझ्याशी बोलूही शकता - फक्त माईकचे बटण दाबा.",
              "नमस्ते किसान भाई! मुझे अपनी फसल की पत्ती दिखाइए, मैं बताऊँगा कि क्या दिक्कत है "
              "और क्या करना है। आप मुझसे बोलकर भी पूछ सकते हैं - बस माइक का बटन दबाइए।"),
    "hear": ("🔊 Hear greeting", "🔊 स्वागत ऐका", "🔊 स्वागत सुनें"),
    "step1": ("1️⃣ Choose your crop", "1️⃣ तुमचे पीक निवडा", "1️⃣ अपनी फसल चुनें"),
    "crop_banana": ("🍌 Banana", "🍌 केळी", "🍌 केला"),
    "crop_other": ("🍅🥔🌽 Other crops (38 types)", "🍅🥔🌽 इतर पिके (३८ प्रकार)",
                   "🍅🥔🌽 अन्य फसलें (38 प्रकार)"),
    "step2": ("2️⃣ Show the leaf", "2️⃣ पान दाखवा", "2️⃣ पत्ती दिखाएँ"),
    "m_cam": ("📷 Take photo", "📷 फोटो काढा", "📷 फोटो लें"),
    "m_up": ("🖼️ Upload photo", "🖼️ फोटो अपलोड करा", "🖼️ फोटो अपलोड करें"),
    "tip": ("Tip: ONE leaf, filling the frame, on a plain light background (paper or your palm), in daylight, in focus. The AI was trained on clean leaf photos, so busy backgrounds confuse it.",
            "टीप: एकच पान, फ्रेम भरून, साध्या फिकट पार्श्वभूमीवर (कागद किंवा तळहात), दिवसाच्या प्रकाशात, स्पष्ट. एआयला स्वच्छ पानांच्या फोटोंवर शिकवले आहे, त्यामुळे गर्दीची पार्श्वभूमी त्याला गोंधळात टाकते.",
            "सुझाव: एक ही पत्ती, पूरे फ्रेम में, सादे हल्के रंग की पृष्ठभूमि (कागज़ या हथेली) पर, दिन की रोशनी में, साफ। एआई को साफ पत्तियों की फोटो पर सिखाया गया है, इसलिए भीड़ वाली पृष्ठभूमि उसे उलझा देती है।"),
    "step3": ("3️⃣ Check the leaf", "3️⃣ पान तपासा", "3️⃣ पत्ती की जाँच करें"),
    "analyze": ("🔍 Check my leaf", "🔍 माझे पान तपासा", "🔍 मेरी पत्ती जाँचें"),
    "checking": ("Looking at your leaf...", "तुमचे पान पाहत आहे...", "आपकी पत्ती देख रहा हूँ..."),
    "need_photo": ("Please take or upload a leaf photo first.",
                   "कृपया आधी पानाचा फोटो काढा किंवा अपलोड करा.",
                   "कृपया पहले पत्ती की फोटो लें या अपलोड करें।"),
    "result": ("Result", "निकाल", "नतीजा"),
    "good": ("✅ Looks healthy", "✅ निरोगी दिसते", "✅ स्वस्थ दिखती है"),
    "bad": ("⚠️ Possible problem", "⚠️ समस्या असू शकते", "⚠️ समस्या हो सकती है"),
    "unsure": ("❓ Not sure", "❓ खात्री नाही", "❓ पक्का नहीं"),
    "unsure_text": ("I am not sure about this leaf, so I will not name a disease. Take a "
                    "clear, close photo of ONE leaf in daylight and try again. If it still "
                    "looks sick, show it to an agriculture officer.",
                    "या पानाबद्दल मला खात्री नाही, म्हणून मी रोगाचे नाव सांगणार नाही. दिवसाच्या "
                    "प्रकाशात एकाच पानाचा जवळून स्पष्ट फोटो काढून पुन्हा प्रयत्न करा. तरीही पान आजारी "
                    "दिसल्यास कृषी अधिकाऱ्यांना दाखवा.",
                    "इस पत्ती के बारे में मुझे पक्का नहीं पता, इसलिए मैं बीमारी का नाम नहीं बताऊँगा। "
                    "दिन की रोशनी में एक ही पत्ती की पास से साफ फोटो लेकर फिर कोशिश करें। फिर भी "
                    "पत्ती बीमार दिखे तो कृषि अधिकारी को दिखाएँ।"),
    "conf": ("AI confidence", "एआय विश्वास", "एआई भरोसा"),
    "others": ("Other possibilities", "इतर शक्यता", "अन्य संभावनाएँ"),
    "tab_treat": ("💊 What to do", "💊 काय करावे", "💊 क्या करें"),
    "tab_prev": ("🛡️ Prevention", "🛡️ बचाव", "🛡️ बचाव"),
    "tab_about": ("ℹ️ About", "ℹ️ माहिती", "ℹ️ जानकारी"),
    "sp_treat": ("What to do", "काय करावे", "क्या करें"),
    "sp_prev": ("How to prevent it", "बचाव कसा करावा", "बचाव कैसे करें"),
    "listen": ("🔊 Listen to the full advice", "🔊 संपूर्ण सल्ला ऐका", "🔊 पूरी सलाह सुनें"),
    "voice_head": ("🎙️ Voice Assistant", "🎙️ आवाज सहाय्यक", "🎙️ वॉइस असिस्टेंट"),
    "voice_hint": ('Press the microphone and say, for example: "What should I do?", '
                   '"How to prevent it?" or "What is this disease?"',
                   'माईक दाबा आणि म्हणा, उदा.: "काय करावे?", "बचाव कसा करावा?" किंवा '
                   '"हा कोणता रोग आहे?"',
                   'माइक दबाएँ और बोलें, जैसे: "क्या करना चाहिए?", "बचाव कैसे करें?" या '
                   '"यह कौन सा रोग है?"'),
    "mic": ("🎤 Record your question", "🎤 तुमचा प्रश्न रेकॉर्ड करा", "🎤 अपना सवाल रिकॉर्ड करें"),
    "or_tap": ("Or tap a button:", "किंवा बटण दाबा:", "या बटन दबाएँ:"),
    "b_treat": ("💊 What to do?", "💊 काय करावे?", "💊 क्या करें?"),
    "b_prev": ("🛡️ How to prevent?", "🛡️ बचाव कसा?", "🛡️ बचाव कैसे?"),
    "b_about": ("ℹ️ What is this?", "ℹ️ हे काय आहे?", "ℹ️ यह क्या है?"),
    "heard": ("I heard:", "मी ऐकले:", "मैंने सुना:"),
    "no_understand": ("Sorry, I did not understand. Please ask: what to do, how to prevent "
                      "it, or what is this disease.",
                      "माफ करा, समजले नाही. कृपया विचारा: काय करावे, बचाव कसा करावा, किंवा हा "
                      "कोणता रोग आहे.",
                      "क्षमा करें, समझ नहीं आया। कृपया पूछें: क्या करें, बचाव कैसे करें, या यह "
                      "कौन सा रोग है।"),
    "need_analysis": ("Please check a leaf first, then ask me.",
                      "आधी पान तपासा, मग मला विचारा.",
                      "पहले पत्ती जाँचें, फिर मुझसे पूछें।"),
    "mic_old": ("Voice recording needs a newer Streamlit. The buttons below still work.",
                "आवाज रेकॉर्डिंगसाठी नवीन Streamlit हवे. खालील बटणे चालतात.",
                "आवाज़ रिकॉर्डिंग के लिए नया Streamlit चाहिए। नीचे के बटन काम करते हैं।"),
    "mic_fail": ("I could not understand the audio. Please speak clearly and try again, or "
                 "use the buttons.",
                 "ऑडिओ समजला नाही. कृपया स्पष्ट बोला आणि पुन्हा प्रयत्न करा, किंवा बटणे वापरा.",
                 "ऑडियो समझ नहीं आया। कृपया साफ बोलें और फिर कोशिश करें, या बटन इस्तेमाल करें।"),
    "mic_silent": ("I heard no sound. Please allow the microphone in your browser (tap the lock icon next to the web address), speak close to the phone, and record again.",
                   "आवाज ऐकू आला नाही. कृपया ब्राउझरमध्ये माईकची परवानगी द्या (वेब पत्त्याजवळील कुलूप चिन्ह दाबा), फोनजवळ बोला आणि पुन्हा रेकॉर्ड करा.",
                   "कोई आवाज़ नहीं सुनाई दी। कृपया ब्राउज़र में माइक की अनुमति दें (वेब पते के पास ताले का निशान दबाएँ), फोन के पास बोलें और फिर रिकॉर्ड करें।"),
    "mic_net": ("The speech service could not be reached. Check the internet connection, or type your question below.",
                "स्पीच सेवा मिळाली नाही. इंटरनेट तपासा, किंवा खाली तुमचा प्रश्न टाइप करा.",
                "स्पीच सेवा तक नहीं पहुँच सके। इंटरनेट जाँचें, या नीचे अपना सवाल लिखें।"),
    "mic_help": ("Tips: use Chrome, open the https link, and speak in the language chosen at the top.",
                 "सूचना: Chrome वापरा, https लिंक उघडा आणि वर निवडलेल्या भाषेत बोला.",
                 "सुझाव: Chrome इस्तेमाल करें, https लिंक खोलें और ऊपर चुनी भाषा में बोलें।"),
    "type_q": ("⌨️ Or type your question and press Enter",
               "⌨️ किंवा तुमचा प्रश्न टाइप करा आणि Enter दाबा",
               "⌨️ या अपना सवाल लिखें और Enter दबाएँ"),
    "likely_note": ("Not fully sure - this is the most likely answer. Look at the leaf yourself: if you see spots, streaks or yellowing, do not rely on it. Take another clear photo to confirm, and check with an agriculture officer.",
                    "पूर्ण खात्री नाही - हे सर्वात शक्य उत्तर आहे. पान स्वतः पाहा: डाग, रेषा किंवा पिवळेपणा दिसल्यास यावर अवलंबून राहू नका. खात्रीसाठी आणखी एक स्पष्ट फोटो काढा आणि कृषी अधिकाऱ्यांकडून तपासून घ्या.",
                    "पूरा भरोसा नहीं है - यह सबसे संभावित जवाब है। पत्ती खुद देखें: धब्बे, धारियाँ या पीलापन दिखे तो इस पर भरोसा न करें। पक्का करने के लिए एक और साफ फोटो लें और कृषि अधिकारी से जँचवाएँ।"),
    "tab_med": ("💊 Medicine", "💊 औषध", "💊 दवा"),
    "sp_med": ("Medicine and spray", "औषध व फवारणी", "दवा और छिड़काव"),
    "b_med": ("🧪 Which medicine?", "🧪 कोणते औषध?", "🧪 कौन सी दवा?"),
    "b_all": ("📖 Tell me everything", "📖 सर्व सांगा", "📖 सब कुछ बताइए"),
    "step_word": ("Step", "पायरी", "चरण"),
    "auto_speak": ("🔊 Speak the answer automatically", "🔊 उत्तर आपोआप बोलून सांगा",
                   "🔊 जवाब अपने आप बोलकर सुनाएँ"),
    "tagline": ("AI leaf doctor with voice help in English, Marathi and Hindi",
                "पानांचा एआय डॉक्टर - इंग्रजी, मराठी आणि हिंदीत आवाजाची मदत",
                "पत्तियों का एआई डॉक्टर - अंग्रेज़ी, मराठी और हिंदी में आवाज़ की मदद"),
    "also": ("Also possible", "हे देखील शक्य", "यह भी संभव"),
    "likely_chip": ("Likely - please confirm", "शक्यता - कृपया खात्री करा",
                    "संभावित - कृपया पुष्टि करें"),
    "no_clear": ("No clear answer", "स्पष्ट उत्तर नाही", "स्पष्ट जवाब नहीं"),
    "probs": ("How sure is the AI?", "एआयला किती खात्री?", "एआई को कितना भरोसा है?"),
    "crop_any": ("🔎 Other / not sure", "🔎 इतर / खात्री नाही", "🔎 अन्य / पक्का नहीं"),
    "wrong_crop": ("The AI is confusing this leaf with another crop ({other}). Photograph ONE leaf on a plain light background, filling the frame, and try again. Do not rely on this result.",
                   "एआय या पानाची दुसऱ्या पिकाशी ({other}) गल्लत करत आहे. साध्या फिकट पार्श्वभूमीवर एकच पान, फ्रेम भरून, फोटो काढून पुन्हा प्रयत्न करा. या निकालावर अवलंबून राहू नका.",
                   "एआई इस पत्ती को दूसरी फसल ({other}) समझ रहा है। सादी हल्की पृष्ठभूमि पर एक ही पत्ती, पूरे फ्रेम में, फोटो लेकर फिर कोशिश करें। इस नतीजे पर भरोसा न करें।"),
    "helpline": ("📞 Free farmer helpline (India): Kisan Call Centre 1800-180-1551, "
                 "6 AM to 10 PM.",
                 "📞 मोफत शेतकरी हेल्पलाइन (भारत): किसान कॉल सेंटर 1800-180-1551, सकाळी ६ ते रात्री १०.",
                 "📞 निःशुल्क किसान हेल्पलाइन (भारत): किसान कॉल सेंटर 1800-180-1551, सुबह 6 से रात 10 बजे।"),
    "footer": ("Experimental prototype. AI can be wrong and this is not a confirmed "
               "diagnosis. Confirm with an agriculture officer before spraying anything.",
               "हा प्रायोगिक नमुना आहे. एआय चुकू शकते आणि हे निश्चित निदान नाही. कोणतीही फवारणी "
               "करण्यापूर्वी कृषी अधिकाऱ्यांकडून खात्री करा.",
               "यह प्रायोगिक प्रोटोटाइप है। एआई गलत हो सकता है और यह पक्का निदान नहीं है। कोई भी "
               "छिड़काव करने से पहले कृषि अधिकारी से पुष्टि करें।"),
}

BANANA_NAME = {
    "healthy": ("Healthy banana leaf", "निरोगी केळीचे पान", "स्वस्थ केले की पत्ती"),
    "sigatoka": ("Sigatoka leaf disease", "सिगाटोका पानांचा रोग", "सिगाटोका पत्ती रोग"),
    "xanthomonas": ("Xanthomonas bacterial wilt", "झॅन्थोमोनास जिवाणूजन्य मर",
                    "ज़ैंथोमोनास जीवाणु उकठा (मुरझान)"),
}

# ----------------------------------------------------------------------
# Solutions. Steps are separated by "|".
# Chemicals are never named or dosed: always "ask your agriculture officer".
# ----------------------------------------------------------------------
KB = {
    "banana_healthy": {
        "about": (
            "The leaf looks healthy. No sign of Sigatoka or Xanthomonas wilt was found in this photo.",
            "पान निरोगी दिसत आहे. या फोटोमध्ये सिगाटोका किंवा झॅन्थोमोनास रोगाची लक्षणे दिसत नाहीत.",
            "पत्ती स्वस्थ दिख रही है। इस फोटो में सिगाटोका या ज़ैंथोमोनास रोग के लक्षण नहीं दिखे।"),
        "treat": (
            "No treatment is needed.|Keep checking older leaves every week.|If spots or yellow streaks appear later, take a new photo.",
            "उपचाराची गरज नाही.|जुनी पाने दर आठवड्याला तपासा.|नंतर डाग किंवा पिवळ्या रेषा दिसल्यास नवीन फोटो काढा.",
            "उपचार की ज़रूरत नहीं है।|पुरानी पत्तियों की हर हफ़्ते जाँच करें।|बाद में धब्बे या पीली धारियाँ दिखें तो नई फोटो लें।"),
        "prevent": (
            "Remove dry and dead leaves regularly.|Keep drainage clear so water does not stand.|Clean your knife and tools before moving to the next plant.|Use healthy, disease-free suckers for planting.",
            "वाळलेली व मेलेली पाने नियमित काढा.|पाणी साचू नये म्हणून निचरा नीट ठेवा.|पुढच्या झाडाकडे जाण्यापूर्वी चाकू व अवजारे स्वच्छ करा.|लागवडीसाठी निरोगी, रोगमुक्त कंद वापरा.",
            "सूखी और मरी पत्तियाँ नियमित रूप से हटाएँ।|पानी जमा न हो, इसके लिए जल निकासी ठीक रखें।|अगले पौधे पर जाने से पहले चाकू और औज़ार साफ करें।|रोपाई के लिए स्वस्थ, रोगमुक्त सकर्स (पौध) इस्तेमाल करें।"),
    },
    "banana_sigatoka": {
        "about": (
            "Sigatoka is a fungal leaf disease. It makes brown streaks and spots on leaves, which dry out and make the bunch smaller. It spreads in warm, humid and rainy weather.",
            "सिगाटोका हा बुरशीजन्य पानावरील रोग आहे. पानांवर तपकिरी रेषा व डाग पडतात, पाने वाळतात आणि घड लहान राहतो. उष्ण, दमट आणि पावसाळी हवामानात हा रोग पसरतो.",
            "सिगाटोका फफूंद से होने वाला पत्तियों का रोग है। इससे पत्तियों पर भूरी धारियाँ और धब्बे बनते हैं, पत्तियाँ सूख जाती हैं और घौद छोटी रह जाती है। यह गर्म, नम और बरसाती मौसम में फैलता है।"),
        "treat": (
            "Cut off badly affected leaves and carry them out of the field. Burn or bury them.|Do not spray anything on your own. Show a leaf to your local agriculture officer or Krishi Vigyan Kendra (KVK) and use only the fungicide they advise, exactly as on the label.|Improve air flow: remove extra suckers and keep plants well spaced.|Take a new photo after 7 days to see whether it is spreading.",
            "जास्त बाधित पाने कापून शेताबाहेर न्या. ती जाळा किंवा पुरून टाका.|स्वतःहून कोणतीही फवारणी करू नका. पान स्थानिक कृषी अधिकारी किंवा कृषी विज्ञान केंद्राला (KVK) दाखवा आणि ते सांगतील तेच बुरशीनाशक लेबलप्रमाणे वापरा.|जास्तीची पिल्ले काढून व झाडांमध्ये योग्य अंतर ठेवून हवा खेळती ठेवा.|७ दिवसांनी नवीन फोटो काढून रोग पसरतोय का ते पहा.",
            "ज़्यादा प्रभावित पत्तियाँ काटकर खेत से बाहर ले जाएँ। उन्हें जला दें या गाड़ दें।|अपने मन से कोई स्प्रे न करें। पत्ती स्थानीय कृषि अधिकारी या कृषि विज्ञान केंद्र (KVK) को दिखाएँ और वे जो फफूंदनाशक बताएँ, वही लेबल के अनुसार इस्तेमाल करें।|फालतू पुत्ती (सकर्स) हटाकर और पौधों के बीच सही दूरी रखकर हवा का आवागमन बढ़ाएँ।|7 दिन बाद नई फोटो लेकर देखें कि रोग फैल रहा है या नहीं।"),
        "prevent": (
            "Do not leave old infected leaves in the field.|Avoid standing water; good drainage is important.|Plant with proper spacing.|Give balanced manure so plants stay strong.|Check older leaves every week, especially after rain.",
            "शेतात जुनी बाधित पाने ठेवू नका.|शेतात पाणी साचू देऊ नका; निचरा महत्त्वाचा आहे.|लागवड योग्य अंतरावर करा.|संतुलित खते द्या म्हणजे झाडे मजबूत राहतात.|विशेषतः पावसानंतर जुनी पाने दर आठवड्याला तपासा.",
            "खेत में पुरानी संक्रमित पत्तियाँ न छोड़ें।|खेत में पानी जमा न होने दें; जल निकासी ज़रूरी है।|सही दूरी पर रोपाई करें।|संतुलित खाद दें ताकि पौधे मज़बूत रहें।|खासकर बारिश के बाद पुरानी पत्तियों की हर हफ़्ते जाँच करें।"),
    },
    "banana_xanthomonas": {
        "about": (
            "Xanthomonas wilt is a serious bacterial disease. Leaves turn yellow and wilt, and the fruit ripens unevenly. It spreads through tools, insects visiting the flowers, and infected planting material.",
            "झॅन्थोमोनास हा गंभीर जिवाणूजन्य रोग आहे. पाने पिवळी पडून सुकतात आणि केळी असमान पिकतात. हा रोग अवजारे, फुलांवर येणारे कीटक आणि बाधित लागवड साहित्याद्वारे पसरतो.",
            "ज़ैंथोमोनास उकठा एक गंभीर जीवाणु रोग है। पत्तियाँ पीली होकर मुरझा जाती हैं और केले असमान रूप से पकते हैं। यह औज़ारों, फूलों पर आने वाले कीटों और संक्रमित रोपण सामग्री से फैलता है।"),
        "treat": (
            "No spray can cure it. Never cut other plants with the same knife.|Tell your agriculture officer at once, because this disease can spread fast.|Experts usually advise removing the infected plant completely and not carrying any part out of the field. Get the disease confirmed before pulling out plants.|After every plant, disinfect tools with a bleach solution or by heating them in a flame.",
            "यावर बरा करणारी फवारणी नाही. एकाच चाकूने इतर झाडे कधीही कापू नका.|हा रोग झपाट्याने पसरू शकतो, म्हणून ताबडतोब कृषी अधिकाऱ्यांना कळवा.|तज्ज्ञ सहसा बाधित झाड पूर्णपणे काढायला सांगतात आणि त्याचे कोणतेही भाग शेताबाहेर नेऊ नयेत असे सांगतात. झाडे उपटण्यापूर्वी रोगाची खात्री करून घ्या.|प्रत्येक झाडानंतर अवजारे ब्लीचच्या द्रावणाने किंवा आगीवर तापवून निर्जंतुक करा.",
            "इसे ठीक करने वाला कोई स्प्रे नहीं है। एक ही चाकू से दूसरे पौधे कभी न काटें।|यह रोग तेज़ी से फैल सकता है, इसलिए तुरंत कृषि अधिकारी को बताएँ।|विशेषज्ञ आमतौर पर संक्रमित पौधे को पूरी तरह हटाने और उसके किसी भी हिस्से को खेत से बाहर न ले जाने की सलाह देते हैं। पौधा उखाड़ने से पहले रोग की पुष्टि करवा लें।|हर पौधे के बाद औज़ारों को ब्लीच के घोल से या आग पर तपाकर कीटाणुरहित करें।"),
        "prevent": (
            "Use disease-free planting material.|Always disinfect your tools between plants.|Do not carry infected plant parts out of the field.|Ask your officer about removing the male bud to stop insects spreading the disease.|Report any suspicion early.",
            "रोगमुक्त लागवड साहित्य वापरा.|झाडांमध्ये अवजारे नेहमी निर्जंतुक करा.|बाधित भाग शेताबाहेर नेऊ नका.|कीटकांमार्फत रोग पसरू नये म्हणून नर कळी काढण्याबद्दल अधिकाऱ्यांना विचारा.|संशय आल्यास लगेच कळवा.",
            "रोगमुक्त रोपण सामग्री इस्तेमाल करें।|पौधों के बीच औज़ार हमेशा कीटाणुरहित करें।|संक्रमित हिस्से खेत से बाहर न ले जाएँ।|कीटों से रोग फैलना रोकने के लिए नर कली हटाने के बारे में अधिकारी से पूछें।|शक होते ही जल्दी बताएँ।"),
    },
    "healthy": {
        "about": (
            "The leaf looks healthy in this photo.",
            "या फोटोमध्ये पान निरोगी दिसत आहे.",
            "इस फोटो में पत्ती स्वस्थ दिख रही है।"),
        "treat": (
            "No treatment is needed.|Keep checking the plant every week.",
            "उपचाराची गरज नाही.|झाड दर आठवड्याला तपासत राहा.",
            "उपचार की ज़रूरत नहीं है।|पौधे की हर हफ़्ते जाँच करते रहें।"),
        "prevent": (
            "Water at the base of the plant.|Keep the field clean and weed-free.|Clean tools between plants.|Take a photo again if you see spots, curling or yellowing.",
            "झाडाच्या बुंध्याशी पाणी द्या.|शेत स्वच्छ व तणमुक्त ठेवा.|झाडांमध्ये अवजारे स्वच्छ करा.|डाग, पाने वळणे किंवा पिवळेपणा दिसल्यास पुन्हा फोटो काढा.",
            "पौधे की जड़ के पास पानी दें।|खेत साफ और खरपतवार-मुक्त रखें।|पौधों के बीच औज़ार साफ करें।|धब्बे, पत्ती मुड़ना या पीलापन दिखे तो फिर फोटो लें।"),
    },
    "fungal": {
        "about": (
            "This looks like a fungal leaf disease. In humid, wet weather fungi grow on leaves and cause spots, rust, mould or blight.",
            "हा बुरशीजन्य पानावरील रोग असल्याचे दिसते. दमट, ओल्या हवामानात बुरशी पानांवर वाढते आणि डाग, तांबेरा, बुरशी किंवा करपा निर्माण करते.",
            "यह फफूंद से होने वाला पत्तियों का रोग लगता है। नम और गीले मौसम में फफूंद पत्तियों पर बढ़ती है और धब्बे, रतुआ, फफूंद या झुलसा पैदा करती है।"),
        "treat": (
            "Remove the worst affected leaves and destroy them away from the field.|Avoid wetting the leaves; water at the base of the plant in the morning.|Do not spray chemicals on your own. Take a leaf to your local agriculture officer or KVK and use only the fungicide they advise, exactly as on the label.|Check again after 5 to 7 days and take a new photo.",
            "सर्वात जास्त बाधित पाने काढून शेतापासून दूर नष्ट करा.|पाने ओली होऊ देऊ नका; सकाळी झाडाच्या बुंध्याजवळ पाणी द्या.|स्वतःहून रसायने फवारू नका. पान स्थानिक कृषी अधिकारी किंवा KVK कडे न्या आणि ते सांगतील तेच बुरशीनाशक लेबलप्रमाणे वापरा.|५ ते ७ दिवसांनी पुन्हा तपासा आणि नवीन फोटो काढा.",
            "सबसे ज़्यादा प्रभावित पत्तियाँ हटाकर खेत से दूर नष्ट करें।|पत्तियाँ गीली न करें; सुबह पौधे की जड़ के पास पानी दें।|अपने मन से रसायन न छिड़कें। पत्ती स्थानीय कृषि अधिकारी या KVK को दिखाएँ और वे जो फफूंदनाशक बताएँ, वही लेबल के अनुसार इस्तेमाल करें।|5 से 7 दिन बाद फिर जाँच करें और नई फोटो लें।"),
        "prevent": (
            "Use disease-free seed or seedlings, and resistant varieties where available.|Rotate crops; do not grow the same crop in the same plot every season.|Keep proper spacing for air flow and remove weeds.|Clear old plant residue after harvest.",
            "रोगमुक्त बियाणे/रोपे आणि उपलब्ध असल्यास रोगप्रतिकारक वाण वापरा.|पीक फेरपालट करा; एकाच प्लॉटमध्ये दर हंगामात तेच पीक घेऊ नका.|हवा खेळती राहण्यासाठी योग्य अंतर ठेवा आणि तण काढा.|काढणीनंतर जुने अवशेष साफ करा.",
            "रोगमुक्त बीज/पौध और उपलब्ध हो तो रोग-प्रतिरोधी किस्में इस्तेमाल करें।|फसल चक्र अपनाएँ; हर मौसम में एक ही खेत में वही फसल न लगाएँ।|हवा के लिए सही दूरी रखें और खरपतवार हटाएँ।|कटाई के बाद पुराने अवशेष साफ करें।"),
    },
    "bacterial": {
        "about": (
            "This looks like a bacterial disease. Bacteria spread through water splashes, wind, tools and hands, and often cause small dark spots with yellow rings.",
            "हा जिवाणूजन्य रोग असल्याचे दिसते. जिवाणू पाण्याचे थेंब, वारा, अवजारे आणि हातांद्वारे पसरतात आणि अनेकदा पिवळ्या कडांचे लहान काळे डाग पडतात.",
            "यह जीवाणु से होने वाला रोग लगता है। जीवाणु पानी की बूँदों, हवा, औज़ारों और हाथों से फैलते हैं और अक्सर पीले घेरे वाले छोटे काले धब्बे बनाते हैं।"),
        "treat": (
            "Remove and destroy badly infected leaves; do not make compost from them.|Do not work in the field when plants are wet.|Ordinary fungicides do not work on bacterial diseases. Ask your agriculture officer which product, if any, is suitable.|Wash your hands and clean tools after touching infected plants.",
            "जास्त बाधित पाने काढून नष्ट करा; त्यांचे कंपोस्ट करू नका.|झाडे ओली असताना शेतात काम करू नका.|जिवाणूजन्य रोगांवर साध्या बुरशीनाशकांचा परिणाम होत नाही. कोणते औषध योग्य आहे ते कृषी अधिकाऱ्यांना विचारा.|बाधित झाडांना हात लावल्यानंतर हात धुवा आणि अवजारे स्वच्छ करा.",
            "ज़्यादा संक्रमित पत्तियाँ हटाकर नष्ट करें; उनकी खाद न बनाएँ।|पौधे गीले हों तब खेत में काम न करें।|जीवाणु रोगों पर साधारण फफूंदनाशक असर नहीं करते। कौन सी दवा ठीक रहेगी, यह कृषि अधिकारी से पूछें।|संक्रमित पौधे छूने के बाद हाथ धोएँ और औज़ार साफ करें।"),
        "prevent": (
            "Use clean, certified seed or seedlings.|Water at the base; avoid overhead sprinkling.|Rotate crops for 2 to 3 seasons.|Disinfect tools between plants.",
            "स्वच्छ प्रमाणित बियाणे/रोपे वापरा.|झाडाच्या बुंध्याशी पाणी द्या; वरून फवारणी पद्धतीने देणे टाळा.|२ ते ३ हंगाम पीक फेरपालट करा.|झाडांमध्ये अवजारे निर्जंतुक करा.",
            "साफ, प्रमाणित बीज/पौध इस्तेमाल करें।|पौधे की जड़ के पास पानी दें; ऊपर से फव्वारा सिंचाई से बचें।|2 से 3 मौसम तक फसल चक्र अपनाएँ।|पौधों के बीच औज़ार कीटाणुरहित करें।"),
    },
    "viral": {
        "about": (
            "This looks like a virus disease. Viruses are carried by insects such as whiteflies and aphids, or by touch. They cause curling, yellowing or mosaic patterns on leaves.",
            "हा विषाणूजन्य रोग असल्याचे दिसते. विषाणू पांढरी माशी, मावा यांसारख्या कीटकांमार्फत किंवा स्पर्शाने पसरतात. यामुळे पाने वळतात, पिवळी पडतात किंवा त्यावर मोझॅकसारखे नक्षीदार डाग येतात.",
            "यह विषाणु (वायरस) से होने वाला रोग लगता है। वायरस सफ़ेद मक्खी और माहू जैसे कीटों या संपर्क से फैलते हैं। इससे पत्तियाँ मुड़ती हैं, पीली पड़ती हैं या उन पर मोज़ेक जैसी चित्तियाँ बनती हैं।"),
        "treat": (
            "There is no cure for a virus. Remove and destroy infected plants so it does not spread.|Control the insects that carry it; ask your agriculture officer about safe options.|Use yellow sticky traps to catch whiteflies.|Wash your hands and tools after touching infected plants.",
            "विषाणूवर इलाज नाही. रोग पसरू नये म्हणून बाधित झाडे उपटून नष्ट करा.|रोग पसरवणाऱ्या कीटकांवर नियंत्रण ठेवा; सुरक्षित पर्यायांबद्दल कृषी अधिकाऱ्यांना विचारा.|पांढऱ्या माशीसाठी पिवळे चिकट सापळे लावा.|बाधित झाडांना हात लावल्यानंतर हात व अवजारे धुवा.",
            "वायरस का कोई इलाज नहीं है। रोग फैलने से रोकने के लिए संक्रमित पौधे उखाड़कर नष्ट करें।|रोग फैलाने वाले कीटों पर नियंत्रण रखें; सुरक्षित विकल्पों के बारे में कृषि अधिकारी से पूछें।|सफ़ेद मक्खी पकड़ने के लिए पीले चिपचिपे जाल लगाएँ।|संक्रमित पौधे छूने के बाद हाथ और औज़ार धोएँ।"),
        "prevent": (
            "Plant virus-resistant varieties and healthy seedlings.|Keep weeds down, since insects live on them.|Use insect-proof nets in nurseries.|Check plants weekly for insects.",
            "विषाणू-प्रतिकारक वाण आणि निरोगी रोपे लावा.|तण काढा, कारण त्यावर कीटक राहतात.|रोपवाटिकेत कीटकरोधक जाळी वापरा.|दर आठवड्याला झाडांवर कीटक आहेत का ते पाहा.",
            "वायरस-प्रतिरोधी किस्में और स्वस्थ पौध लगाएँ।|खरपतवार हटाएँ, क्योंकि उन पर कीट रहते हैं।|नर्सरी में कीट-रोधी जाली का उपयोग करें।|हर हफ़्ते पौधों पर कीट देखें।"),
    },
    "pest": {
        "about": (
            "This looks like damage from spider mites, tiny pests that suck sap from the underside of leaves and cause yellow speckles and fine webbing. They thrive in hot, dry weather.",
            "हे कोळी (स्पायडर माइट) नुकसान असल्याचे दिसते. हे लहान कीटक पानाच्या खालच्या बाजूने रस शोषतात, त्यामुळे पिवळे ठिपके व बारीक जाळी दिसते. गरम, कोरड्या हवामानात ते वाढतात.",
            "यह मकड़ी (स्पाइडर माइट) से हुआ नुकसान लगता है। ये छोटे कीट पत्ती के नीचे से रस चूसते हैं, जिससे पीले बिंदु और बारीक जाले दिखते हैं। गर्म और सूखे मौसम में ये बढ़ते हैं।"),
        "treat": (
            "Look at the underside of the leaf for tiny moving dots or webs.|Spray a strong jet of water on the underside of the leaves to wash the mites off.|Remove heavily damaged leaves.|If it is severe, ask your agriculture officer for a suitable miticide and use it only as per the label.",
            "पानाच्या खालच्या बाजूला लहान हलणारे ठिपके किंवा जाळी आहे का ते पाहा.|पानांच्या खालच्या बाजूवर पाण्याचा जोरदार फवारा मारून कोळी धुवून काढा.|जास्त खराब झालेली पाने काढा.|प्रादुर्भाव जास्त असल्यास कृषी अधिकाऱ्यांकडून योग्य माइटनाशक विचारा आणि ते लेबलप्रमाणेच वापरा.",
            "पत्ती के नीचे की ओर छोटे हिलते बिंदु या जाले देखें।|पत्तियों के नीचे की ओर पानी की तेज़ धार मारकर माइट धो दें।|ज़्यादा खराब पत्तियाँ हटा दें।|प्रकोप ज़्यादा हो तो कृषि अधिकारी से उपयुक्त माइटनाशक पूछें और केवल लेबल के अनुसार इस्तेमाल करें।"),
        "prevent": (
            "Avoid dusty, water-stressed conditions; irrigate regularly.|Check the underside of leaves weekly in hot weather.|Protect natural enemies of mites by avoiding unnecessary spraying.",
            "धुळीचे व पाण्याचा ताण असलेले वातावरण टाळा; नियमित सिंचन करा.|गरम हवामानात दर आठवड्याला पानांची खालची बाजू तपासा.|अनावश्यक फवारणी टाळून मित्रकीटकांचे संरक्षण करा.",
            "धूल भरे और पानी की कमी वाले हालात से बचें; नियमित सिंचाई करें।|गर्म मौसम में हर हफ़्ते पत्तियों की निचली सतह देखें।|बेवजह छिड़काव से बचकर मित्र कीटों को बचाएँ।"),
    },
    "vector": {
        "about": (
            "Citrus greening (Huanglongbing) is a serious bacterial disease spread by a small insect called the psyllid. It causes yellow, blotchy leaves and small, bitter, misshapen fruit.",
            "सिट्रस ग्रीनिंग (हुआंगलाँगबिंग) हा सायला नावाच्या लहान किडीमार्फत पसरणारा गंभीर जिवाणूजन्य रोग आहे. यामुळे पाने पिवळी-ठिपकेदार होतात आणि फळे लहान, कडू व वेडीवाकडी होतात.",
            "सिट्रस ग्रीनिंग (हुआंगलोंगबिंग) साइला नाम के छोटे कीट से फैलने वाला गंभीर जीवाणु रोग है। इससे पत्तियाँ पीली-चितकबरी हो जाती हैं और फल छोटे, कड़वे और बेडौल बनते हैं।"),
        "treat": (
            "There is no cure for citrus greening. Report it to your agriculture officer immediately.|Infected trees are usually removed so they do not infect others; follow official advice.|Ask your local officer how to control psyllids.|Do not take cuttings or buds from affected trees.",
            "सिट्रस ग्रीनिंगवर इलाज नाही. ताबडतोब कृषी अधिकाऱ्यांना कळवा.|इतर झाडांना लागण होऊ नये म्हणून बाधित झाडे सहसा काढली जातात; अधिकृत सल्ल्याप्रमाणे करा.|सायलाच्या नियंत्रणासाठी स्थानिक अधिकाऱ्यांचा सल्ला घ्या.|बाधित झाडांपासून कलम किंवा डोळे घेऊ नका.",
            "सिट्रस ग्रीनिंग का कोई इलाज नहीं है। तुरंत कृषि अधिकारी को बताएँ।|दूसरे पेड़ों में फैलने से रोकने के लिए संक्रमित पेड़ अक्सर हटाए जाते हैं; सरकारी सलाह के अनुसार करें।|साइला नियंत्रण के लिए स्थानीय अधिकारी से सलाह लें।|संक्रमित पेड़ों से कलम या आँख (बड) न लें।"),
        "prevent": (
            "Plant only certified disease-free saplings.|Watch new shoots for psyllids.|Keep the orchard clean and well fed.|Do not bring plants from affected areas.",
            "फक्त प्रमाणित रोगमुक्त रोपे लावा.|नवीन फुटव्यांवर सायला आहे का ते पाहा.|बाग स्वच्छ व चांगली पोषित ठेवा.|बाधित भागातून रोपे आणू नका.",
            "केवल प्रमाणित रोगमुक्त पौध लगाएँ।|नई कोंपलों पर साइला देखें।|बाग साफ और अच्छे पोषण वाला रखें।|प्रभावित इलाकों से पौध न लाएँ।"),
    },
}


# ----------------------------------------------------------------------
# More categories + medicine advice + safety (added to KB below)
# ----------------------------------------------------------------------
KB["late_blight"] = {
    "about": (
        "Late blight is a very fast-spreading disease of potato and tomato in cool, cloudy, wet weather. Leaves get dark, water-soaked patches, often with white fuzzy growth underneath, and a whole field can be lost in a few days.",
        "करपा (लेट ब्लाइट) हा बटाटा व टोमॅटोवर थंड, ढगाळ, ओल्या हवामानात खूप वेगाने पसरणारा रोग आहे. पानांवर गडद, पाणथळ डाग पडतात, खालच्या बाजूला अनेकदा पांढरी बुरशी दिसते, आणि काही दिवसांत संपूर्ण शेत नष्ट होऊ शकते.",
        "पछेती झुलसा (लेट ब्लाइट) आलू और टमाटर का बहुत तेज़ी से फैलने वाला रोग है, जो ठंडे, बादल वाले, गीले मौसम में होता है। पत्तियों पर गहरे, पानी जैसे धब्बे बनते हैं, नीचे अक्सर सफ़ेद फफूंद दिखती है, और कुछ ही दिनों में पूरा खेत बर्बाद हो सकता है।"),
    "treat": (
        "Act the same day: remove and destroy infected leaves and plants away from the field.|Do not water from above; keep the leaves dry.|Start protective sprays right away (see Medicine) and cover the whole field, not only the sick plants.|For potato, cut the vines if the disease is severe, and dig the tubers only after the skin has hardened. Do not store spotted tubers.|In wet weather, check the whole field every 2 to 3 days.",
        "त्याच दिवशी कृती करा: बाधित पाने व झाडे काढून शेतापासून दूर नष्ट करा.|वरून पाणी देऊ नका; पाने कोरडी ठेवा.|संरक्षक फवारण्या लगेच सुरू करा (औषध पहा) आणि फक्त आजारी झाडांवर नाही, तर संपूर्ण शेतावर फवारा.|बटाट्यात रोग जास्त असल्यास वेल कापा आणि साल घट्ट झाल्यावरच कंद काढा. डाग असलेले कंद साठवू नका.|ओल्या हवामानात संपूर्ण शेत दर २ ते ३ दिवसांनी तपासा.",
        "उसी दिन कार्रवाई करें: संक्रमित पत्तियाँ और पौधे हटाकर खेत से दूर नष्ट करें।|ऊपर से पानी न दें; पत्तियाँ सूखी रखें।|सुरक्षात्मक छिड़काव तुरंत शुरू करें (दवा देखें) और केवल बीमार पौधों पर नहीं, पूरे खेत पर करें।|आलू में रोग ज़्यादा हो तो बेल काट दें और छिलका पक्का होने के बाद ही कंद खोदें। धब्बे वाले कंद भंडारित न करें।|गीले मौसम में पूरा खेत हर 2 से 3 दिन में देखें।"),
    "prevent": (
        "Plant certified, disease-free seed potatoes and resistant varieties if available.|Do not plant potato and tomato side by side or one after the other.|Destroy leftover plants and spotted tubers after harvest.|Keep proper spacing and drain water well.|Watch the weather: start sprays before long, cloudy, wet spells.",
        "प्रमाणित, रोगमुक्त बियाणे बटाटे आणि उपलब्ध असल्यास रोगप्रतिकारक वाण लावा.|बटाटा व टोमॅटो शेजारी शेजारी किंवा एकापाठोपाठ लावू नका.|काढणीनंतर उरलेली झाडे व डाग असलेले कंद नष्ट करा.|योग्य अंतर ठेवा आणि पाण्याचा चांगला निचरा करा.|हवामानावर लक्ष ठेवा: दीर्घ ढगाळ, ओल्या काळापूर्वी फवारणी सुरू करा.",
        "प्रमाणित, रोगमुक्त बीज आलू और उपलब्ध हो तो रोग-प्रतिरोधी किस्में लगाएँ।|आलू और टमाटर साथ-साथ या एक के बाद एक न लगाएँ।|कटाई के बाद बची हुई बेलें और धब्बे वाले कंद नष्ट करें।|सही दूरी रखें और पानी की अच्छी निकासी करें।|मौसम पर नज़र रखें: लंबे बादल और नमी वाले दौर से पहले छिड़काव शुरू करें।"),
}

KB["powdery"] = {
    "about": (
        "Powdery mildew shows as a white, flour-like powder on leaves. It likes warm days, cool nights and dry air, and it weakens the plant and lowers the yield.",
        "भुरी (पावडरी मिल्ड्यू) रोगात पानांवर पिठासारखी पांढरी पूड दिसते. उबदार दिवस, थंड रात्री आणि कोरड्या हवेत तो वाढतो; झाड कमजोर होते आणि उत्पन्न घटते.",
        "चूर्णिल आसिता (पाउडरी मिल्ड्यू) में पत्तियों पर आटे जैसा सफ़ेद चूर्ण दिखता है। यह गर्म दिन, ठंडी रात और सूखी हवा में बढ़ता है; पौधा कमज़ोर होता है और उपज घटती है।"),
    "treat": (
        "Remove and destroy the badly affected leaves.|Keep plants well spaced and prune for air flow; avoid too much nitrogen fertiliser.|Water at the base of the plant in the morning.|Start spraying at the first white patches (see Medicine); early action works best.",
        "जास्त बाधित पाने काढून नष्ट करा.|झाडांमध्ये योग्य अंतर ठेवा व हवा खेळती राहण्यासाठी छाटणी करा; नत्रयुक्त खत जास्त देऊ नका.|सकाळी झाडाच्या बुंध्याशी पाणी द्या.|पहिले पांढरे डाग दिसताच फवारणी सुरू करा (औषध पहा); लवकर कृती केल्यास जास्त फायदा होतो.",
        "ज़्यादा प्रभावित पत्तियाँ हटाकर नष्ट करें।|पौधों के बीच सही दूरी रखें और हवा के लिए छँटाई करें; नाइट्रोजन खाद ज़्यादा न दें।|सुबह पौधे की जड़ के पास पानी दें।|पहले सफ़ेद धब्बे दिखते ही छिड़काव शुरू करें (दवा देखें); जल्दी कदम उठाना सबसे अच्छा रहता है।"),
    "prevent": (
        "Plant resistant varieties where available.|Avoid crowding; keep air moving between plants.|Remove infected plant waste after harvest.|Check plants weekly from the flowering stage.",
        "उपलब्ध असल्यास रोगप्रतिकारक वाण लावा.|गर्दी टाळा; झाडांमध्ये हवा खेळती ठेवा.|काढणीनंतर बाधित झाडांचे अवशेष काढून टाका.|फुलोऱ्यापासून दर आठवड्याला झाडे तपासा.",
        "उपलब्ध हो तो रोग-प्रतिरोधी किस्में लगाएँ।|भीड़ से बचें; पौधों के बीच हवा चलने दें।|कटाई के बाद संक्रमित अवशेष हटा दें।|फूल आने की अवस्था से हर हफ़्ते पौधे जाँचें।"),
}

MEDICINE = {
    "banana_healthy": (
        "No medicine or spray is needed.|Do not spray chemicals on a healthy crop; it wastes money and harms useful insects.",
        "औषध किंवा फवारणीची गरज नाही.|निरोगी पिकावर रसायने फवारू नका; त्यामुळे पैसा वाया जातो आणि मित्रकीटकांना हानी होते.",
        "दवा या छिड़काव की ज़रूरत नहीं है।|स्वस्थ फसल पर रसायन न छिड़कें; इससे पैसा बेकार जाता है और मित्र कीटों को नुकसान होता है।"),
    "healthy": (
        "No medicine or spray is needed.|Do not spray chemicals on a healthy crop; it wastes money and harms useful insects.",
        "औषध किंवा फवारणीची गरज नाही.|निरोगी पिकावर रसायने फवारू नका; त्यामुळे पैसा वाया जातो आणि मित्रकीटकांना हानी होते.",
        "दवा या छिड़काव की ज़रूरत नहीं है।|स्वस्थ फसल पर रसायन न छिड़कें; इससे पैसा बेकार जाता है और मित्र कीटों को नुकसान होता है।"),
    "banana_sigatoka": (
        "Medicines commonly advised for Sigatoka (Tamil Nadu Agricultural University): Propiconazole about 1 ml per litre of water, or Mancozeb about 2.5 g per litre, or Carbendazim about 1 g per litre, mixed with a sticker such as Teepol.|Usually 3 sprays, 10 to 15 days apart, starting when the first specks appear on the leaves. Spray both sides of the leaves.|Do not use the same product every time; alternate products.|Confirm the product and dose with your agriculture officer or dealer, and follow the label.",
        "सिगाटोकासाठी सामान्यतः सुचवली जाणारी औषधे (तमिळनाडू कृषी विद्यापीठानुसार): प्रोपिकोनाझोल सुमारे १ मिली प्रति लिटर पाणी, किंवा मॅन्कोझेब सुमारे २.५ ग्रॅम प्रति लिटर, किंवा कार्बेन्डाझिम सुमारे १ ग्रॅम प्रति लिटर, त्यात टीपॉलसारखे चिकटवणारे द्रव्य मिसळा.|साधारणपणे ३ फवारण्या, १० ते १५ दिवसांच्या अंतराने, पानांवर पहिले ठिपके दिसताच सुरू करा. पानांच्या दोन्ही बाजूंना फवारा.|दर वेळी तेच औषध वापरू नका; औषधे बदलून वापरा.|औषध व मात्रा कृषी अधिकारी किंवा विक्रेत्याकडून निश्चित करा आणि लेबल पाळा.",
        "सिगाटोका के लिए आमतौर पर सुझाई जाने वाली दवाएँ (तमिलनाडु कृषि विश्वविद्यालय के अनुसार): प्रोपिकोनाज़ोल लगभग 1 मिली प्रति लीटर पानी, या मैन्कोज़ेब लगभग 2.5 ग्राम प्रति लीटर, या कार्बेन्डाज़िम लगभग 1 ग्राम प्रति लीटर, साथ में टीपॉल जैसा चिपकाने वाला पदार्थ मिलाएँ।|आमतौर पर 3 छिड़काव, 10 से 15 दिन के अंतर पर, पत्तियों पर पहले धब्बे दिखते ही शुरू करें। पत्तियों के दोनों तरफ़ छिड़कें।|हर बार वही दवा न दोहराएँ; दवाएँ बदल-बदलकर इस्तेमाल करें।|दवा और मात्रा कृषि अधिकारी या दुकानदार से पक्की कर लें और लेबल का पालन करें।"),
    "banana_xanthomonas": (
        "No spray or medicine cures Xanthomonas wilt.|Do not waste money on antibiotics or fungicides for this disease.|The real remedy is to remove infected plants as the officer advises, disinfect tools and stop the spread.",
        "झॅन्थोमोनास मरावर कोणतीही फवारणी किंवा औषध इलाज करत नाही.|या रोगासाठी प्रतिजैविके किंवा बुरशीनाशकांवर पैसे खर्च करू नका.|खरा उपाय म्हणजे अधिकाऱ्यांच्या सल्ल्याने बाधित झाडे काढणे, अवजारे निर्जंतुक करणे आणि प्रसार थांबवणे.",
        "ज़ैंथोमोनास उकठा को कोई स्प्रे या दवा ठीक नहीं करती।|इस रोग के लिए एंटीबायोटिक या फफूंदनाशक पर पैसा बर्बाद न करें।|असली इलाज है अधिकारी की सलाह से संक्रमित पौधे हटाना, औज़ार कीटाणुरहित करना और फैलाव रोकना।"),
    "fungal": (
        "Protective fungicides commonly advised: Mancozeb (about 2 to 2.5 g per litre of water) or Copper oxychloride. Spray both sides of the leaves.|If the disease spreads fast, the officer may advise a systemic fungicide such as Propiconazole, Hexaconazole or Azoxystrobin. Use only the one the officer chooses.|Repeat after 10 to 15 days (7 to 10 days in wet weather) and alternate products.|Follow the label for the dose and the waiting period before harvest.",
        "सामान्यतः सुचवली जाणारी संरक्षक बुरशीनाशके: मॅन्कोझेब (सुमारे २ ते २.५ ग्रॅम प्रति लिटर पाणी) किंवा कॉपर ऑक्सिक्लोराईड. पानांच्या दोन्ही बाजूंना फवारा.|रोग वेगाने पसरत असल्यास अधिकारी प्रोपिकोनाझोल, हेक्साकोनाझोल किंवा अझॉक्सिस्ट्रोबिनसारखे आंतरप्रवाही बुरशीनाशक सुचवू शकतात. अधिकारी सांगतील तेच वापरा.|१० ते १५ दिवसांनी (ओल्या हवामानात ७ ते १० दिवसांनी) पुन्हा फवारा आणि औषधे बदलून वापरा.|मात्रा व काढणीपूर्वीचा प्रतीक्षा कालावधी लेबलप्रमाणे पाळा.",
        "आमतौर पर सुझाए जाने वाले सुरक्षात्मक फफूंदनाशक: मैन्कोज़ेब (लगभग 2 से 2.5 ग्राम प्रति लीटर पानी) या कॉपर ऑक्सीक्लोराइड। पत्तियों के दोनों तरफ़ छिड़कें।|रोग तेज़ी से फैल रहा हो तो अधिकारी प्रोपिकोनाज़ोल, हेक्साकोनाज़ोल या एज़ोक्सीस्ट्रोबिन जैसा अंतः-प्रवाही फफूंदनाशक बता सकते हैं। वही इस्तेमाल करें जो अधिकारी चुनें।|10 से 15 दिन बाद (गीले मौसम में 7 से 10 दिन बाद) दोबारा छिड़कें और दवाएँ बदलते रहें।|मात्रा और कटाई से पहले का इंतज़ार-काल लेबल के अनुसार मानें।"),
    "late_blight": (
        "Spray protective Mancozeb (about 2 g per litre of water) at the first sign of disease, or even before it in cool, wet weather.|Then, as the officer advises, spray Metalaxyl + Mancozeb or Cymoxanil + Mancozeb, repeating every 7 to 10 days while the weather stays wet.|Do not use the same product every time; alternate to prevent resistance.|Follow the label for the dose and the waiting period before harvest.",
        "रोगाची पहिली लक्षणे दिसताच, किंवा थंड ओल्या हवामानात त्यापूर्वीच, संरक्षक मॅन्कोझेब (सुमारे २ ग्रॅम प्रति लिटर पाणी) फवारा.|नंतर अधिकाऱ्यांच्या सल्ल्याने मेटॅलॅक्सिल + मॅन्कोझेब किंवा सायमॉक्झॅनिल + मॅन्कोझेब फवारा; हवामान ओले राहिल्यास दर ७ ते १० दिवसांनी पुन्हा करा.|दर वेळी तेच औषध वापरू नका; प्रतिकारशक्ती तयार होऊ नये म्हणून बदलून वापरा.|मात्रा व काढणीपूर्वीचा प्रतीक्षा कालावधी लेबलप्रमाणे पाळा.",
        "रोग के पहले लक्षण दिखते ही, या ठंडे गीले मौसम में उससे पहले ही, सुरक्षात्मक मैन्कोज़ेब (लगभग 2 ग्राम प्रति लीटर पानी) छिड़कें।|फिर अधिकारी की सलाह से मेटालैक्सिल + मैन्कोज़ेब या साइमोक्सानिल + मैन्कोज़ेब छिड़कें; मौसम गीला रहे तो हर 7 से 10 दिन में दोहराएँ।|हर बार वही दवा न दोहराएँ; प्रतिरोध न बने इसलिए बदलते रहें।|मात्रा और कटाई से पहले का इंतज़ार-काल लेबल के अनुसार मानें।"),
    "powdery": (
        "Wettable Sulphur is commonly advised (about 2 to 3 g per litre of water). Do not spray sulphur in very hot weather, above about 32 degrees, because it can burn the leaves.|For severe cases the officer may advise a systemic option such as Hexaconazole or Azoxystrobin.|Neem oil spray can help at the early stage.|Repeat after 10 to 15 days and follow the label.",
        "वेटेबल सल्फर सामान्यतः सुचवले जाते (सुमारे २ ते ३ ग्रॅम प्रति लिटर पाणी). खूप उष्ण हवामानात, सुमारे ३२ अंशांपेक्षा जास्त तापमानात, सल्फर फवारू नका; पाने जळू शकतात.|जास्त प्रादुर्भावात अधिकारी हेक्साकोनाझोल किंवा अझॉक्सिस्ट्रोबिनसारखा आंतरप्रवाही पर्याय सुचवू शकतात.|सुरुवातीच्या टप्प्यात कडुनिंब तेलाची फवारणी उपयोगी ठरू शकते.|१० ते १५ दिवसांनी पुन्हा फवारा आणि लेबल पाळा.",
        "वेटेबल सल्फर आमतौर पर सुझाया जाता है (लगभग 2 से 3 ग्राम प्रति लीटर पानी)। बहुत गर्म मौसम में, लगभग 32 डिग्री से ऊपर, सल्फर न छिड़कें; पत्तियाँ जल सकती हैं।|ज़्यादा प्रकोप में अधिकारी हेक्साकोनाज़ोल या एज़ोक्सीस्ट्रोबिन जैसा अंतः-प्रवाही विकल्प बता सकते हैं।|शुरुआती अवस्था में नीम तेल का छिड़काव मदद कर सकता है।|10 से 15 दिन बाद दोहराएँ और लेबल का पालन करें।"),
    "bacterial": (
        "Copper-based sprays such as Copper oxychloride are commonly advised; sometimes Streptocycline is added on the officer's advice. They slow the spread but cannot cure leaves that are already damaged.|Spray both sides of the leaves; repeat after 7 to 10 days if the weather stays wet.|Ordinary fungicides do not work on bacteria; ask the officer before buying anything.|Follow the label for the dose and the waiting period before harvest.",
        "कॉपर ऑक्सिक्लोराईडसारखी तांबेयुक्त औषधे सामान्यतः सुचवली जातात; अधिकाऱ्यांच्या सल्ल्याने कधी कधी स्ट्रेप्टोसायक्लिन मिसळतात. ती प्रसार कमी करतात, पण आधीच खराब झालेली पाने बरी करू शकत नाहीत.|पानांच्या दोन्ही बाजूंना फवारा; हवामान ओले राहिल्यास ७ ते १० दिवसांनी पुन्हा करा.|साध्या बुरशीनाशकांचा जिवाणूंवर परिणाम होत नाही; काहीही खरेदी करण्यापूर्वी अधिकाऱ्यांना विचारा.|मात्रा व काढणीपूर्वीचा प्रतीक्षा कालावधी लेबलप्रमाणे पाळा.",
        "कॉपर ऑक्सीक्लोराइड जैसे तांबे वाले स्प्रे आमतौर पर सुझाए जाते हैं; अधिकारी की सलाह से कभी-कभी स्ट्रेप्टोसाइक्लिन मिलाई जाती है। ये फैलाव धीमा करते हैं, पर पहले से खराब पत्तियों को ठीक नहीं कर सकते।|पत्तियों के दोनों तरफ़ छिड़कें; मौसम गीला रहे तो 7 से 10 दिन बाद दोहराएँ।|साधारण फफूंदनाशक जीवाणु पर असर नहीं करते; कुछ भी खरीदने से पहले अधिकारी से पूछें।|मात्रा और कटाई से पहले का इंतज़ार-काल लेबल के अनुसार मानें।"),
    "viral": (
        "No medicine cures a virus.|Sprays are used only against the insects that spread it (whitefly, aphids, thrips). Neem oil is a safe first option; if the officer advises, insecticides such as Imidacloprid or Thiamethoxam can be used.|Do not spray insecticides when plants are flowering and bees are visiting.|Follow the label.",
        "विषाणूवर कोणतेही औषध इलाज करत नाही.|फवारण्या फक्त तो पसरवणाऱ्या कीटकांवर (पांढरी माशी, मावा, फुलकिडे) वापरतात. कडुनिंब तेल हा सुरक्षित पहिला पर्याय आहे; अधिकाऱ्यांनी सांगितल्यास इमिडाक्लोप्रिड किंवा थायामेथोक्झामसारखी कीटकनाशके वापरता येतात.|झाडांना फुले असताना व मधमाश्या येत असताना कीटकनाशके फवारू नका.|लेबल पाळा.",
        "वायरस का कोई दवा इलाज नहीं करती।|छिड़काव केवल उन कीटों पर होता है जो इसे फैलाते हैं (सफ़ेद मक्खी, माहू, थ्रिप्स)। नीम तेल सुरक्षित पहला विकल्प है; अधिकारी की सलाह हो तो इमिडाक्लोप्रिड या थायमेथोक्सम जैसे कीटनाशक इस्तेमाल हो सकते हैं।|फूल आने और मधुमक्खियों के आने के समय कीटनाशक न छिड़कें।|लेबल का पालन करें।"),
    "pest": (
        "First wash the mites off with a strong water jet (see What to do). If they continue, the officer may advise Wettable Sulphur, Abamectin or Spiromesifen.|Spray the underside of the leaves, where the mites live.|Neem oil can help at the early stage.|Alternate products and follow the label.",
        "आधी जोरदार पाण्याच्या फवाऱ्याने कोळी धुवून काढा (काय करावे पहा). तरीही राहिल्यास अधिकारी वेटेबल सल्फर, अबामेक्टिन किंवा स्पायरोमेसिफेन सुचवू शकतात.|पानांच्या खालच्या बाजूला फवारा, जिथे कोळी राहतात.|सुरुवातीच्या टप्प्यात कडुनिंब तेल उपयोगी ठरू शकते.|औषधे बदलून वापरा आणि लेबल पाळा.",
        "पहले पानी की तेज़ धार से माइट धो दें (क्या करें देखें)। फिर भी रहें तो अधिकारी वेटेबल सल्फर, एबामेक्टिन या स्पाइरोमेसिफेन बता सकते हैं।|पत्तियों के नीचे की ओर छिड़कें, जहाँ माइट रहते हैं।|शुरुआती अवस्था में नीम तेल मदद कर सकता है।|दवाएँ बदलते रहें और लेबल का पालन करें।"),
    "vector": (
        "No medicine cures citrus greening.|Sprays such as Imidacloprid or Thiamethoxam, on the officer's advice, are used only to control the psyllid insect.|Removing infected trees and planting healthy saplings is the real solution.|Follow the label.",
        "सिट्रस ग्रीनिंगवर कोणतेही औषध इलाज करत नाही.|अधिकाऱ्यांच्या सल्ल्याने इमिडाक्लोप्रिड किंवा थायामेथोक्झामसारख्या फवारण्या फक्त सायला किडीच्या नियंत्रणासाठी वापरतात.|बाधित झाडे काढणे आणि निरोगी रोपे लावणे हाच खरा उपाय आहे.|लेबल पाळा.",
        "सिट्रस ग्रीनिंग का कोई दवा इलाज नहीं करती।|अधिकारी की सलाह से इमिडाक्लोप्रिड या थायमेथोक्सम जैसे स्प्रे केवल साइला कीट को रोकने के लिए इस्तेमाल होते हैं।|संक्रमित पेड़ हटाना और स्वस्थ पौध लगाना ही असली समाधान है।|लेबल का पालन करें।"),
}
KB["wilt"] = {
    "about": (
        "This looks like a wilt disease (for example Fusarium or Verticillium wilt). The fungus lives in the soil and blocks the water vessels of the plant, so leaves turn yellow and droop even when there is enough water. It cannot be cured once the plant is infected.",
        "हा मर रोग (उदा. फ्युजेरियम किंवा व्हर्टिसिलियम मर) असल्याचे दिसते. बुरशी जमिनीत राहते आणि झाडाच्या पाणी वाहून नेणाऱ्या नलिका बंद करते, त्यामुळे पाणी असूनही पाने पिवळी पडून लोंबकळतात. झाडाला लागण झाल्यावर तो बरा करता येत नाही.",
        "यह उकठा (मुरझान) रोग लगता है (जैसे फ्यूज़ेरियम या वर्टिसिलियम उकठा)। फफूंद मिट्टी में रहती है और पौधे की पानी ले जाने वाली नलिकाएँ बंद कर देती है, इसलिए पानी होने पर भी पत्तियाँ पीली होकर लटक जाती हैं। पौधे में लगने के बाद इसे ठीक नहीं किया जा सकता।"),
    "treat": (
        "Cut the main stem of a wilting plant near the ground: brown streaks inside point to wilt. Show it to your agriculture officer to confirm.|Remove and destroy infected plants with their roots, away from the field. Do not make compost from them.|Do not move soil, water or tools from the sick patch to healthy areas.|Avoid waterlogging; improve drainage in the affected patch.|Do not grow cotton, tomato or related crops in that patch for 2 to 3 seasons.",
        "मर येत असलेल्या झाडाचे मुख्य खोड जमिनीजवळ कापा: आतून तपकिरी रेषा दिसल्यास मर रोगाची शक्यता आहे. कृषी अधिकाऱ्यांना दाखवून खात्री करा.|बाधित झाडे मुळांसह उपटून शेतापासून दूर नष्ट करा. त्यांचे कंपोस्ट करू नका.|आजारी भागातील माती, पाणी किंवा अवजारे निरोगी भागात नेऊ नका.|पाणी साचू देऊ नका; बाधित भागात निचरा सुधारा.|त्या भागात २ ते ३ हंगाम कापूस, टोमॅटो किंवा संबंधित पिके घेऊ नका.",
        "मुरझा रहे पौधे का मुख्य तना ज़मीन के पास काटें: अंदर भूरी धारियाँ दिखें तो उकठा की संभावना है। कृषि अधिकारी को दिखाकर पुष्टि करें।|संक्रमित पौधों को जड़ समेत उखाड़कर खेत से दूर नष्ट करें। उनकी खाद न बनाएँ।|बीमार हिस्से की मिट्टी, पानी या औज़ार स्वस्थ हिस्से में न ले जाएँ।|जलभराव न होने दें; प्रभावित हिस्से में जल निकासी सुधारें।|उस हिस्से में 2 से 3 मौसम तक कपास, टमाटर या संबंधित फसलें न लगाएँ।"),
    "prevent": (
        "Use wilt-resistant varieties and certified, treated seed.|Rotate with cereals such as maize or sorghum for 2 to 3 seasons.|Keep fields well drained and avoid injuring roots when weeding.|Clean tools and boots when moving between fields.",
        "मर-प्रतिकारक वाण आणि प्रमाणित, प्रक्रिया केलेले बियाणे वापरा.|२ ते ३ हंगाम मका किंवा ज्वारीसारख्या तृणधान्यांसह पीक फेरपालट करा.|शेतात चांगला निचरा ठेवा आणि निंदणी करताना मुळांना इजा करू नका.|एका शेतातून दुसऱ्या शेतात जाताना अवजारे व पादत्राणे स्वच्छ करा.",
        "उकठा-प्रतिरोधी किस्में और प्रमाणित, उपचारित बीज इस्तेमाल करें।|2 से 3 मौसम तक मक्का या ज्वार जैसी अनाज फसलों के साथ फसल चक्र अपनाएँ।|खेत में अच्छी जल निकासी रखें और निराई में जड़ों को चोट न पहुँचाएँ।|एक खेत से दूसरे में जाते समय औज़ार और जूते साफ करें।"),
}
KB["insect"] = {
    "about": (
        "This looks like damage from sucking or chewing insects (such as leafhoppers/jassids or caterpillars). They scrape or suck the leaves, causing yellow, curled, burnt-looking or holed leaves, and some also spread virus diseases.",
        "हे रस शोषणाऱ्या किंवा पाने कुरतडणाऱ्या किडींचे (जसे तुडतुडे/जॅसिड किंवा अळ्या) नुकसान असल्याचे दिसते. ते पाने खरवडतात किंवा रस शोषतात, त्यामुळे पाने पिवळी, वळलेली, करपल्यासारखी किंवा छिद्रे पडलेली दिसतात; काही कीटक विषाणूजन्य रोगही पसरवतात.",
        "यह रस चूसने या पत्ती कुतरने वाले कीटों (जैसे जैसिड/हरा तेला या इल्लियाँ) का नुकसान लगता है। ये पत्तियों को खुरचते या रस चूसते हैं, जिससे पत्तियाँ पीली, मुड़ी, झुलसी हुई या छेद वाली दिखती हैं; कुछ कीट वायरस रोग भी फैलाते हैं।"),
    "treat": (
        "Early in the morning, look at the underside of leaves and the growing tips for small insects or caterpillars.|Pick off and destroy caterpillars and heavily damaged leaves by hand where possible.|Put up yellow sticky traps for sucking insects and, for caterpillars, pheromone traps if your officer advises.|Remove weeds around the field, because insects hide on them.|Check the whole field again after 5 to 7 days.",
        "सकाळी लवकर पानांची खालची बाजू व शेंड्यांवर लहान कीटक किंवा अळ्या आहेत का ते पाहा.|शक्य असल्यास अळ्या आणि जास्त खराब झालेली पाने हाताने वेचून नष्ट करा.|रस शोषणाऱ्या किडींसाठी पिवळे चिकट सापळे लावा आणि अळ्यांसाठी अधिकाऱ्यांनी सांगितल्यास कामगंध सापळे लावा.|शेताभोवतीचे तण काढा, कारण कीटक त्यावर लपतात.|५ ते ७ दिवसांनी संपूर्ण शेत पुन्हा तपासा.",
        "सुबह जल्दी पत्तियों के नीचे और बढ़ते सिरों पर छोटे कीट या इल्लियाँ देखें।|संभव हो तो इल्लियाँ और ज़्यादा खराब पत्तियाँ हाथ से चुनकर नष्ट करें।|रस चूसने वाले कीटों के लिए पीले चिपचिपे जाल लगाएँ और इल्लियों के लिए अधिकारी की सलाह हो तो फेरोमोन जाल लगाएँ।|खेत के आसपास के खरपतवार हटाएँ, क्योंकि कीट उन पर छिपते हैं।|5 से 7 दिन बाद पूरा खेत फिर जाँचें।"),
    "prevent": (
        "Scout the field every week, especially in the first 2 months.|Keep field edges free of weeds, and grow a border crop if your officer advises.|Protect helpful insects such as ladybirds and spiders by avoiding unnecessary spraying.|Use healthy seedlings and clear crop residue after harvest.",
        "विशेषतः पहिल्या २ महिन्यांत शेताची दर आठवड्याला पाहणी करा.|बांध तणमुक्त ठेवा आणि अधिकाऱ्यांनी सांगितल्यास बांधावर सापळा पीक लावा.|अनावश्यक फवारणी टाळून लेडीबर्ड व कोळी यांसारख्या मित्रकीटकांचे संरक्षण करा.|निरोगी रोपे वापरा आणि काढणीनंतर पिकाचे अवशेष साफ करा.",
        "खासकर पहले 2 महीनों में हर हफ़्ते खेत का निरीक्षण करें।|मेड़ें खरपतवार-मुक्त रखें और अधिकारी की सलाह हो तो मेड़ पर ट्रैप फसल लगाएँ।|बेवजह छिड़काव से बचकर लेडीबर्ड और मकड़ी जैसे मित्र कीटों को बचाएँ।|स्वस्थ पौध लगाएँ और कटाई के बाद फसल अवशेष साफ करें।"),
}
KB["nutrient"] = {
    "about": (
        "This looks like a nutrient problem or a physiological disorder, not an infection. Reddening, yellowing or odd colour patterns often point to a lack of potassium, magnesium or nitrogen, or to water stress. Insects and viruses can look similar, so check carefully.",
        "ही रोगाची लागण नसून अन्नद्रव्यांची कमतरता किंवा शरीरक्रियात्मक विकृती असल्याचे दिसते. पाने लाल होणे, पिवळी पडणे किंवा विचित्र रंगांचे पट्टे हे अनेकदा पोटॅश, मॅग्नेशियम किंवा नत्राची कमतरता किंवा पाण्याचा ताण दर्शवतात. कीड व विषाणूही असेच दिसू शकतात, म्हणून काळजीपूर्वक तपासा.",
        "यह संक्रमण नहीं बल्कि पोषक तत्वों की कमी या शारीरिक विकार लगता है। पत्तियों का लाल होना, पीला पड़ना या अजीब रंग के पैटर्न अक्सर पोटाश, मैग्नीशियम या नाइट्रोजन की कमी या पानी के तनाव का संकेत होते हैं। कीट और वायरस भी ऐसे दिख सकते हैं, इसलिए ध्यान से जाँचें।"),
    "treat": (
        "First check the underside of leaves for insects, so you are sure it is not a pest.|Take a soil test at your nearest soil testing lab or Krishi Vigyan Kendra (KVK).|Apply fertiliser only as the soil test and your officer advise; more is not better.|Keep soil moisture even, avoiding both drought and waterlogging.|Look again after 10 to 14 days to see whether the new leaves are healthy.",
        "आधी पानांच्या खालच्या बाजूला किडी आहेत का ते पाहा, म्हणजे ही कीड नाही याची खात्री होईल.|जवळच्या माती परीक्षण प्रयोगशाळेत किंवा कृषी विज्ञान केंद्रात (KVK) माती परीक्षण करा.|माती परीक्षण व अधिकाऱ्यांच्या सल्ल्यानुसारच खत द्या; जास्त म्हणजे चांगले नाही.|जमिनीत ओलावा एकसारखा ठेवा; दुष्काळ आणि पाणी साचणे दोन्ही टाळा.|१० ते १४ दिवसांनी पुन्हा पाहा, नवीन पाने निरोगी येत आहेत का.",
        "पहले पत्तियों के नीचे कीट देखें ताकि पक्का हो कि यह कीट का असर नहीं है।|नज़दीकी मिट्टी जाँच प्रयोगशाला या कृषि विज्ञान केंद्र (KVK) में मिट्टी की जाँच कराएँ।|खाद केवल मिट्टी जाँच और अधिकारी की सलाह के अनुसार दें; ज़्यादा डालना बेहतर नहीं है।|मिट्टी में नमी एक-सी रखें; सूखा और जलभराव दोनों से बचें।|10 से 14 दिन बाद फिर देखें कि नई पत्तियाँ स्वस्थ आ रही हैं या नहीं।"),
    "prevent": (
        "Do a soil test every year or two.|Use balanced fertiliser together with farmyard manure or compost.|Keep irrigation regular and drainage good.|Choose varieties suited to your soil and season.",
        "दर एक-दोन वर्षांनी माती परीक्षण करा.|शेणखत किंवा कंपोस्टसह संतुलित खते वापरा.|सिंचन नियमित व निचरा चांगला ठेवा.|तुमच्या जमिनीला व हंगामाला योग्य वाण निवडा.",
        "हर एक-दो साल में मिट्टी की जाँच कराएँ।|गोबर की खाद या कम्पोस्ट के साथ संतुलित खाद दें।|सिंचाई नियमित और जल निकासी अच्छी रखें।|अपनी मिट्टी और मौसम के अनुकूल किस्में चुनें।"),
}
MEDICINE.update({
    "wilt": (
        "No spray cures a wilt once the plant is infected.|To protect healthy plants, your officer may advise a soil treatment with a beneficial fungus (Trichoderma) or a seed treatment before sowing.|Do not spend money on leaf sprays for this disease.|Follow the label for any product.",
        "झाडाला लागण झाल्यावर कोणतीही फवारणी मर रोग बरा करत नाही.|निरोगी झाडांच्या संरक्षणासाठी अधिकारी जमिनीत उपयुक्त बुरशी (ट्रायकोडर्मा) देण्याचा किंवा पेरणीपूर्वी बीजप्रक्रिया करण्याचा सल्ला देऊ शकतात.|या रोगासाठी पानांवरील फवारण्यांवर पैसे खर्च करू नका.|कोणत्याही औषधासाठी लेबल पाळा.",
        "पौधे में लगने के बाद कोई छिड़काव उकठा को ठीक नहीं करता।|स्वस्थ पौधों की सुरक्षा के लिए अधिकारी मिट्टी में लाभकारी फफूंद (ट्राइकोडर्मा) डालने या बुवाई से पहले बीज उपचार की सलाह दे सकते हैं।|इस रोग के लिए पत्तियों पर छिड़काव में पैसा बर्बाद न करें।|किसी भी उत्पाद के लिए लेबल का पालन करें।"),
    "insect": (
        "Neem oil or neem seed kernel extract is a safe first spray for young insects.|If the damage is above the level your officer sets, they may advise an insecticide suitable for the insect you found. Do not choose one on your own.|Spray in the evening and avoid flowering time, when bees are active.|Alternate products and follow the label.",
        "लहान कीटकांसाठी कडुनिंब तेल किंवा निंबोळी अर्क ही सुरक्षित पहिली फवारणी आहे.|नुकसान अधिकाऱ्यांनी ठरवलेल्या पातळीपेक्षा जास्त असल्यास ते तुम्हाला आढळलेल्या किडीला योग्य कीटकनाशक सुचवू शकतात. स्वतःहून निवडू नका.|संध्याकाळी फवारा आणि मधमाश्या येत असलेल्या फुलोऱ्याच्या काळात टाळा.|औषधे बदलून वापरा आणि लेबल पाळा.",
        "छोटे कीटों के लिए नीम तेल या नीम बीज अर्क सुरक्षित पहला छिड़काव है।|नुकसान अधिकारी द्वारा तय स्तर से ज़्यादा हो तो वे आपको मिले कीट के अनुसार कीटनाशक बता सकते हैं। खुद से न चुनें।|शाम को छिड़कें और फूल आने के समय से बचें, जब मधुमक्खियाँ आती हैं।|दवाएँ बदलते रहें और लेबल का पालन करें।"),
    "nutrient": (
        "Pesticides are usually not needed for this problem.|Your officer may advise a specific fertiliser or a leaf nutrient spray (for example potassium or magnesium) based on the soil test.|Do not spray pesticides unless you have actually found insects or disease.|Follow the label.",
        "या समस्येसाठी सहसा कीटकनाशके/बुरशीनाशके लागत नाहीत.|माती परीक्षणानुसार अधिकारी विशिष्ट खत किंवा पानांवरील अन्नद्रव्य फवारणी (उदा. पोटॅश किंवा मॅग्नेशियम) सुचवू शकतात.|प्रत्यक्ष कीड किंवा रोग आढळल्याशिवाय कीटकनाशके फवारू नका.|लेबल पाळा.",
        "इस समस्या के लिए आमतौर पर कीटनाशक/फफूंदनाशक की ज़रूरत नहीं होती।|मिट्टी जाँच के आधार पर अधिकारी कोई खास खाद या पत्तियों पर पोषक तत्व का छिड़काव (जैसे पोटाश या मैग्नीशियम) बता सकते हैं।|जब तक कीट या रोग सच में न मिले, कीटनाशक न छिड़कें।|लेबल का पालन करें।"),
})

for _key, _entry in KB.items():
    _entry["medicine"] = MEDICINE[_key]

FOLLOWUP = (
    "If it keeps spreading after 7 days, or many plants show it, call your agriculture officer or the Kisan Call Centre (1800-180-1551).",
    "७ दिवसांनंतरही रोग पसरत असेल किंवा अनेक झाडांवर दिसत असेल, तर कृषी अधिकारी किंवा किसान कॉल सेंटरला (1800-180-1551) फोन करा.",
    "7 दिन बाद भी रोग फैल रहा हो या कई पौधों पर दिखे, तो कृषि अधिकारी या किसान कॉल सेंटर (1800-180-1551) को फोन करें।",
)
for _key, _entry in KB.items():
    if "healthy" not in _key and _key not in ("banana_xanthomonas", "vector"):
        _entry["treat"] = tuple(t + "|" + f for t, f in zip(_entry["treat"], FOLLOWUP))

SAFETY = (
    "Safety while spraying: wear gloves, a mask and full-sleeve clothes. Spray in the early morning or evening, not in wind or rain. Keep children and animals away. Buy only from a licensed dealer, follow the dose and the waiting period before harvest given on the label, and never reuse empty containers.",
    "फवारणी करताना सुरक्षा: हातमोजे, मास्क आणि पूर्ण बाह्यांचे कपडे घाला. सकाळी लवकर किंवा संध्याकाळी फवारणी करा, वारा किंवा पावसात करू नका. मुले व जनावरे दूर ठेवा. परवानाधारक विक्रेत्याकडूनच औषध घ्या, लेबलवरील मात्रा व काढणीपूर्वीचा प्रतीक्षा कालावधी पाळा आणि रिकामे डबे पुन्हा वापरू नका.",
    "छिड़काव के समय सुरक्षा: दस्ताने, मास्क और पूरी बाँह के कपड़े पहनें। सुबह जल्दी या शाम को छिड़काव करें, हवा या बारिश में नहीं। बच्चों और पशुओं को दूर रखें। केवल लाइसेंसधारी दुकानदार से दवा लें, लेबल पर लिखी मात्रा और कटाई से पहले का इंतज़ार-काल मानें और खाली डिब्बे दोबारा इस्तेमाल न करें।",
)

# ======================================================================
# Voice commands: words that tell us what the farmer is asking
# ======================================================================
INTENTS = [
    ("prevent", ["prevent", "avoid", "protect", "precaution", "future", "बचाव", "प्रतिबंध",
                 "टाळ", "खबरदारी", "रोकथाम", "बचना", "बचें"]),
    ("medicine", ["medicine", "spray", "pesticide", "fungicide", "insecticide", "chemical",
                  "drug", "dose", "औषध", "फवारणी", "कीटकनाशक", "बुरशीनाशक", "रसायन", "दवा",
                  "दवाई", "स्प्रे", "कीटनाशक", "फफूंदनाशक", "छिड़काव"]),
    ("treat", ["treat", "cure", "solution", "remedy", "fix", "what to do", "what should i do",
               "help", "उपचार", "काय करावे", "काय करू", "उपाय", "इलाज", "क्या करें",
               "क्या करना", "समाधान"]),
    ("all", ["read all", "everything", "full advice", "all advice", "सर्व", "संपूर्ण",
             "सब कुछ", "पूरी सलाह", "पूरा"]),
    ("about", ["what is", "which disease", "disease", "cause", "why", "symptom", "about",
               "problem", "रोग", "कारण", "लक्षण", "काय आहे", "कोणता", "क्या है", "बीमारी",
               "समस्या", "कौन सा"]),
]


def detect_topic(text):
    text = text.lower()
    for topic, words in INTENTS:
        if any(word in text for word in words):
            return topic
    return None


# ======================================================================
# Helpers
# ======================================================================
with st.container(key="card_lang"):
    lang = st.radio("Language", LANGS, horizontal=True, label_visibility="collapsed")


def tt(key):
    return UI[key][LI[lang]]


def tidy(text):
    return " ".join(text.replace("_", " ").split())


def category(condition):
    c = condition.lower()
    if "healthy" in c:
        return "healthy"
    if "virus" in c or "necrosis" in c or "mln" in c:
        return "viral"
    if "bacterial" in c:
        return "bacterial"
    if "wilt" in c:
        return "wilt"
    if "hopper" in c or "jassid" in c or "caterpillar" in c or "diabrotica" in c:
        return "insect"
    if "spider" in c or "mite" in c:
        return "pest"
    if "haunglongbing" in c or "greening" in c:
        return "vector"
    if "late blight" in c:
        return "late_blight"
    if "powdery" in c:
        return "powdery"
    if "reddening" in c or "variegation" in c or "deficiency" in c:
        return "nutrient"
    return "fungal"


@st.cache_resource(show_spinner=False)
def load_model(path, default_classes):
    checkpoint = torch.load(path, map_location=DEVICE, weights_only=True)
    state_dict = checkpoint.get("state_dict", checkpoint.get("model_state_dict", checkpoint))
    classes = checkpoint.get("class_names", checkpoint.get("classes")) or default_classes
    if classes is None:
        raise ValueError("Class names are missing from this checkpoint.")
    model = getattr(models, checkpoint.get("arch", "resnet18"))(weights=None)
    model.fc = nn.Linear(model.fc.in_features, len(classes))
    model.load_state_dict(state_dict, strict=True)
    model.to(DEVICE).eval()
    return model, list(classes), float(checkpoint.get("temperature", 1.0))


@st.cache_resource(show_spinner=False)
def field_classes():
    path = MODEL_DIR / MODEL_OPTIONS["field"]["file"]
    if not path.exists():
        return ()
    try:
        checkpoint = torch.load(str(path), map_location="cpu", weights_only=True)
        return tuple(checkpoint.get("class_names", ()))
    except Exception:
        return ()


def available_crops():
    have = {name.split("___")[0] for name in field_classes()}
    field = [key for key, value in FIELD_CROPS.items() if value[0] in have]
    plant_village = [key for key in PV_CROPS if not (key == "corn" and "maize" in field)]
    return ["banana"] + field + plant_village + ["any"]


def split_text(text, limit=280):
    chunks, current = [], ""
    for sentence in re.split(r"(?<=[.।!?])\s+", text):
        if current and len(current) + len(sentence) + 1 > limit:
            chunks.append(current)
            current = sentence
        else:
            current = (current + " " + sentence).strip()
    if current:
        chunks.append(current)
    return chunks or [text]


def tts_chunk(job):
    text, code = job
    buffer = BytesIO()
    gTTS(text=text, lang=code).write_to_fp(buffer)
    return buffer.getvalue()


@st.cache_data(show_spinner=False)
def make_audio(text, code):
    """Speak long advice quickly: convert pieces in parallel, then join the audio."""
    jobs = [(chunk, code) for chunk in split_text(text)]
    with ThreadPoolExecutor(max_workers=6) as pool:
        return b"".join(pool.map(tts_chunk, jobs))


def play(text, autoplay=False):
    try:
        data = make_audio(text, TTS_CODES[lang])
    except Exception as error:
        st.caption(f"🔇 Audio needs internet: {error}")
        return
    try:
        st.audio(data, format="audio/mp3", autoplay=autoplay)
    except TypeError:
        st.audio(data, format="audio/mp3")


def show_image(image):
    try:
        st.image(image, use_container_width=True)
    except TypeError:
        st.image(image)


def steps_text(raw):
    parts = []
    for step in raw.split("|"):
        step = step.strip()
        if step and step[-1] not in ".।!?":
            step += "."
        parts.append(step)
    return " ".join(parts)


def numbered(raw):
    """Read-aloud style: 'Step 1: ... Step 2: ...'"""
    parts = []
    for n, step in enumerate(raw.split("|"), start=1):
        step = step.strip()
        if step and step[-1] not in ".।!?":
            step += "."
        parts.append(f"{tt('step_word')} {n}: {step}")
    return " ".join(parts)


def softmax(values):
    top = max(values)
    exps = [math.exp(v - top) for v in values]
    total = sum(exps)
    return [v / total for v in exps]


def average(rows):
    return [sum(column) / len(rows) for column in zip(*rows)]


def class_probs(logit_rows, allowed=None):
    """Average over several views. If `allowed` is given, only those classes compete.
    Returns (probabilities, probability mass the model put on classes outside `allowed`)."""
    full = average([softmax(row) for row in logit_rows])
    if not allowed:
        return full, 0.0
    restricted = []
    for row in logit_rows:
        sub = softmax([row[i] for i in allowed])
        out = [0.0] * len(row)
        for i, value in zip(allowed, sub):
            out[i] = value
        restricted.append(out)
    return average(restricted), 1.0 - sum(full[i] for i in allowed)


def square_views(image):
    """The training photos were squares, so never squash a wide or tall photo."""
    w, h = image.size
    side = min(w, h)
    left, top = (w - side) // 2, (h - side) // 2
    center = image.crop((left, top, left + side, top + side))
    fill = tuple(int(v) for v in ImageStat.Stat(image).median)
    padded = ImageOps.pad(image, (max(w, h), max(w, h)), color=fill)
    return [center, padded, ImageOps.mirror(center), ImageOps.mirror(padded)]


def predict(image, model_key, crop_key=None):
    option = MODEL_OPTIONS[model_key]
    path = MODEL_DIR / option["file"]
    if not path.exists():
        raise FileNotFoundError(f"Missing model file: {path.name}")
    model, classes, temperature = load_model(str(path), option["default_classes"])

    views = square_views(image) if model_key in ("plantvillage", "field") else [image]
    batch = torch.stack([TRANSFORM(v) for v in views]).to(DEVICE)
    with torch.no_grad():
        logit_rows = [[v / temperature for v in row] for row in model(batch).cpu().tolist()]

    prefix = None
    if model_key == "plantvillage" and crop_key in PV_CROPS:
        prefix = PV_CROPS[crop_key][0]
    elif model_key == "field" and crop_key in FIELD_CROPS:
        prefix = FIELD_CROPS[crop_key][0]
    allowed = [i for i, name in enumerate(classes) if name.split("___")[0] == prefix] if prefix else None

    probs, outside = class_probs(logit_rows, allowed)
    order = sorted(range(len(classes)), key=lambda i: probs[i], reverse=True)
    top3 = [(classes[i], probs[i]) for i in order[:3]]

    other = None
    if allowed:
        full = average([softmax(row) for row in logit_rows])
        other = classes[max(range(len(classes)), key=lambda i: full[i])]
    return {"top3": top3, "outside": outside, "other": other}


def build_result(prediction, crop):
    top3 = prediction["top3"]
    top_label, top_score = top3[0]
    second = top3[1][1] if len(top3) > 1 else 0.0

    # The AI mostly believes this is a different crop: do not trust any disease name.
    if prediction["outside"] > 0.5:
        return {"status": "unsure", "top3": top3, "score": top_score,
                "wrong_crop": True, "other": prediction["other"]}
    if top_score < LIKELY_SCORE or (top_score - second) < MIN_MARGIN:
        return {"status": "unsure", "top3": top3, "score": top_score}
    likely = top_score < MIN_SCORE      # an answer, but with a "please confirm" note

    if crop == "banana":
        key = top_label.lower()
        return {"status": "ok" if key == "healthy" else "bad", "kb": "banana_" + key,
                "names": BANANA_NAME[key], "name": None,
                "top3": top3, "score": top_score, "likely": likely}

    crop_name, _, condition = top_label.partition("___")
    cat = category(tidy(condition))
    pretty = f"{tidy(crop_name)} - {tidy(condition)}"
    return {"status": "ok" if cat == "healthy" else "bad", "kb": cat, "names": None,
            "name": pretty, "top3": top3, "score": top_score, "likely": likely}


def crop_label(key):
    if key == "banana":
        return tt("crop_banana")
    if key == "any":
        return tt("crop_any")
    entry = FIELD_CROPS.get(key) or PV_CROPS[key]
    return f"{entry[1]} {entry[2][LI[lang]]}"


def unsure_message(r):
    if r.get("wrong_crop") and r.get("other"):
        prefix = r["other"].split("___")[0]
        other = PREFIX_NAMES.get(prefix, (tidy(prefix),) * 3)[LI[lang]]
        return tt("wrong_crop").format(other=other)
    return tt("unsure_text")


def label_name(label):
    key = label.lower()
    if key in BANANA_NAME:
        return BANANA_NAME[key][LI[lang]]
    crop_name, _, condition = label.partition("___")
    return f"{tidy(crop_name)} - {tidy(condition)}" if condition else tidy(label)


def result_name(r):
    if r.get("names"):
        return r["names"][LI[lang]]
    return r["name"]


def full_advice(r):
    entry, i = KB[r["kb"]], LI[lang]
    parts = [tt("likely_note")] if r.get("likely") else []
    parts += [
        result_name(r) + ".",
        steps_text(entry["about"][i]),
        tt("sp_treat") + ": " + numbered(entry["treat"][i]),
        tt("sp_med") + ": " + numbered(entry["medicine"][i]),
    ]
    if r["status"] != "ok":
        parts.append(SAFETY[i])
    parts.append(tt("sp_prev") + ": " + numbered(entry["prevent"][i]))
    return " ".join(parts)


def build_answer(topic):
    r = st.session_state.get("result")
    if not r or r["status"] == "unsure":
        return tt("need_analysis")
    entry, i = KB[r["kb"]], LI[lang]
    name = result_name(r) + ". "
    if topic == "treat":
        return name + tt("sp_treat") + ": " + numbered(entry["treat"][i])
    if topic == "medicine":
        text = name + tt("sp_med") + ": " + numbered(entry["medicine"][i])
        return text + (" " + SAFETY[i] if r["status"] != "ok" else "")
    if topic == "prevent":
        return name + tt("sp_prev") + ": " + numbered(entry["prevent"][i])
    if topic == "about":
        return name + steps_text(entry["about"][i])
    return full_advice(r)


class SilentAudio(Exception):
    pass


def to_wav16k(data):
    """Convert whatever the browser recorded to 16 kHz mono 16-bit WAV."""
    import os
    import subprocess
    import tempfile
    src = dst = None
    try:
        with tempfile.NamedTemporaryFile(suffix=".in", delete=False) as f:
            f.write(data)
            src = f.name
        dst = src + ".wav"
        result = subprocess.run(
            ["ffmpeg", "-y", "-loglevel", "error", "-i", src, "-ac", "1", "-ar", "16000",
             "-c:a", "pcm_s16le", dst],
            capture_output=True, timeout=30,
        )
        if result.returncode == 0 and os.path.getsize(dst) > 44:
            with open(dst, "rb") as f:
                return f.read()
    except Exception:
        pass
    finally:
        for path in (src, dst):
            if path and os.path.exists(path):
                os.remove(path)
    return data


def audio_stats(wav):
    """Return (seconds, loudness 0-1) of a PCM WAV."""
    import wave
    import numpy as np
    with wave.open(BytesIO(wav)) as w:
        seconds = w.getnframes() / float(w.getframerate())
        frames = w.readframes(w.getnframes())
        width = w.getsampwidth()
    if width != 2 or not frames:
        return seconds, None
    samples = np.frombuffer(frames, dtype=np.int16).astype("float32") / 32768.0
    return seconds, float((samples ** 2).mean() ** 0.5)


def transcribe(data):
    if sr is None:
        raise RuntimeError("SpeechRecognition is not installed.")
    wav = to_wav16k(data)
    try:
        seconds, level = audio_stats(wav)
    except Exception:
        seconds, level = 0.0, None
    if level is not None and level < 0.0015:
        raise SilentAudio(f"{seconds:.1f}s recorded, loudness {level:.4f}")
    recognizer = sr.Recognizer()
    with sr.AudioFile(BytesIO(wav)) as source:
        audio = recognizer.record(source)
    text = recognizer.recognize_google(audio, language=SR_CODES[lang])
    return text, seconds, level


# ======================================================================
# Small display helpers
# ======================================================================
def section(number, key):
    title = tt(key).split(" ", 1)[-1]          # drop the leading number emoji
    st.markdown(
        f'<div class="sec"><span class="num">{number}</span>'
        f'<span class="ttl">{html.escape(title)}</span></div>',
        unsafe_allow_html=True,
    )


def bars_html(top3):
    rows = []
    for label, score in top3:
        color = "#43a047" if "healthy" in label.lower() else "#ef6c00"
        pct = score * 100
        rows.append(
            f'<div class="bar-row"><div>{html.escape(label_name(label))}</div>'
            f'<div class="bar"><div class="fill" style="width:{pct:.0f}%;background:{color}"></div></div>'
            f'<div class="bar-val">{pct:.0f}%</div></div>'
        )
    return f'<div class="bars"><div class="bars-title">{tt("probs")}</div>{"".join(rows)}</div>'


def result_html(r):
    pct = round(r["score"] * 100)
    if r["status"] == "unsure":
        cls, color = "unsure", "#fb8c00"
        pills = f'<span class="pill unsure">{tt("unsure")}</span>'
        name = tt("no_clear")
    else:
        cls = r["status"]
        color = "#2e7d32" if cls == "ok" else "#e53935"
        pills = f'<span class="pill {cls}">{tt("good") if cls == "ok" else tt("bad")}</span>'
        if r.get("likely"):
            pills += f'<span class="pill likely">{tt("likely_chip")}</span>'
        name = result_name(r)
    return (
        f'<div class="res {cls}"><div class="ring" style="--p:{pct};--c:{color}">'
        f'<span>{pct}%</span></div><div><div>{pills}</div>'
        f'<div class="rname">{html.escape(name)}</div>'
        f'<div class="rsub">{tt("conf")}</div></div></div>'
    )


# ======================================================================
# Header + farmer guide
# ======================================================================
with st.container(key="card_head"):
    st.markdown(
        f'<div class="brand"><div class="logo">🌾</div><div><h1>{html.escape(tt("title"))}</h1>'
        f'<p>{html.escape(tt("tagline"))}</p></div></div>'
        f'<img class="hero" src="data:image/svg+xml;base64,{HERO_B64}" alt="farm">',
        unsafe_allow_html=True,
    )

with st.container(key="card_guide"):
    st.markdown(
        f'<div class="farmer-row"><div class="farmer">👨‍🌾</div>'
        f'<div class="bubble">{html.escape(tt("greet"))}</div></div>',
        unsafe_allow_html=True,
    )
    if st.button(tt("hear")):
        play(tt("greet"), autoplay=True)

# ======================================================================
# Steps 1-3: crop, photo, check
# ======================================================================
with st.container(key="card_input"):
    section(1, "step1")
    crop = st.selectbox("crop", available_crops(), label_visibility="collapsed", format_func=crop_label)

    section(2, "step2")
    method = st.radio(
        "method", ["cam", "up"], horizontal=True, label_visibility="collapsed",
        format_func=lambda k: tt("m_cam") if k == "cam" else tt("m_up"),
    )
    st.caption(tt("tip"))
    if method == "cam":
        source = st.camera_input(tt("m_cam"), label_visibility="collapsed")
    else:
        source = st.file_uploader(tt("m_up"), type=["jpg", "jpeg", "png"],
                                  label_visibility="collapsed")

    image = None
    if source is not None:
        raw_bytes = source.getvalue()
        image = ImageOps.exif_transpose(Image.open(BytesIO(raw_bytes))).convert("RGB")
        digest = hashlib.md5(raw_bytes).hexdigest()
        if digest != st.session_state.get("img_hash"):      # a new photo: forget old answer
            st.session_state["img_hash"] = digest
            st.session_state.pop("result", None)
            st.session_state.pop("voice", None)
        if method == "up":
            show_image(image)

    section(3, "step3")
    autospeak = st.toggle(tt("auto_speak"), value=True)
    if st.button(tt("analyze"), type="primary", use_container_width=True):
        if image is None:
            st.warning(tt("need_photo"))
        else:
            try:
                with st.spinner(tt("checking")):
                    model_key = "banana" if crop == "banana" else ("field" if crop in FIELD_CROPS else "plantvillage")
                    prediction = predict(image, model_key, crop)
                st.session_state["result"] = build_result(prediction, crop)
                st.session_state["result"]["crop"] = crop
                st.session_state["speak_now"] = True
                st.session_state.pop("voice", None)
            except Exception as error:
                st.error(f"Prediction failed: {error}")

# ======================================================================
# Result + solution
# ======================================================================
r = st.session_state.get("result")
if r:
    with st.container(key="card_result"):
        section("✓", "result")
        i = LI[lang]
        st.markdown(result_html(r), unsafe_allow_html=True)

        if r["status"] == "unsure":
            st.warning(unsure_message(r))
            st.markdown(bars_html(r["top3"]), unsafe_allow_html=True)
        else:
            if r.get("likely"):
                st.warning(tt("likely_note"))
            others = [(lb, sc) for lb, sc in r["top3"][1:2] if sc >= 0.15]
            if others:
                st.markdown(
                    f'<div class="also">{tt("also")}: <b>{html.escape(label_name(others[0][0]))}</b>'
                    f' ({others[0][1] * 100:.0f}%)</div>',
                    unsafe_allow_html=True,
                )
            st.markdown(bars_html(r["top3"]), unsafe_allow_html=True)

            entry = KB[r["kb"]]
            t_treat, t_med, t_prev, t_about = st.tabs(
                [tt("tab_treat"), tt("tab_med"), tt("tab_prev"), tt("tab_about")]
            )
            with t_treat:
                st.markdown("\n".join(f"{n}. {x}" for n, x in enumerate(entry["treat"][i].split("|"), 1)))
            with t_med:
                st.markdown("\n".join(f"{n}. {x}" for n, x in enumerate(entry["medicine"][i].split("|"), 1)))
                if r["status"] != "ok":
                    st.warning(SAFETY[i])
            with t_prev:
                st.markdown("\n".join(f"- {x}" for x in entry["prevent"][i].split("|")))
            with t_about:
                st.write(entry["about"][i])

            if st.button(tt("listen")):
                play(full_advice(r), autoplay=True)

        if st.session_state.pop("speak_now", False) and autospeak:
            spoken = unsure_message(r) if r["status"] == "unsure" else full_advice(r)
            play(spoken, autoplay=True)

# ======================================================================
# Voice assistant
# ======================================================================
with st.container(key="card_voice"):
    section("🎙", "voice_head")
    st.caption(tt("voice_hint"))

    if hasattr(st, "audio_input"):
        recording = st.audio_input(tt("mic"))
        if recording is not None:
            data = recording.getvalue()
            h = hashlib.md5(data).hexdigest()
            if h != st.session_state.get("voice_hash"):     # process each recording once
                st.session_state["voice_hash"] = h
                try:
                    heard, secs, level = transcribe(data)
                    st.session_state["voice"] = {
                        "heard": heard, "topic": detect_topic(heard), "played": False,
                    }
                except Exception as error:
                    kind = type(error).__name__
                    if kind == "SilentAudio":
                        err = "mic_silent"
                    elif kind == "RequestError":
                        err = "mic_net"
                    else:
                        err = "mic_fail"
                    st.session_state["voice"] = {
                        "heard": None, "topic": "error", "err": err,
                        "detail": f"{kind}: {error}", "played": False,
                    }
    else:
        st.info(tt("mic_old"))

    st.caption(tt("mic_help"))
    st.caption(tt("or_tap"))
    BUTTONS = [("treat", "b_treat"), ("medicine", "b_med"), ("prevent", "b_prev"),
               ("about", "b_about"), ("all", "b_all")]
    for row in (BUTTONS[0:2], BUTTONS[2:4], BUTTONS[4:5]):
        for column, (topic, label) in zip(st.columns(len(row)), row):
            if column.button(tt(label), use_container_width=True, key="btn_" + topic):
                st.session_state["voice"] = {"heard": None, "topic": topic, "played": False}

    typed = st.text_input(tt("type_q"), key="typed_q")
    if typed and typed != st.session_state.get("typed_done"):
        st.session_state["typed_done"] = typed
        st.session_state["voice"] = {"heard": typed, "topic": detect_topic(typed), "played": False}

    voice = st.session_state.get("voice")
    if voice:
        if voice["heard"]:
            st.caption(f'{tt("heard")} "{voice["heard"]}"')
        if voice["topic"] == "error":
            answer = tt(voice.get("err", "mic_fail"))
            st.caption("Details: " + voice.get("detail", ""))
        elif voice["topic"] is None:
            answer = tt("no_understand")
        else:
            answer = build_answer(voice["topic"])
        st.markdown(f'<div class="answer">🗣️ {html.escape(answer)}</div>', unsafe_allow_html=True)
        play(answer, autoplay=not voice["played"])
        voice["played"] = True

# ======================================================================
# Footer
# ======================================================================
with st.container(key="card_foot"):
    st.info(tt("helpline"))
    st.caption(tt("footer"))
'''

# Safety check: never save an app with a syntax error
ast.parse(APP_CODE)
APP_PATH.write_text(APP_CODE, encoding="utf-8")
print("app.py written:", APP_PATH)

# ---------- 5. Start Streamlit ----------
from pyngrok import ngrok
import requests

subprocess.run(["bash", "-lc", f"fuser -k {PORT}/tcp 2>/dev/null || true"],
               capture_output=True)
ngrok.kill()
time.sleep(2)

log_file = open(LOG_PATH, "w")
process = subprocess.Popen(
    [sys.executable, "-m", "streamlit", "run", str(APP_PATH),
     "--server.port", str(PORT),
     "--server.address", "0.0.0.0",
     "--server.headless", "true",
     "--theme.base", "light",
     "--theme.primaryColor", "#2e7d32",
     "--theme.backgroundColor", "#fffaeb",
     "--theme.secondaryBackgroundColor", "#f3e7c3",
     "--theme.textColor", "#3b2a14"],
    stdout=log_file,
    stderr=subprocess.STDOUT,
)

ready = False
for _ in range(40):
    try:
        if requests.get(f"http://127.0.0.1:{PORT}/_stcore/health",
                        timeout=2).status_code == 200:
            ready = True
            break
    except requests.RequestException:
        pass
    time.sleep(2)

if not ready:
    print("Streamlit did not start. Log:\n")
    print(LOG_PATH.read_text(errors="replace")[-3000:])
else:
    # ---------- 6. Public https URL (camera and microphone need https) ----------
    token = None
    try:
        from google.colab import userdata           # Colab "Secrets" (key icon)
        token = userdata.get("NGROK_AUTH_TOKEN")
    except Exception:
        pass
    if not token:
        from getpass import getpass
        token = getpass("Enter your ngrok auth token: ")
    ngrok.set_auth_token(token)

    tunnel = ngrok.connect(f"127.0.0.1:{PORT}", "http")
    print("\n🌾 AgriDoctor AI - Farmer Edition is live:")
    print(tunnel.public_url)
