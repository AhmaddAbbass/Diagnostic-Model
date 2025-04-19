"""
Toy multimodal predictor for end-to-end testing.
Always returns a valid (label, precautions).
"""
import os, random, re
from PIL import Image  # via Pillow

LABELS = ["Acne", "Eczema", "Psoriasis"]
PRECAUTIONS = {
    "Acne": "• Wash twice daily\n• Avoid oily creams",
    "Eczema": "• Moisturize frequently\n• Use mild soap",
    "Psoriasis": "• Keep skin hydrated\n• Avoid stress"
}
_KEYWORD_MAP = {
    r"\bacne\b": "Acne",
    r"\beczema\b": "Eczema",
    r"\bpsoriasis\b": "Psoriasis",
}

def _from_text(text):
    if not text: return None
    t = text.lower()
    for pat, lbl in _KEYWORD_MAP.items():
        if re.search(pat, t):
            return lbl
    return None

def _from_image(path):
    if not path or not os.path.exists(path):
        return None
    im = Image.open(path).resize((32,32)).convert("RGB")
    r, g, b = [sum(ch.getdata()) for ch in im.split()]
    # arbitrary mapping
    return "Eczema" if r>g and r>b else "Psoriasis" if g>b else "Acne"

def predict(text=None, image_path=None):
    label = _from_text(text) or _from_image(image_path) or random.choice(LABELS)
    return label, PRECAUTIONS[label]
