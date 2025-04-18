#!/usr/bin/env python
# make_unstructured_llm_only_budget.py  –  hard‑range edition (bug‑fixed)

from __future__ import annotations
import json, os, random, re, sys, time
from datetime import datetime
from pathlib import Path
import pandas as pd
from openai import OpenAI

# ---------------------------- paths ----------------------------------
ROOT       = Path('.')
SCIN_PATH  = ROOT / 'scin_master.xlsx'
TXT_ROOT   = ROOT / 'datasets_texts'
SAFE_PATH  = ROOT / 'safe_rows_644_2645.xlsx'
OUT_XLSX   = ROOT / 'scin_master_unstructured.xlsx'
PROG_XLSX  = ROOT / 'progress.xlsx'
CHECKFILE  = ROOT / 'last_done.txt'

# ---------------------------- budget ---------------------------------
BUDGET_USD     = 5.00
SAFETY_MARGIN  = 0.0
USD_PP_TOKEN   = 0.0005 / 1000
USD_PC_TOKEN   = 0.0015 / 1000
spent_usd      = 0.0

# ------------------------- model knobs -------------------------------
MODEL_NAME     = 'gpt-3.5-turbo'
MAX_TOKENS_OUT = 125
TEMPERATURE    = 0.6
SLEEP_SECONDS  = 0.4
SAVE_EVERY     = 50
SKIP_LOG_STEP  = 200               # log every Nth skipped row

# ------------------------- resume logic ------------------------------
START_ROW = 0
if CHECKFILE.exists():
    START_ROW = int(CHECKFILE.read_text())
print(f"[Boot] {datetime.now():%H:%M:%S} • START_ROW={START_ROW}")

# ------------------------- OpenAI client -----------------------------
api_key = "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"
if not api_key:
    sys.exit('❌  Please set OPENAI_API_KEY in your environment.')
client = OpenAI(api_key=api_key)

# --------------------- category helpers ------------------------------
FOLDER_TO_CAT = {
    "acne": "acne",
    "athlete's_foot_(tinea_pedis)": "athlete foot",
    "contact_dermatitis": "contact dermatitis",
    "eczema": "eczema",
    "folliculitis": "folliculitis",
    "hives_(urticaria)": "urticaria",
    "impetigo": "impetigo",
    "psoriasis": "psoriasis",
    "ringworm": "ringworm",
    "rosacea": "rosacea",
    "scabies": "scabies",
    "shingles_(herpes_zoster)": "herpes zoster",
    "vitiligo": "vitiligo",
}

CATEGORY_KEYWORDS = {
    "acne": ["acne", "acne keloidalis", "acne urticata"],
    "athlete foot": ["athlete", "tinea pedis", "athlete's foot"],
    "contact dermatitis": [
        "contact dermatitis", "photocontact dermatitis", "irritant contact dermatitis",
        "allergic contact dermatitis", "phytophotodermatitis",
    ],
    "eczema": ["eczema", "dermatitis", "dyshidrotic", "chronic dermatitis", "acute dermatitis"],
    "folliculitis": ["folliculitis"],
    "urticaria": ["urticaria", "hives"],
    "impetigo": ["impetigo", "ecthyma"],
    "psoriasis": ["psoriasis", "inverse psoriasis"],
    "ringworm": ["tinea", "ringworm"],
    "rosacea": ["rosacea"],
    "scabies": ["scabies"],
    "herpes zoster": ["herpes zoster", "shingles"],
    "vitiligo": ["vitiligo"],
}

def canonical(s: str) -> str:
    s = re.sub(r"[\[\(].*?[\]\)]", " ", str(s).lower())   # ← fixed pattern
    s = re.sub(r"[^a-z0-9\s]", " ", s)
    return re.sub(r"\s+", " ", s).strip()

def cat_from_label(canon: str) -> str:
    for cat, kws in CATEGORY_KEYWORDS.items():
        if any(kw in canon for kw in kws):
            return cat
    return canon

# ------------------ load example sentences ---------------------------
def load_examples(root: Path) -> pd.DataFrame:
    recs = []
    for fld in root.iterdir():
        cat = FOLDER_TO_CAT.get(fld.name.lower())
        if not cat:
            continue
        csvs = list(fld.glob("*.csv"))
        if not csvs:
            continue
        df = pd.read_csv(csvs[0], sep=None, engine="python")
        df = df.loc[:, ~df.columns.str.contains("^unnamed", case=False)]
        txt_col = next((c for c in df.columns if "text" in c.lower()), None)
        if not txt_col:
            continue
        rec = df[[txt_col]].dropna().rename(columns={txt_col: "text"})
        rec["cat"] = cat
        recs.append(rec)
    return pd.concat(recs, ignore_index=True) if recs else pd.DataFrame(columns=["text", "cat"])

examples_df = load_examples(TXT_ROOT)
print(f"[Boot] Loaded {len(examples_df):,} example sentences")

def sample_example(cat: str) -> str:
    pool = examples_df[examples_df.cat == cat].text.tolist() or examples_df.text.tolist()
    random.shuffle(pool)
    return pool[0][:120]

# --------------- prompt builder & cost tracker -----------------------
def prompt(facts: str, label: str, ex: str):
    return [
        {"role":"system","content":(
            "You are a patient describing your skin condition. "
            "Write ONE paragraph in a natural, patient‑like voice that mentions ALL "
            "facts listed. Do NOT add new facts.")},
        {"role":"user","content":f"Facts:\n{facts}\n\nExample for {label}:\n- {ex}"}
    ]

def chat_and_cost(msgs):
    global spent_usd
    r = client.chat.completions.create(
        model=MODEL_NAME, messages=msgs,
        temperature=TEMPERATURE, max_tokens=MAX_TOKENS_OUT
    )
    u = r.usage
    spent_usd += u.prompt_tokens*USD_PP_TOKEN + u.completion_tokens*USD_PC_TOKEN
    return r.choices[0].message.content.strip()

# ---------------------------------------------------------------------
def main():
    # -------- load / clean master sheet ------------------------------
    df = pd.read_excel(SCIN_PATH)
    df.columns = df.columns.str.strip().str.lower()
    df = df[df.label.str.lower() != "unknown"].dropna(subset=["label"])
    df.reset_index(drop=True, inplace=True)                 #  <<< KEY FIX
    df["canon"] = df.label.apply(canonical)
    df["cat"]   = df.canon.apply(cat_from_label)

    facts_col = "text_data" if "text_data" in df.columns else "_struct_blob"
    if facts_col not in df.columns:
        df[facts_col] = df.drop(columns=["case_id","image","label","canon","cat"], errors="ignore")\
                           .apply(lambda r: json.dumps(r.dropna().to_dict()), axis=1)

    df["text_unstructured"] = ""

    # -------- inject safe rows 644‑2645 ------------------------------
    if SAFE_PATH.exists():
        safe = pd.read_excel(SAFE_PATH).rename(str.lower, axis=1)
        for _, r in safe.iterrows():
            ix = int(r.orig_index)
            if ix < len(df):
                df.at[ix, "text_unstructured"] = r.text_unstructured
        print(f"[Init] Injected {len(safe)} safe rows (644‑2645)")

    # -------- exact generation ranges -------------------------------
    gen_indices = list(range(0, 644)) + list(range(2646, len(df)))
    print(f"[Init] Will generate {len(gen_indices)} rows via API")

    try:
        for i, idx in enumerate(gen_indices, start=1):
            # skip if row already has text (NaN‑safe check)
            if pd.notna(df.at[idx, "text_unstructured"]) and str(df.at[idx,"text_unstructured"]).strip():
                if idx % SKIP_LOG_STEP == 0:
                    print(f"[Skip] row {idx} already done")
                continue

            if spent_usd >= BUDGET_USD - SAFETY_MARGIN:
                print(f"💸 Reached budget cap at ${spent_usd:.2f}. Stopping.")
                break

            ex  = sample_example(df.at[idx, "cat"])
            txt = chat_and_cost(prompt(df.at[idx, facts_col], df.at[idx, "label"], ex))
            df.at[idx, "text_unstructured"] = txt
            print(f"[Gen ] row {idx} ({df.at[idx,'label']}) • spent ${spent_usd:.2f}")

            if i % SAVE_EVERY == 0:
                df[df.text_unstructured.astype(bool)][
                    ["case_id","image","text_unstructured","label","cat"]
                ].to_excel(PROG_XLSX, index=False)
                CHECKFILE.write_text(str(idx))
                print(f"[Autosave] up to row {idx} • spent ${spent_usd:.2f}")

            time.sleep(SLEEP_SECONDS)

    finally:
        done = df[df.text_unstructured.astype(bool)]
        done[["case_id","image","text_unstructured","label","cat"]].to_excel(OUT_XLSX, index=False)
        print(f"✅ Saved {len(done)} rows → {OUT_XLSX.name} • total spend ${spent_usd:.2f}")

        if len(done) < len(df):
            CHECKFILE.write_text(str(done.index.max()))
            print(f"🔖 Resume will start at row {done.index.max()}")
        else:
            for p in (CHECKFILE, PROG_XLSX):
                p.unlink(missing_ok=True)
            print("🎉 All rows processed!")

# ---------------------------------------------------------------------
if __name__ == "__main__":
    main()
