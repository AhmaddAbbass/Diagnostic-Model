#!/usr/bin/env python
# make_unstructured_llm_only_budget.py
#
#  pip install "openai>=1.0" pandas openpyxl
#  set  OPENAI_API_KEY=sk-...   (CMD or PowerShell)
#  python make_unstructured_llm_only_budget.py
# ----------------------------------------------------------

import os, re, random, time, json, sys
from pathlib import Path
import pandas as pd
import openai
from openai import OpenAI

# ------------ paths ---------------------------------------------------
ROOT      = Path(".")
SCIN_PATH = ROOT / "scin_master.xlsx"
TXT_ROOT  = ROOT / "datasets_texts"
OUT_XLSX  = ROOT / "scin_master_unstructured.xlsx"
PROG_XLSX = ROOT / "progress.xlsx"
CHECKFILE = ROOT / "last_done.txt"

# ------------ budget --------------------------------------------------
BUDGET_USD      = 5.00       # your $5 credit
SAFETY_MARGIN   = 0       # stop at $4.75
USD_PP_TOKEN    = 0.0005/1000
USD_PC_TOKEN    = 0.0015/1000
spent_usd       = 0.0

# ------------ runtime knobs -------------------------------------------
MODEL_NAME     = "gpt-3.5-turbo"
N_EXAMPLES     = 1
MAX_TOKENS_OUT = 125
TEMPERATURE    = 0.6
SLEEP_SECONDS  = 0.4       # ~150 req/min ≈52k TPM
SAVE_EVERY     = 50

# ------------ resume controls -----------------------------------------
START_ROW = int(os.getenv("START_ROW", 0))
if len(sys.argv) > 1:
    START_ROW = int(sys.argv[1])
if CHECKFILE.exists() and START_ROW == 0:
    START_ROW = int(CHECKFILE.read_text().strip())
    print(f"[Resume] starting at row {START_ROW}")

# ------------ OpenAI client & key -------------------------------------
api_key =  "your_key"
if not api_key:
    raise SystemExit("❌  Set OPENAI_API_KEY in your environment first.")
client = OpenAI(api_key=api_key)

# ------------ label utils ---------------------------------------------
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
        "allergic contact dermatitis", "phytophotodermatitis"
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

def canonical(raw: str) -> str:
    s = str(raw).lower()
    s = re.sub(r"[\[\(].*?[\]\)]", " ", s)
    s = re.sub(r"[^a-z0-9\s]", " ", s)
    return re.sub(r"\s+", " ", s).strip()

def map_to_category(canon: str) -> str:
    for cat, kws in CATEGORY_KEYWORDS.items():
        if any(kw in canon for kw in kws):
            return cat
    return canon

# ------------ load unstructured examples -----------------------------
def load_free_texts(root: Path) -> pd.DataFrame:
    recs = []
    for folder in root.iterdir():
        cat = FOLDER_TO_CAT.get(folder.name.lower())
        if not cat:
            continue
        csvs = list(folder.glob("*.csv"))
        if not csvs:
            continue
        df = pd.read_csv(csvs[0], sep=None, engine="python")
        df = df.loc[:, ~df.columns.str.contains("^unnamed", case=False)]
        txt_col = next((c for c in df.columns if "text" in c.lower()), None)
        if not txt_col:
            continue
        tmp = df[[txt_col]].dropna().copy()
        tmp.columns = ["text"]
        tmp["cat"] = cat
        recs.append(tmp)
    return pd.concat(recs, ignore_index=True) if recs else pd.DataFrame(columns=["text","cat"])

txt_df = load_free_texts(TXT_ROOT)

def sample_example(cat: str) -> str:
    pool = txt_df[txt_df["cat"] == cat]["text"].tolist()
    if not pool:
        pool = txt_df["text"].tolist()
    random.shuffle(pool)
    return pool[0][:120]

def make_messages(facts, label, example):
    return [
        {
            "role": "system",
            "content": (
                "You are a patient describing your skin condition. "
                "Given the structured medical facts below, write ONE descriptive paragraph "
                "in a natural, patient-like voice. Mention all facts, but do NOT invent new ones. "
                "You can be casual or expressive, but accurate. The goal is to sound like a real patient "
                "reporting symptoms to a dermatologist."
            ),
        },
        {
            "role": "user",
            "content": (
                f"Clinical facts (structured key-value pairs):\n{facts.strip()}\n\n"
                f"Here’s an example patient description for {label}:\n"
                f"- {example}"
            ),
        },
    ]

# ------------ chat + cost tracking ------------------------------------
def chat_and_cost(messages):
    global spent_usd
    resp = client.chat.completions.create(
        model=MODEL_NAME,
        messages=messages,
        temperature=TEMPERATURE,
        max_tokens=MAX_TOKENS_OUT,
    )
    u = resp.usage
    spent_usd += u.prompt_tokens  * USD_PP_TOKEN
    spent_usd += u.completion_tokens * USD_PC_TOKEN
    return resp.choices[0].message.content.strip()

# ------------ main pipeline -------------------------------------------
def main():
    df = pd.read_excel(SCIN_PATH)
    df.columns = df.columns.str.strip().str.lower()
    df = df[df["label"].str.lower()!="unknown"].dropna(subset=["label"])
    df["canon"] = df["label"].apply(canonical)
    df["cat"]   = df["canon"].apply(map_to_category)

    facts_col = "text_data" if "text_data" in df.columns else "_struct_blob"
    if facts_col not in df.columns:
        df[facts_col] = df.drop(columns=["case_id","image","label","canon","cat"], errors="ignore")\
                           .apply(lambda r: json.dumps(r.dropna().to_dict()), axis=1)
    df["text_unstructured"] = ""

    try:
        for idx, row in df.iloc[START_ROW:].iterrows():
            if spent_usd >= BUDGET_USD - SAFETY_MARGIN:
                print(f"💸 Budget nearly reached ({spent_usd:.2f}$). Stopping.")
                break

            example = sample_example(row["cat"])
            df.at[idx,"text_unstructured"] = chat_and_cost(
                make_messages(row[facts_col], row["label"], example)
            )

            if (idx+1)%SAVE_EVERY==0:
                df.iloc[:idx+1][["case_id","image","text_unstructured","label","cat"]] \
                  .to_excel(PROG_XLSX, index=False)
                CHECKFILE.write_text(str(idx+1))
                print(f"[Autosave] row {idx+1} • spent {spent_usd:.2f}$")

            time.sleep(SLEEP_SECONDS)

    finally:
        done = df[df["text_unstructured"].astype(bool)]
        done[["case_id","image","text_unstructured","label","cat"]] \
            .to_excel(OUT_XLSX, index=False)
        print(f"✅ Saved {len(done)} rows → {OUT_XLSX} • total spent {spent_usd:.2f}$")
        if len(done)<len(df):
            CHECKFILE.write_text(str(len(done)))
            print(f"🔖 Next start row: {len(done)}")
        else:
            CHECKFILE.unlink(missing_ok=True)
            PROG_XLSX.unlink(missing_ok=True)
            print("🎉 All rows processed!")

if __name__=="__main__":
    main()