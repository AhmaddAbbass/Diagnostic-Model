# Derm-MMMModal: A Fully-Documented Multimodal Pipeline for Skin-Disease Classification

> *An end-to-end system—trained entirely from scratch—that classifies 14+ dermatology conditions from paired or unpaired images and textual descriptions.*

---

## 📘 Overview

Building a robust vision-language dermatology classifier proved exceptionally challenging:

1. **Data scarcity**  
   Finding high-quality image–text pairs in dermatology is nearly impossible. Public datasets either lack paired descriptions or cover only a handful of conditions.  

2. **Complex cleaning pipeline**  
   We devised a **five-step data-cleaning process** (scripts in this repo) to filter, normalize, de-duplicate, validate, and restructure raw assets. The final cleaned dataset is available [here](<https://drive.google.com/drive/folders/1rj9GXAptMMJ3a9JVuqOkmOHq7wX4htC3?usp=sharing>).

3. **Scratch training**  
   **No pretrained checkpoints** were used. Both the image and text encoders were initialized randomly and trained from the ground up.

---

## 🗃️ Datasets

We organize our data into **three core modalities**:

- **Image-only** (`/data/image_only/`)  
  ~12 000 clinical skin images, organized by split (`train`/`val`/`test`) and disease label.

- **Image-Text (img_text)** (`/data/img_text/`)  
  ~3 000 paired samples.  
  Original text was highly structured (spreadsheet-style); we applied `make_unstructured.py` to convert it into natural descriptions in ahmad_unstructured branch.

- **Text-only** (`/data/text_only/`)  
  A scarce collection of free-form clinical notes (~500 samples) that guided our unstructuring logic and served as seeds for augmentation.

### Synthetic Augmentation

To overcome the text shortage:

1. **Clustering**  
   For each disease, cluster existing text from `img_text` + `text_only`.

2. **LLM prompting**  
   Provide cluster centroids (symptoms + writing style) to a large language model, requesting new, realistic clinical descriptions.

3. **Validation**  
   Automatically filter for length, medical terminology compliance, and label consistency.

Synthetic samples now live in `/data/synthetic/`.

---

## 🚀 Project Structure

```
├── fusion_head/            # Gated Cross-Attention fusion implementation
│   ├── dataset.py
│   ├── model.py
│   ├── train.py
│   ├── test.py
│   ├── main.py
│   └── requirements.txt
│
├── image_encoder/          # Scratch-trained EfficientNet-B0 + SimCLR losses
│   ├── requirements.txt
│   ├── utils.py
│   ├── summary.py
│   ├── dataset.py
│   ├── model.py
│   ├── train_simclr.py
│   ├── train_probe.py
│   └── evaluate.py
│
├── text_encoder/           # Scratch-trained Transformer text encoder
│   ├── model.py
│   ├── training.py
│   ├── test.py
│   └── Text_Encoder_README.txt
│
├── application/            # Deployed Flask + static frontend
│   ├── backend/
│   │   ├── app.py
│   │   ├── inference.py
│   │   ├── requirements.txt
│   │   └── models/
│   │       ├── effnet_simclr.pth
│   │       ├── fusion_bundle.pt
│   │       └── text_ckpt/      # ClinicalBERT weights + tokenizer files
│   │
│   └── frontend/
│       ├── index.html
│       ├── style.css
│       └── script.js
│
└── README.md               # ← you’re here!
```

---

## 🎓 Training

### Image Encoder

- **Architecture**: EfficientNet-B0  
- **Objective**: SimCLR contrastive loss from random init  
- **Challenge**: Severe class imbalance (some conditions had 10× more images)  
- **Solution**:  
  - Compute per-class weights via inverse-frequency  
  - Use `torch.utils.data.WeightedRandomSampler` for balanced batches  

**Result**: Stable convergence in 30 epochs on Colab Pro GPU  
[▶️ Colab Notebook: Train Image Encoder](<https://colab.research.google.com/drive/1HaOTOd2-AUBsBcR0aKt0yVqZQIahK7dq?usp=sharing>)

---

### Text Encoder

- **Architecture**: 6-layer Transformer from scratch  
- **Objective**: Categorical cross-entropy on unstructured + synthetic text  
- **Observation**: Very high validation accuracy (> 90%)—likely overfit to synthetic patterns  

**Result**: Fast training in 15 epochs  
[▶️ Colab Notebook: Train Text Encoder](<https://colab.research.google.com/drive/1GiuA1kWxFO24JizPNPhjUhS9NKCtKTK-?usp=sharing>)

---

### Fusion Head

- **Model**: Gated Cross-Attention (GCA) combines image & text embeddings  
- **Training set**: Only true pairs from `img_text/`  
- **Initial accuracy**: ~ 33 % overall  
- **Improvement 1**: Re-train image encoder → fusion jumps to ~ 40 %  
- **Improvement 2**: Pair additional unpaired samples via ECDF methods (image_only ↔ text_only) → end result ~ 83 %  

[▶️ Colab Notebook: Train Fusion Head](<https://colab.research.google.com/drive/1T5JEcZo-JtXMUeKX5yxyRYm7SU9vD08s?usp=sharing>)

---

## 📊 Results

| Stage               | Dataset                | Accuracy |
| ------------------- | ---------------------- | -------- |
| **Image-only probe**| image_only             | 71.3 % (val) / 69.8 % (test) |
| **Text-only probe** | text_only + synthetic  | 92.4 % (val) / 93.1 % (test) |
| **Fusion (GCA)**    | img_text + paired ECDF | **83 %** overall |

> *All models trained from scratch, no external checkpoints used.*

---

## 📂 Report & Notebooks

- **Full Project Report (PDF)**: [Download here](<https://drive.google.com/uc?export=download&id=1l_dZ5z-d889mubzot3bb3gRelR-Z93Ck>)  
- **Image Encoder Notebook**: [Open in Colab](<https://colab.research.google.com/drive/1HaOTOd2-AUBsBcR0aKt0yVqZQIahK7dq?usp=sharing>)  
- **Text Encoder Notebook**: [Open in Colab](<https://colab.research.google.com/drive/1GiuA1kWxFO24JizPNPhjUhS9NKCtKTK-?usp=sharing>)  
- **Fusion Head Notebook**: [Open in Colab](<https://colab.research.google.com/drive/1T5JEcZo-JtXMUeKX5yxyRYm7SU9vD08s?usp=sharing>)
- **Data Preparation Notebooks**:
  -   Databalancer using K-means [Open in Colab](<https://colab.research.google.com/drive/1ftwkx6G5C_PKZUxX60mh_Xg15CF9Nznw?authuser=1>) ,
  -   Synthetic data generator  [Open in Colab](<https://colab.research.google.com/drive/1Ls1F_6LXzSmSIxf5DspKMksRabS_bpEU?usp=sharing>) ,
  -   datacleaner [Open in Colab](<https://colab.research.google.com/drive/1OrnjgdNaMpeso44aDfWCZ2D7z9qLwy80?usp=sharing>)
  -   datapreprocessing [Open in Colab](<https://colab.research.google.com/drive/1q7Y6s6BJjMFmayGe6tX_xbBJ3mKigz4I?usp=sharing>)
  -   dataloading  [Open in Colab](<https://colab.research.google.com/drive/1PoGXSypnftOo9JD6MRIEPiVVzAVMoOVG?usp=sharing>)
                                

---

## 🌱 Future Work

- **Derm1M integration**  
  - We eagerly await the release of **Derm1M** (“Derm1M: A Million-Scale Vision-Language Dataset…”, arXiv:2503.14911) to expand to > 1 M paired samples.  
  - Link: https://arxiv.org/abs/2503.14911  

- **Hard negative mining** in contrastive loss  
- **Prompt-tuned LLM** for richer synthetic text  
- **Ensemble stacking** of per-modality logits

### 🚀 Quickstart Guide (From Repo to Running the App)

1. **Clone the repository**  
   ```bash
   git clone https://github.com/AhmaddAbbass/Diagnostic-Model.git
   cd Diagnostic-Model/application
   ```

2. **Build the Docker image**  
   ```bash
   docker build -t derm-mmmodal:latest .
   ```

3. **Run the Docker container**  
   ```bash
   docker run --rm -d \
     --name derm-app \
     -p 8000:8000 \
     derm-mmmodal:latest
   ```

4. **Open the app in your browser**  
   Navigate to:  
   ```
   http://localhost:8000
   ```

5. **Stop & clean up**  
   ```bash
   docker stop derm-app
   docker rm derm-app
   ```


---

Thank you for exploring Derm-MMModal! Feel free to open issues or submit pull requests for enhancements.

