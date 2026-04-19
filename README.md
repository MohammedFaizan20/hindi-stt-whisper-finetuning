# Hindi Speech-to-Text Fine-tuning
### Fine-tuning OpenAI Whisper Small on Hindi Audio using Google FLEURS

[![Python](https://img.shields.io/badge/Python-3.12-blue)](https://python.org)
[![HuggingFace](https://img.shields.io/badge/HuggingFace-Transformers_4.40-yellow)](https://huggingface.co)
[![Kaggle](https://img.shields.io/badge/Kaggle-GPU_T4_x2-20BEFF)](https://kaggle.com)
[![WER](https://img.shields.io/badge/WER-27.99%25-brightgreen)](https://github.com)

---

## Overview

This project fine-tunes OpenAI's `whisper-small` model on Hindi speech data to significantly improve transcription accuracy over the base model. The base Whisper Small model, while multilingual, performs poorly on Hindi out of the box. Fine-tuning on a curated Hindi dataset brings Word Error Rate down from **86.46% to 27.99%** — a 58% improvement.

The entire pipeline is built on HuggingFace Transformers and trained on Kaggle's free GPU tier, making it fully reproducible without any paid compute.

---

## Results

| Model | WER | CER |
|---|---|---|
| whisper-small (base) | 86.46% | 52.87% |
| whisper-small (fine-tuned) | 27.99% | 10.23% |
| **Improvement** | **↓ 58.47%** | **↓ 42.64%** |

Evaluation performed exclusively on a held-out test set of 200 samples never seen during training.

### Training Progression

| Step | Training Loss | Validation Loss | Validation WER |
|---|---|---|---|
| 200 | 0.0531 | 0.3062 | 29.19% |
| 400 | 0.0112 | 0.3683 | 30.57% |
| 600 | 0.0021 | 0.4325 | 29.59% |
| 800 | 0.0004 | 0.5047 | **28.80% ← best** |
| 1000 | 0.0001 | 0.5665 | 28.97% |

---

## Dataset

**[Google FLEURS Hindi](https://huggingface.co/datasets/google/fleurs)** (`hi_in`)

| Split | Samples |
|---|---|
| Train | 2,120 |
| Validation | 239 |
| Test | 418 |

FLEURS (Few-shot Learning Evaluation of Universal Representations of Speech) is a Google-released multilingual benchmark dataset with clean Devanagari transcriptions at 16kHz — exactly what Whisper expects, no resampling required.

---

## Model

**[openai/whisper-small](https://huggingface.co/openai/whisper-small)** — 244M parameters

Whisper Small was chosen as the practical sweet spot:
- Strong multilingual pretraining including Hindi exposure
- Fits comfortably in free GPU VRAM (15GB T4)
- Well-documented fine-tuning pipeline on HuggingFace
- Meaningful improvement headroom over the base model

---

## Training Setup

| Hyperparameter | Value | Reasoning |
|---|---|---|
| Learning rate | 1e-5 | Prevents catastrophic forgetting of pretrained weights |
| Batch size | 16 | Maximum that fits in T4 VRAM |
| Max steps | 1000 | ~15 epochs over 2120 training samples |
| Warmup steps | 100 | Gradual LR warmup for stable early training |
| fp16 | True | Halves VRAM usage with no accuracy loss |
| Gradient checkpointing | True | Required for stable training on T4 |
| Evaluation strategy | Every 200 steps | Catch overfitting early |
| Best model metric | Validation WER | Directly optimizes evaluation metric |

**Hardware:** Kaggle free tier — 2x Tesla T4 (15GB VRAM each)  
**Training time:** ~6.4 hours  
**Best checkpoint:** Step 800  

---

## Project Structure

```
hindi-stt-whisper-finetuning/
│
├── hindi-stt-finetuning.ipynb   # Complete notebook — setup to evaluation
└── README.md
```

---

## How to Run

### 1. Clone the repo
```bash
git clone https://github.com/yourusername/hindi-stt-whisper-finetuning
cd hindi-stt-whisper-finetuning
```

### 2. Install dependencies
```bash
pip install transformers==4.40.0 \
            datasets==2.16.0 \
            accelerate==0.33.0 \
            evaluate==0.4.1 \
            jiwer==3.0.3 \
            librosa==0.10.1 \
            soundfile==0.12.1 \
            numpy==1.26.4 \
            peft==0.10.0
```

### 3. Run on Kaggle (recommended)
- Upload the notebook to [Kaggle](https://kaggle.com)
- Enable GPU T4 x2 accelerator
- Enable internet access
- Run all cells in order

### 4. HuggingFace login (required for dataset)
```python
from huggingface_hub import login
login(token="your_hf_token")
```

---

## Inference

```python
from transformers import WhisperForConditionalGeneration, WhisperProcessor
import torch

model = WhisperForConditionalGeneration.from_pretrained("path/to/fine-tuned-model")
processor = WhisperProcessor.from_pretrained("path/to/fine-tuned-model")

device = "cuda" if torch.cuda.is_available() else "cpu"
model = model.to(device)
model.eval()

def transcribe(audio_array, sampling_rate):
    input_features = processor(
        audio_array,
        sampling_rate=sampling_rate,
        return_tensors="pt"
    ).input_features.to(device)

    with torch.no_grad():
        predicted_ids = model.generate(
            input_features,
            language="hi",
            task="transcribe"
        )
    return processor.batch_decode(predicted_ids, skip_special_tokens=True)[0]
```

---

## Error Analysis

After fine-tuning, sentence structure is largely correct. Remaining errors fall into predictable patterns:

1. **Rare vocabulary** — uncommon words still occasionally wrong
2. **Similar sounding characters** — ष/श/स confusion consistent with Hindi phonetics
3. **Long sentence degradation** — accuracy drops slightly toward end of longer sentences
4. **Technical/foreign words** — loanwords and proper nouns still challenging
5. **Diacritic errors** — ा/े/ी confusion on fast speech

---

## What I Would Do With More Data and Compute

- **Larger dataset** — Shrutilipi or IndicSUPERB Hindi (100+ hours) to push WER below 15%
- **Whisper Medium** — significantly better results with A100 GPU
- **LoRA/PEFT** — parameter-efficient fine-tuning to reduce training time and memory
- **Data augmentation** — speed perturbation, background noise for real-world robustness
- **Hinglish support** — mixed Hindi-English dataset to handle code-switching
- **Beam search tuning** — beam size and length penalty optimization for longer sentences

---

## Tech Stack

| Component | Tool |
|---|---|
| Model | openai/whisper-small |
| Dataset | google/fleurs (hi_in) |
| Training framework | HuggingFace Transformers 4.40 |
| Trainer | Seq2SeqTrainer |
| Infrastructure | Kaggle — 2x Tesla T4 (free tier) |
| Evaluation | WER, CER — HuggingFace evaluate |
| Audio processing | librosa, soundfile |
| Language | Python 3.12 |

---

## License

MIT
