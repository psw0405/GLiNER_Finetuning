# GLiNER Finetuning Pipeline

A minimal, CPU-friendly pipeline for finetuning [`urchade/gliner_small-v2`](https://huggingface.co/urchade/gliner_small-v2) on a custom named-entity recognition dataset with **26 fixed labels**.

---

## Table of Contents

1. [Setup](#setup)
2. [Dataset Format](#dataset-format)
3. [Label List & Policy Notes](#label-list--policy-notes)
4. [Running the Pipeline](#running-the-pipeline)
   - [Data Validation](#data-validation)
   - [Training](#training)
   - [Evaluation](#evaluation)
   - [Prediction / Inference](#prediction--inference)
5. [Saving & Reloading the Finetuned Model](#saving--reloading-the-finetuned-model)
6. [Configuration](#configuration)
7. [Project Layout](#project-layout)

---

## Setup

> **CPU-only** – no GPU is required or expected.

```bash
# 1. Clone the repo
git clone https://github.com/psw0405/GLiNER_Finetuning.git
cd GLiNER_Finetuning

# 2. (Optional) create & activate a virtual environment
python3 -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt
```

---

## Dataset Format

Training and validation data are stored as **JSONL** files – one JSON object per line.

```jsonl
{"text": "Alice met Bob in Paris on Monday.", "entities": [{"start": 0, "end": 5, "label": "Person"}, {"start": 17, "end": 22, "label": "LocationCity"}, {"start": 26, "end": 32, "label": "Date"}]}
{"text": "The Eiffel Tower is a famous landmark.", "entities": [{"start": 4, "end": 16, "label": "CultureSite"}]}
```

**Field specifications**

| Field | Type | Description |
|-------|------|-------------|
| `text` | `str` | The raw text of the document. |
| `entities` | `list` | Zero or more entity annotations. |
| `entities[].start` | `int` | 0-based character index of the entity start (inclusive). |
| `entities[].end` | `int` | 0-based character index of the entity end (**exclusive**). `text[start:end]` must be the entity string and must be non-empty. |
| `entities[].label` | `str` | Must be one of the 26 valid labels (see below). |

**Expected file paths** (configurable via CLI):

| Purpose | Default path |
|---------|-------------|
| Training data | `data/train.jsonl` |
| Validation data | `data/valid.jsonl` |

---

## Label List & Policy Notes

The model is trained on exactly **26 labels**.  Spelling is preserved as-is (including intentional quirks noted below).

```
Person                 – individual people
Location               – general location
Organization           – companies, institutions, groups
Date                   – calendar dates
Time                   – time-of-day expressions
Animal                 – animals / wildlife
Quantity               – general numeric quantities
Event                  – general events
LocationCountry        – country names
LocationCity           – city names
Shop                   – retail stores / shops
CultureSite            – museums, galleries, heritage sites
Building               – buildings / structures
DateDuraion            – non-time durations (days, weeks, months, years) ⚠️
TimeDuration           – time durations (hours, minutes, seconds)
Sports                 – sport names / activities
Food                   – food / beverages
CivilizationCurrency   – currencies
CivilizationLaw        – legal acts, laws
QuantityAge            – age values
QunatityTemperature    – temperature values ⚠️
QuantityPrice          – price / monetary amounts
EventSports            – sports events / matches
EventFestival          – festivals / cultural events
TermMedical            – medical terminology
TermSports             – sports terminology
```

### Labeling policy

* **`DateDuraion`** *(note: intentional spelling)* – Use for **non-time durations** expressed as days, weeks, months, or years (e.g. *"three weeks"*, *"five years"*).  There is no generic `Duration` label.
* **`TimeDuration`** – Use for durations in hours, minutes, or seconds.
* **`QunatityTemperature`** *(note: intentional spelling)* – Use for **temperature values** (e.g. *"37°C"*, *"98.6°F"*).  All other standalone numeric quantities should be `Quantity` unless `QuantityAge` or `QuantityPrice` applies.

---

## Running the Pipeline

### Data Validation

Validate your JSONL files before training:

```bash
python scripts/check_data.py --train data/train.jsonl --valid data/valid.jsonl
```

The script checks for:
- Required keys (`text`, `entities`, `start`, `end`, `label`)
- Valid label membership
- Valid character index bounds (`0 ≤ start < end ≤ len(text)`)
- Non-empty, non-whitespace span text
- Overlapping spans (warning only)

It also prints dataset statistics (record count, span count, per-label frequencies).

### Training

```bash
python -m src.train \
    --train data/train.jsonl \
    --valid data/valid.jsonl \
    --output_dir outputs/run1
```

**All CLI arguments**

| Argument | Default | Description |
|----------|---------|-------------|
| `--train` | `data/train.jsonl` | Training JSONL path |
| `--valid` | `data/valid.jsonl` | Validation JSONL path |
| `--output_dir` | `outputs/run1` | Directory for checkpoints & final model |
| `--base_model` | `urchade/gliner_small-v2` | HF model name or local path |
| `--epochs` | `3` | Number of training epochs |
| `--lr` | `5e-6` | Learning rate |
| `--batch_size` | `4` | Per-device batch size (keep small for CPU) |
| `--max_length` | `384` | Maximum sequence length |
| `--seed` | `42` | Random seed for reproducibility |

Training saves:
- `outputs/run1/checkpoint-epoch{N}/` – per-epoch checkpoints
- `outputs/run1/final/` – final model weights + config + `run_config.json`

### Evaluation

```bash
python -m src.eval \
    --model_dir outputs/run1/final \
    --valid data/valid.jsonl
```

Prints precision, recall and F1 overall and per label.

| Argument | Default | Description |
|----------|---------|-------------|
| `--model_dir` | *(required)* | Path to saved model directory |
| `--valid` | `data/valid.jsonl` | Validation JSONL path |
| `--threshold` | `0.5` | Entity confidence threshold |
| `--labels` | *(all 26)* | Space-separated label override |

### Prediction / Inference

**Single text**

```bash
python -m src.predict \
    --model_dir outputs/run1/final \
    --text "Alice met Bob in Paris on Monday."
```

Output (stdout, JSONL):

```json
{"text": "Alice met Bob in Paris on Monday.", "entities": [{"start": 0, "end": 5, "label": "Person", "text": "Alice", "score": 0.92}, ...]}
```

**Batch (JSONL file)**

```bash
python -m src.predict \
    --model_dir outputs/run1/final \
    --input_file data/valid.jsonl \
    --output_file predictions.jsonl
```

| Argument | Default | Description |
|----------|---------|-------------|
| `--model_dir` | *(required)* | Path to saved model directory |
| `--text` | – | Single input text |
| `--input_file` | – | Input JSONL with `text` key |
| `--output_file` | stdout | Output JSONL for predictions |
| `--threshold` | `0.5` | Confidence threshold |
| `--labels` | *(all 26)* | Space-separated label override |

---

## Saving & Reloading the Finetuned Model

Training automatically saves the final model to `outputs/run1/final/`.

**Load for inference in Python:**

```python
from gliner import GLiNER

model = GLiNER.from_pretrained("outputs/run1/final")
model.eval()

entities = model.predict_entities(
    "Alice met Bob in Paris on Monday.",
    labels=["Person", "LocationCity", "Date"],
    threshold=0.5,
)
print(entities)
# [{'start': 0, 'end': 5, 'label': 'Person', 'text': 'Alice', 'score': 0.92}, ...]
```

**Publish to Hugging Face Hub (optional):**

```python
model.push_to_hub("your-username/gliner-finetuned")
```

---

## Configuration

Key hyperparameters are documented in `configs/config.yaml`:

```yaml
base_model: "urchade/gliner_small-v2"
train: "data/train.jsonl"
valid: "data/valid.jsonl"
output_dir: "outputs/run1"
epochs: 3
lr: 5.0e-6
batch_size: 4
max_length: 384
seed: 42
```

All values can be overridden at the command line.

---

## Project Layout

```
.
├── configs/
│   └── config.yaml          # Default hyperparameters
├── data/
│   ├── train.jsonl          # Training data (provide your own)
│   └── valid.jsonl          # Validation data (provide your own)
├── docs/
│   ├── data_contract.md
│   └── model_analysis.md
├── outputs/                 # Generated by training
│   └── run1/
│       ├── checkpoint-epoch1/
│       └── final/           # Final model for inference
├── scripts/
│   ├── check_data.py        # Data validation convenience wrapper
│   └── ...                  # Legacy data-generation scripts
├── src/
│   ├── __init__.py
│   ├── labels.py            # LABELS constant (26 labels)
│   ├── dataio.py            # JSONL loader + validation + stats
│   ├── train.py             # Training entrypoint
│   ├── eval.py              # Evaluation (P/R/F1)
│   └── predict.py           # Inference script
├── Makefile
├── README.md
└── requirements.txt
```
