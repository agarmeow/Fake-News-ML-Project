import os
import sys
from pathlib import Path
from dotenv import load_dotenv
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

load_dotenv()

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "src"))

from reasoning.text_prep import prep_text
from reasoning.faketype_llm import classify

T_STRONG = 0.85
T2 = 0.7

_TRAINED_MODEL = None
_TOKENIZER = None
_DEVICE = None

ID2LABEL_TRAINED = {0: "ordinary", 1: "satire", 2: "clickbait"}  # 0 was 'plain' -> ordinary


def _get_trained_model():
    global _TRAINED_MODEL, _TOKENIZER, _DEVICE
    if _TRAINED_MODEL is None:
        model_dir_env = os.getenv("FAKETYPE_MODEL_DIR", "models/faketype_en_continued")
        model_path = Path(model_dir_env)
        if not model_path.is_absolute():
            model_path = PROJECT_ROOT / model_path

        _DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        _TOKENIZER = AutoTokenizer.from_pretrained(str(model_path))
        _TRAINED_MODEL = AutoModelForSequenceClassification.from_pretrained(str(model_path)).to(_DEVICE)
        _TRAINED_MODEL.eval()
    return _TRAINED_MODEL, _TOKENIZER, _DEVICE


def _predict_trained(text: str):
    model, tokenizer, device = _get_trained_model()
    prepped = prep_text(text)
    inputs = tokenizer(
        prepped,
        return_tensors="pt",
        truncation=True,
        max_length=128,
        padding=True,
    ).to(device)

    with torch.no_grad():
        outputs = model(**inputs)
        probs = torch.softmax(outputs.logits, dim=-1)[0]
        conf, pred_id = torch.max(probs, dim=0)
        label = ID2LABEL_TRAINED[pred_id.item()]
        return label, float(conf.item())


def classify_type(text: str, verifier_fn=None) -> dict:
    """
    classify_type(text, verifier_fn=None) ->
    {"label", "source", "trained_label", "trained_conf", "llm_label", "llm_conf", "verdict", "uncertain"}
    """
    # 1. Trained model on prep_text(text)
    trained_label, trained_conf = _predict_trained(text)

    # 2. LLM on RAW text
    llm_res = classify(text)
    llm_label = llm_res.get("label", "ordinary")
    llm_conf = float(llm_res.get("confidence", 0.0))

    final_label = "ordinary"
    source = "default"
    verdict = None
    uncertain = False

    # Logic:
    # a) trained in {satire, clickbait} and LLM label == trained -> final = that label, source "both"
    if trained_label in {"satire", "clickbait"} and llm_label == trained_label:
        final_label = trained_label
        source = "both"
        uncertain = False

    # b) else if trained in {satire, clickbait} with conf >= T_STRONG, or LLM in {satire, clickbait} with conf >= T_STRONG
    elif (trained_label in {"satire", "clickbait"} and trained_conf >= T_STRONG) or (
        llm_label in {"satire", "clickbait"} and llm_conf >= T_STRONG
    ):
        trained_qualifies = trained_label in {"satire", "clickbait"} and trained_conf >= T_STRONG
        llm_qualifies = llm_label in {"satire", "clickbait"} and llm_conf >= T_STRONG

        if trained_qualifies and llm_qualifies:
            # If both qualify with different labels, the higher confidence
            if trained_conf >= llm_conf:
                final_label = trained_label
                source = "trained"
            else:
                final_label = llm_label
                source = "llm"
        elif trained_qualifies:
            final_label = trained_label
            source = "trained"
        else:
            final_label = llm_label
            source = "llm"
        uncertain = True

    # c) else call verifier_fn(text) LAZILY (only here)
    else:
        if callable(verifier_fn):
            v_res = verifier_fn(text)
            if isinstance(v_res, dict):
                raw_v = v_res.get("verdict")
            else:
                raw_v = v_res

            if raw_v is True or (isinstance(raw_v, str) and raw_v.upper() == "TRUE"):
                verdict = "TRUE"
            elif raw_v is False or (isinstance(raw_v, str) and raw_v.upper() == "FALSE"):
                verdict = "FALSE"
            elif isinstance(raw_v, str) and raw_v.upper() == "UNVERIFIABLE":
                verdict = "UNVERIFIABLE"
            else:
                verdict = None
        else:
            verdict = None

        if verdict == "FALSE":
            final_label = "misleading"
            source = "verifier"
            uncertain = False
        elif verdict == "TRUE":
            final_label = "ordinary"
            source = "verifier"
            uncertain = False
        else:  # UNVERIFIABLE or None
            if llm_label == "misleading" and llm_conf >= T2:
                final_label = "misleading"
                source = "llm"
                uncertain = True
            else:
                final_label = "ordinary"
                source = "default"
                uncertain = False

    return {
        "label": final_label,
        "source": source,
        "trained_label": trained_label,
        "trained_conf": trained_conf,
        "llm_label": llm_label,
        "llm_conf": llm_conf,
        "verdict": verdict,
        "uncertain": uncertain,
    }
