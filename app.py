from dotenv import load_dotenv
load_dotenv()
import sys
import os

sys.path.append(
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "src")
)

import streamlit as st
from PIL import Image
import numpy as np
import torch
import easyocr

from fact_check.fact_checker import fact_check

# ASSUMPTION -- adjust this import if wrong: multimodal_predict.py sits
# at the project root (same folder as this app.py) and exposes
# predict(text, image) -> (label, confidence)
from multimodal_predict import predict
from reasoning.llm_reasoner import transcribe_audio, describe_image
from reasoning.claim_normalizer import normalize_claim


st.set_page_config(page_title="Fake-News Fact-Checker", layout="centered")

st.title("Multilingual Multimodal Fake-News Fact-Checker")
st.caption(
    "Type a claim, upload an image, or both. The multimodal classifier "
    "flags learned fake/real patterns; the evidence verifier independently "
    "checks the claim against the web. These are shown separately on purpose "
    "-- they answer different questions."
)


@st.cache_resource
def get_ocr_reader():
    return easyocr.Reader(["en"], gpu=True)


FAKETYPE_ID2LABEL = {0: "ordinary", 1: "satire", 2: "misleading", 3: "clickbait"}


@st.cache_resource
def get_faketype_muril():
    model_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models", "faketype_muril")
    if not os.path.exists(model_dir):
        return None, None
    try:
        from transformers import AutoTokenizer, AutoModelForSequenceClassification
        tokenizer = AutoTokenizer.from_pretrained(model_dir)
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model = AutoModelForSequenceClassification.from_pretrained(model_dir).to(device)
        model.eval()
        return tokenizer, model
    except Exception as e:
        print(f"Warning: could not load faketype_muril: {e}")
        return None, None


def predict_faketype_muril(text):
    tokenizer, model = get_faketype_muril()
    if tokenizer is None or model is None:
        return "unclear", 0.0
    device = next(model.parameters()).device
    inputs = tokenizer(
        str(text),
        truncation=True,
        padding=True,
        max_length=64,
        return_tensors="pt"
    ).to(device)
    with torch.inference_mode():
        outputs = model(**inputs)
        probs = torch.softmax(outputs.logits, dim=-1)[0]
        pred_idx = torch.argmax(probs).item()
        confidence = probs[pred_idx].item()
        label = FAKETYPE_ID2LABEL.get(pred_idx, "unclear")
    return label, confidence


@st.cache_resource
def preload_all_models():
    """Preload NLI, SentenceTransformer, and CLIP models once at app startup."""
    from fact_check.fact_checker import get_cached_models
    get_cached_models()
    try:
        import multimodal_predict
    except Exception as e:
        print(f"Warning: CLIP preload failed: {e}")
    return True


preload_all_models()


def run_ocr(pil_image):
    reader = get_ocr_reader()
    results = reader.readtext(np.array(pil_image))
    return " ".join(text for (_, text, _) in results)


# ------------------------------------------------------------------
# SESSION STATE
# ------------------------------------------------------------------

if "claim_box" not in st.session_state:
    st.session_state.claim_box = ""

if "last_image_id" not in st.session_state:
    st.session_state.last_image_id = None

if "last_audio_id" not in st.session_state:
    st.session_state.last_audio_id = None


# ------------------------------------------------------------------
# INPUT
# ------------------------------------------------------------------

uploaded_file = st.file_uploader(
    "Upload an image (optional)", type=["png", "jpg", "jpeg"]
)

image = None

if uploaded_file is not None:

    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded image", width=300)

    # Only run OCR once per newly-uploaded image, not on every Streamlit
    # rerun (which happens on every widget interaction).
    image_id = f"{uploaded_file.name}_{uploaded_file.size}"

    if st.session_state.last_image_id != image_id:

        try:
            with st.spinner("Reading text from image..."):
                ocr_text = run_ocr(image)

            st.caption(f"(OCR extracted {len(ocr_text)} characters)")

            # OCR found little/no text -- probably a real photo, not a
            # text screenshot. Fall back to vision-LLM description so
            # image-only submissions still get something to verify.
            if len(ocr_text.strip()) < 10:
                try:
                    with st.spinner("No readable text -- describing image content..."):
                        described = describe_image(image)
                    if described:
                        st.caption("(Little/no text found -- used image description instead)")
                        ocr_text = described
                except Exception as e:
                    st.caption(f"(Image description unavailable: {e})")

            # T1: Normalize OCR or describe_image text
            norm_res = normalize_claim(ocr_text)
            st.session_state.claim_box = norm_res["claim"]
            st.caption(f"Original: {ocr_text} | language: {norm_res['language']}")
            st.session_state.last_image_id = image_id

        except Exception as e:
            st.error(f"OCR failed: {e}")
            # Still mark this image as "handled" so we don't loop
            # retrying OCR on every rerun of a broken image.
            st.session_state.last_image_id = image_id

st.write("Or record the claim by voice:")
audio_value = st.audio_input("Record a claim")

if audio_value is not None:
    audio_bytes = audio_value.getvalue()
    audio_id = f"{len(audio_bytes)}"

    if st.session_state.last_audio_id != audio_id:
        try:
            with st.spinner("Transcribing voice..."):
                transcript = transcribe_audio(audio_bytes, filename="claim.wav")

            if transcript:
                norm_res = normalize_claim(transcript)
                st.session_state.claim_box = norm_res["claim"]
                st.caption(f"Original: {transcript} | language: {norm_res['language']}")
            else:
                st.warning(
                    "Couldn't transcribe (check GROQ_API_KEY is set on this "
                    "machine). Type the claim manually instead."
                )
            st.session_state.last_audio_id = audio_id

        except Exception as e:
            st.warning(f"Voice transcription failed: {e}")
            st.session_state.last_audio_id = audio_id

claim_text = st.text_area(
    "Claim (auto-filled from image/voice if provided -- edit as needed):",
    key="claim_box",
    height=100,
)

col_btn, col_chk = st.columns([1, 2])
with col_btn:
    analyze = st.button("Analyze")
with col_chk:
    analyze_type_anyway = st.checkbox("Analyze type anyway", value=False)


# ------------------------------------------------------------------
# PIPELINE
# ------------------------------------------------------------------

if analyze:

    if not claim_text.strip() and image is None:
        st.warning("Enter a claim or upload an image first.")
        st.stop()

    # ----------------------------------------------------------
    # Evidence-based verifier FIRST -- this is the actual answer
    # ----------------------------------------------------------

    if claim_text.strip():

        norm_typed = normalize_claim(claim_text)
        final_claim = norm_typed["claim"]
        st.info(f"Checking claim: {final_claim}")

        with st.spinner("Verifying claim against evidence..."):
            result = fact_check(final_claim)

        st.subheader("Final Verdict")

        if result:

            if result.get("conflict"):
                st.warning("Rule-based pipeline and LLM disagree - review the evidence below")

            verdict = result["verdict"]
            conf_val = result.get("confidence", 0.0)
            if conf_val < 0.30:
                strength_label = "weak evidence"
            elif conf_val <= 0.60:
                strength_label = "moderate"
            else:
                strength_label = "strong"

            if verdict == "TRUE":
                if strength_label == "weak evidence":
                    st.warning(f"**{verdict}** -- {strength_label}")
                else:
                    st.success(f"**{verdict}** -- {strength_label}")
            elif verdict == "FALSE":
                if strength_label == "weak evidence":
                    st.warning(f"**{verdict}** -- {strength_label}")
                else:
                    st.error(f"**{verdict}** -- {strength_label}")
            else:
                st.warning(f"**{verdict}**")

            st.caption(
                f"Confidence: {conf_val * 100:.1f}% ({strength_label}) | "
                f"Based on {len(result['evidence'])} evidence sources -- "
                f"support score {result['support_score']:.2f}, "
                f"contradiction score {result['contradiction_score']:.2f}"
            )

            if result.get("llm_reasoning"):
                if result.get("verdict_source") == "llm_fallback":
                    st.info(
                        f"**LLM arbitration (Groq gpt-oss)** -- rule-based "
                        f"pipeline was inconclusive, so this verdict comes "
                        f"from the LLM reading the same evidence: "
                        f"{result['llm_reasoning']}"
                    )
                else:
                    st.caption(
                        f"LLM's read on the same evidence: {result['llm_reasoning']}"
                    )

            with st.expander(f"Evidence considered ({len(result['evidence'])})"):

                for i, ev in enumerate(result["evidence"], start=1):

                    st.markdown(f"**Evidence {i} -- {ev['title']}**")
                    st.write(
                        f"NLI: {ev['nli_label']} ({ev['nli_confidence']:.2f}) | "
                        f"Consistency: {ev['consistency_verdict']}"
                    )
                    st.write(ev["evidence"][:300] + "...")
                    st.write(ev["url"])
                    st.markdown("---")

            # ----------------------------------------------------------
            # G5: Fake-type analysis block
            # ----------------------------------------------------------
            should_analyze_faketype = (
                verdict == "FALSE"
                or result.get("conflict") is True
                or analyze_type_anyway
            )

            if should_analyze_faketype:
                st.subheader("Fake-Type Analysis")
                lang = str(norm_typed.get("language", "en")).strip().lower()
                if lang != "en":
                    st.info("Fake-type analysis supports English only.")
                else:
                    with st.spinner("Analyzing fake-type category..."):
                        from reasoning.faketype_pipeline import classify_type
                        try:
                            pt_res = classify_type(final_claim, verifier_fn=lambda _: result)
                            pipe_label = pt_res["label"]
                            source = pt_res["source"]
                            trained_lbl = pt_res["trained_label"]
                            trained_conf = pt_res["trained_conf"]
                            llm_lbl = pt_res["llm_label"]
                            llm_conf = pt_res["llm_conf"]
                            uncertain = pt_res["uncertain"]

                            st.markdown(f"**Pipeline Label:** `{pipe_label}` (Decided by: `{source}`)")
                            if uncertain:
                                st.warning("uncertain")

                            col1, col2 = st.columns(2)
                            with col1:
                                st.markdown(f"**Trained Model:** `{trained_lbl}` ({trained_conf * 100:.1f}%)")
                            with col2:
                                st.markdown(f"**Groq LLM:** `{llm_lbl}` ({llm_conf * 100:.1f}%)")

                        except Exception as e:
                            st.error(f"Fake-type analysis error: {e}")

        else:
            st.write("No evidence could be retrieved for this claim.")

    else:
        st.info(
            "No claim text available to verify "
            "(image had no readable or entered text)."
        )

    # ----------------------------------------------------------
    # Multimodal / image classifier signal -- secondary, small,
    # clearly not the final answer (only shown if image present)
    # ----------------------------------------------------------

    if image is not None:

        with st.spinner("Running multimodal classifier..."):
            try:
                label, confidence = predict(claim_text, image)

                st.caption(
                    f"Additional signal -- learned text+image pattern "
                    f"classifier (trained on Fakeddit): **{label}** "
                    f"({confidence * 100:.0f}%). This reflects visual/textual "
                    f"style patterns, not fact-checking -- the Final Verdict "
                    f"above is the actual answer."
                )

            except Exception as e:
                st.caption(f"(Multimodal classifier unavailable: {e})")