import sys
import os

sys.path.append(
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "src")
)

import streamlit as st
from PIL import Image
import numpy as np
import easyocr

from fact_check.fact_checker import fact_check

# ASSUMPTION -- adjust this import if wrong: multimodal_predict.py sits
# at the project root (same folder as this app.py) and exposes
# predict(text, image) -> (label, confidence)
from multimodal_predict import predict


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

            st.session_state.claim_box = ocr_text
            st.session_state.last_image_id = image_id

        except Exception as e:
            st.error(f"OCR failed: {e}")
            # Still mark this image as "handled" so we don't loop
            # retrying OCR on every rerun of a broken image.
            st.session_state.last_image_id = image_id

claim_text = st.text_area(
    "Claim (auto-filled from image text if uploaded -- edit as needed):",
    key="claim_box",
    height=100,
)

analyze = st.button("Analyze")


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

        with st.spinner("Verifying claim against evidence..."):
            result = fact_check(claim_text)

        st.subheader("Final Verdict")

        if result:

            verdict = result["verdict"]

            if verdict == "TRUE":
                st.success(f"**{verdict}** -- {result['confidence'] * 100:.0f}% confidence")
            elif verdict == "FALSE":
                st.error(f"**{verdict}** -- {result['confidence'] * 100:.0f}% confidence")
            else:
                st.warning(f"**{verdict}**")

            st.caption(
                f"Based on {len(result['evidence'])} evidence sources -- "
                f"support score {result['support_score']:.2f}, "
                f"contradiction score {result['contradiction_score']:.2f}"
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