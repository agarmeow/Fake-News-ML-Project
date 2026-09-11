import re
import sys
import os

from sentence_transformers import SentenceTransformer, util

# Add the retrieval folder to Python's import path
sys.path.append(
    os.path.dirname(os.path.abspath(__file__))
)

from evidence_retriever import retrieve_evidence


MODEL_NAME = (
    "sentence-transformers/"
    "paraphrase-multilingual-MiniLM-L12-v2"
)


def split_into_sentences(text):
    """
    Split webpage text into individual sentences.
    """

    text = re.sub(r"\s+", " ", text).strip()

    if not text:
        return []

    sentences = re.split(
        r"(?<=[.!?])\s+",
        text
    )

    return [
        sentence.strip()
        for sentence in sentences
        if len(sentence.split()) >= 8
    ]


def semantic_rank_evidence(
    claim,
    evidence_list,
    model,
    top_k=10
):
    """
    Rank individual evidence sentences by semantic
    similarity to the claim while considering source quality.
    """

    passages = []

    for evidence in evidence_list:

        text = evidence.get("text", "").strip()

        if not text:
            text = evidence.get("snippet", "").strip()

        if not text:
            continue

        sentences = split_into_sentences(text)

        for sentence in sentences:

            passages.append({
                "title": evidence.get("title", ""),
                "url": evidence.get("url", ""),
                "snippet": evidence.get("snippet", ""),
                "text": sentence,
                "source_quality": evidence.get(
                    "source_quality",
                    "Unknown"
                ),
                "source_quality_score": evidence.get(
                    "source_quality_score",
                    0.5
                )
            })

    if not passages:
        return []

    # Encode claim
    claim_embedding = model.encode(
        claim,
        convert_to_tensor=True,
        normalize_embeddings=True
    )

    # Encode individual evidence sentences
    passage_texts = [
        passage["text"]
        for passage in passages
    ]

    passage_embeddings = model.encode(
        passage_texts,
        convert_to_tensor=True,
        normalize_embeddings=True,
        show_progress_bar=False
    )

    similarities = util.cos_sim(
        claim_embedding,
        passage_embeddings
    )[0]

    ranked_results = []

    for passage, similarity in zip(
        passages,
        similarities
    ):

        semantic_similarity = float(similarity)

        combined_score = (
            0.75 * semantic_similarity
            + 0.25 * passage[
                "source_quality_score"
            ]
        )

        passage["semantic_similarity"] = (
            semantic_similarity
        )

        passage["combined_score"] = (
            combined_score
        )

        ranked_results.append(passage)

    ranked_results.sort(
        key=lambda x: x["combined_score"],
        reverse=True
    )

    return ranked_results[:top_k]


def main():

    claim = input("\nEnter claim: ").strip()

    if not claim:
        print("No claim entered.")
        return

    print("\nLoading multilingual embedding model...")

    model = SentenceTransformer(MODEL_NAME)

    print("\nRetrieving evidence...")

    evidence = retrieve_evidence(
        claim,
        max_results=5
    )

    print(
        f"\nRetrieved {len(evidence)} sources."
    )

    print("\nRanking individual evidence sentences...")

    ranked_evidence = semantic_rank_evidence(
        claim,
        evidence,
        model
    )

    print("\n" + "=" * 80)
    print("CLAIM")
    print("=" * 80)

    print(claim)

    print("\n" + "=" * 80)
    print("TOP EVIDENCE SENTENCES")
    print("=" * 80)

    for i, item in enumerate(
        ranked_evidence,
        start=1
    ):

        print(f"\n--- Evidence {i} ---")

        print(f"Title: {item['title']}")
        print(
            f"Source Quality: "
            f"{item['source_quality']}"
        )

        print(
            f"Semantic Similarity: "
            f"{item['semantic_similarity']:.4f}"
        )

        print(
            f"Combined Score: "
            f"{item['combined_score']:.4f}"
        )

        print(f"URL: {item['url']}")

        print(
            f"Evidence:\n{item['text']}"
        )


if __name__ == "__main__":
    main()