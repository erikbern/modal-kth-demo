import sys

import modal

stub = modal.Stub(
    image=modal.Image.debian_slim().pip_install(["torch", "transformers"])
)


@stub.function
def predict(phrase: str):
    from transformers import pipeline

    sentiment_pipeline = pipeline(
        model="distilbert-base-uncased-finetuned-sst-2-english"
    )
    pred = sentiment_pipeline(phrase, truncation=True, max_length=512, top_k=2)
    # pred will look like: [{'label': 'NEGATIVE', 'score': 0.99}, {'label': 'POSITIVE', 'score': 0.01}]
    probs = {p["label"]: p["score"] for p in pred}
    return probs["POSITIVE"]


if __name__ == "__main__":
    with stub.run():
        score = predict(sys.argv[1])
        print(f"score: {score:.4f}")
