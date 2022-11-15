import sys

import modal

stub = modal.Stub(
    "kth-demo-sentiment-analysis-webhook",
    image=modal.Image.debian_slim().pip_install(["torch", "transformers"]),
)


@stub.webhook
def predict(phrase: str):
    from transformers import pipeline

    sentiment_pipeline = pipeline(
        model="distilbert-base-uncased-finetuned-sst-2-english"
    )
    pred = sentiment_pipeline(phrase, truncation=True, max_length=512, top_k=2)
    probs = {p["label"]: p["score"] for p in pred}
    return probs["POSITIVE"]


if __name__ == "__main__":
    stub.serve()
