from modal import Image, Stub

stub = Stub(image=Image.debian_slim().pip_install(["torch", "transformers"]))


@stub.function()
def predict(phrase: str):
    from transformers import pipeline

    sentiment_pipeline = pipeline(
        model="distilbert-base-uncased-finetuned-sst-2-english"
    )
    pred = sentiment_pipeline(phrase, truncation=True, max_length=512, top_k=2)
    # pred will look like:
    # [{'label': 'NEGATIVE', 'score': 0.99}, {'label': 'POSITIVE', 'score': 0.01}]
    probs = {p["label"]: p["score"] for p in pred}
    score = probs["POSITIVE"]
    print(f"{score=}")
    return score
