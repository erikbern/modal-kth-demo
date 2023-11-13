from modal import Image, Stub, web_endpoint

stub = Stub(
    "kth-demo-sentiment-analysis-webhook",
    image=Image.debian_slim().pip_install(["torch", "transformers"]),
)


@stub.function()
@web_endpoint()
def predict(phrase: str):
    from transformers import pipeline

    sentiment_pipeline = pipeline(
        model="distilbert-base-uncased-finetuned-sst-2-english"
    )
    pred = sentiment_pipeline(phrase, truncation=True, max_length=512, top_k=2)
    probs = {p["label"]: p["score"] for p in pred}
    return probs["POSITIVE"]
