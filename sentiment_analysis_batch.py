import sys

import modal

stub = modal.Stub(
    image=modal.Image.debian_slim().pip_install(["datasets", "torch", "transformers"])
)


class Predictor:
    def __enter__(self):
        from transformers import pipeline

        self.sentiment_pipeline = pipeline(
            model="distilbert-base-uncased-finetuned-sst-2-english"
        )

    @stub.function(concurrency_limit=16, cpu=8)
    def predict(self, phrase: str):
        pred = self.sentiment_pipeline(phrase, truncation=True, max_length=512, top_k=2)
        # pred will look like: [{'label': 'NEGATIVE', 'score': 0.99}, {'label': 'POSITIVE', 'score': 0.01}]
        probs = {p["label"]: p["score"] for p in pred}
        return (phrase, probs["POSITIVE"])


@stub.function
def big_map():
    from datasets import load_dataset

    imdb = load_dataset("imdb")
    data = [row["text"] for row in imdb["test"]]
    for phrase, score in Predictor().predict.map(data):
        print(f"{score:.4f} {phrase[:80]}")


if __name__ == "__main__":
    with stub.run():
        big_map()
