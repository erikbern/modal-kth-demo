from modal import Image, Stub, method

stub = Stub(
    image=Image.debian_slim().pip_install(["datasets", "torch", "transformers"])
)

with stub.image.run_inside():
    from transformers import pipeline
    from datasets import load_dataset


@stub.cls(concurrency_limit=64, gpu="a100")
class Predictor:
    def __enter__(self):
        self.sentiment_pipeline = pipeline(
            model="distilbert-base-uncased-finetuned-sst-2-english"
        )

    @method()
    def predict(self, phrase: str):
        pred = self.sentiment_pipeline(phrase, truncation=True, max_length=512, top_k=2)
        probs = {p["label"]: p["score"] for p in pred}
        return (phrase, probs["POSITIVE"])


@stub.function()
def big_map():
    imdb = load_dataset("imdb")
    data = [row["text"] for row in imdb["test"]]
    for phrase, score in Predictor().predict.map(data):
        print(f"{score:.4f} {phrase[:80]}")


@stub.local_entrypoint()
def run():
    big_map.remote()
