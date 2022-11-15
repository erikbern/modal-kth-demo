import io
import os
from typing import Optional

import modal

stub = modal.Stub()

volume = modal.SharedVolume().persist("stable-diff-model-vol")

CACHE_PATH = "/root/model_cache"


@stub.function(
    gpu=True,
    image=(
        modal.Image.debian_slim()
        .run_commands(["pip install torch --extra-index-url https://download.pytorch.org/whl/cu117"])
        .pip_install(["diffusers", "transformers", "scipy", "ftfy", "accelerate"])
    ),
    shared_volumes={CACHE_PATH: volume},
    secret=modal.Secret.from_name("huggingface-secret"),
)
async def run_stable_diffusion(prompt: str, channel_name: Optional[str] = None):
    from diffusers import StableDiffusionPipeline
    from torch import float16

    pipe = StableDiffusionPipeline.from_pretrained(
        "runwayml/stable-diffusion-v1-5",
        use_auth_token=os.environ["HUGGINGFACE_TOKEN"],
        revision="fp16",
        torch_dtype=float16,
        cache_dir=CACHE_PATH,
        device_map="auto",
    )

    image = pipe(prompt, num_inference_steps=100).images[0]

    # Convert PIL Image to PNG byte array.
    buf = io.BytesIO()
    image.save(buf, format="PNG")
    img_bytes = buf.getvalue()

    if channel_name:
        # `post_to_slack` is implemented further below.
        post_image_to_slack(prompt, channel_name, img_bytes)

    return img_bytes



if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1:
        prompt = sys.argv[1]
    else:
        prompt = "oil painting of a shiba"

    with stub.run():
        img_bytes = run_stable_diffusion(prompt)
        output_path = "stable_diffusion.png"
        with open(output_path, "wb") as f:
            f.write(img_bytes)
        print(f"Wrote data to {output_path}")
