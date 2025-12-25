from huggingface_hub import login
login(token="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")
from transformers import pipeline

generator = pipeline(
    "image-text-to-text",
    model="google/t5gemma-2-1b-1b",
)

generator(
    "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/bee.jpg",
    text="<start_of_image> in this image, there is",
    generate_kwargs={"do_sample": False, "max_new_tokens": 50},
)