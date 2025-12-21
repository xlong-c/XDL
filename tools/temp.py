# from diffusers.pipelines.prx import PRXPipeline
# import torch

# pipe = PRXPipeline.from_pretrained(
#     "Photoroom/prx-1024-t2i-beta",
#     torch_dtype=torch.bfloat16
# ).to("cuda")

# prompt = 'A capybara holding a sign that reads Hello World'
# image = pipe(prompt, num_inference_steps=28, guidance_scale=3.5,height= 512,width=1024).images[0]
# image.save("lion.png")



import torch
from diffusers import StableDiffusion3Pipeline
from huggingface_hub import login
login(token="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")

pipe = StableDiffusion3Pipeline.from_pretrained("stabilityai/stable-diffusion-3.5-medium", torch_dtype=torch.float16)
print(pipe.transformer)
# pipe = pipe.to("cuda")
# image = pipe(
#     "A red capybara holding a sign that reads Hello World",
#     num_inference_steps=20,
#     guidance_scale=3.5,
#     width=512*3,
#     height=512
# ).images[0]
# image.save("capybara.png")
