import fal_client
import base64
import torch
import numpy as np
from PIL import Image
import io

class FalFluxNode:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "width": ("INT",{"default": 1024}),
                "height": ("INT",{ "default": 1024}),
                "seed": ("INT", {"default": 0}),
                "prompt": ("STRING", {"multiline": True}),
                "model": (["pro", "dev", "schnell"], {"default": "schnell"}),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "generate_image"
    CATEGORY = "Katalist Tools"

    def generate_image(self, width, height, seed, prompt, model):
        url = 'fal-ai/flux'
        if model == 'dev':
            url = 'fal-ai/flux/dev'
        elif model == 'schnell':
            url = 'fal-ai/flux/schnell'

        # payload
        handler = fal_client.submit(
            url,
            arguments={
                "prompt": prompt,
                "image_size": {
                    "width": width,
                    "height": height
                },
                "sync_mode": True,
                "seed": seed
            },
        )

        result = handler.get()

        # get image result and decode it from base64
        # Extract the base64 image data
        image_data_url = result['images'][0]['url']
        image_data_base64 = image_data_url.split(',')[1]
        
        # Decode the base64 image data
        image_data = base64.b64decode(image_data_base64)

        # Convert to PIL Image
        pil_image = Image.open(io.BytesIO(image_data))

        # Convert PIL Image to numpy array
        np_image = np.array(pil_image).astype(np.float32) / 255.0

        # Convert numpy array to torch tensor
        torch_image = torch.from_numpy(np_image)[None,]

        return (torch_image,)

# NODE_CLASS_MAPPINGS = {
#     "FalFluxNode": FalFluxNode
# }

# NODE_DISPLAY_NAME_MAPPINGS = {
#     "FalFluxNode": "Fal Flux Image Generator"
# }