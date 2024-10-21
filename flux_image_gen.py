import fal_client
import base64
import torch
import numpy as np
from PIL import Image
import io
import tempfile
import os
import requests

class FalFluxNode:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "width": ("INT", {"default": 1024}),
                "height": ("INT", {"default": 1024}),
                "seed": ("INT", {"default": 0}),
                "prompt": ("STRING", {"multiline": True}),
                "model": (["pro", "dev", "schnell"], {"default": "schnell"}),
            },
            "optional": {
                "controlnet_image": ("IMAGE",),
                "controlnet_union_model": ("STRING", {"default": "https://huggingface.co/InstantX/FLUX.1-dev-Controlnet-Union/blob/main/diffusion_pytorch_model.safetensors"}),
                "controlnet_strength": ("FLOAT", {"default": 0.7, "min": 0.0, "max": 1.0, "step": 0.01}),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "generate_image"
    CATEGORY = "Katalist Tools"

    def generate_image(self, width, height, seed, prompt, model, controlnet_image=None, controlnet_union_model=None, controlnet_strength=0.3):
        url = 'fal-ai/flux'
        if model == 'dev':
            url = 'fal-ai/flux/dev'
        elif model == 'schnell':
            url = 'fal-ai/flux/schnell'

        # Prepare the arguments
        arguments = {
            "prompt": prompt,
            "image_size": {
                "width": width,
                "height": height
            },
            "sync_mode": True,
            "seed": seed
        }

        # Add controlnet if image is provided
        if controlnet_image is not None and model == 'dev':
            
            # change model to flux-general
            url = 'fal-ai/flux-general'

            # Convert torch tensor to PIL Image
            pil_image = Image.fromarray((controlnet_image[0].numpy() * 255).astype(np.uint8))
            
            # Save image as a temporary file
            with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as temp_file:
                pil_image.save(temp_file, format="PNG")
                temp_file_path = temp_file.name

            try:
                # Upload the temporary file and get the URL
                file_url = fal_client.upload_file(temp_file_path)
                print(f"Uploaded controlnet image URL: {file_url}")
                
                arguments["controlnet_unions"] = [{
                    "path": controlnet_union_model,
                    "variant": None,
                    "controls": [{
                        "control_image_url": file_url,
                        "conditioning_scale": controlnet_strength,
                        "control_mode": "pose"
                    }]
                }]
            finally:
                # Clean up the temporary file
                os.unlink(temp_file_path)

        # Submit the job
        handler = fal_client.submit(url, arguments=arguments)
        result = handler.get()

        # Process the result
        if controlnet_image is not None and model == 'dev':

            print(result)

            # download image from url
            image_data_url = result['images'][0]['url']
            # download image
            image_data = requests.get(image_data_url).content
            # convert to torch image
            pil_image = Image.open(io.BytesIO(image_data))
            np_image = np.array(pil_image).astype(np.float32) / 255.0
            torch_image = torch.from_numpy(np_image)[None,]
        else:
            image_data_url = result['images'][0]['url']
            image_data_base64 = image_data_url.split(',')[1]
            image_data = base64.b64decode(image_data_base64)

            pil_image = Image.open(io.BytesIO(image_data))
            np_image = np.array(pil_image).astype(np.float32) / 255.0
            torch_image = torch.from_numpy(np_image)[None,]

        return (torch_image,)

# NODE_CLASS_MAPPINGS = {
#     "FalFluxNode": FalFluxNode
# }

# NODE_DISPLAY_NAME_MAPPINGS = {
#     "FalFluxNode": "Fal Flux Image Generator"
# }
