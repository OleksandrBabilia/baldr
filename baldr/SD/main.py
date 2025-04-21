from diffusers import StableDiffusionXLInpaintPipeline
import torch
from PIL import Image
import numpy as np

damaged_ground = Image.open("/workspace/diploma/baldr/baldr/LISA/imgs/camera_lens.jpg").convert("RGB")  
mask= Image.open("/workspace/diploma/baldr/baldr/LISA/vis_output/camera_lens_mask_1.jpg").convert("L")  


original_size = mask.size  # (width, height)
processed_mask = mask.convert("L")
processed_mask_array = np.array(processed_mask)
binary_mask_array = np.where(processed_mask_array > 0, 255, 0).astype(np.uint8)
processed_mask = Image.fromarray(binary_mask_array)
resized_image = processed_mask.resize(original_size, Image.LANCZOS)

pipe = StableDiffusionXLInpaintPipeline.from_pretrained(
    "stabilityai/stable-diffusion-xl-1.0-inpainting",
    torch_dtype=torch.float16,
).to("cuda")  

restored_landscape = pipe(
    prompt="Vibrant green bushes with dense, healthy leaves, beautifully landscaped shrubs, soft grass underneath, colorful flowers, sunlight filtering through the leaves, a peaceful and serene atmosphere.",
    image=damaged_ground,
    mask_image=processed_mask,
    negative_prompt="trash, rubble, broken bricks, debris, dirt, dust, abandoned wreckage, ruined ground, burnt marks, ash, garbage, dead plants, dry patches",
).images[0]

restored_landscape.save("restored_bushes.png")
