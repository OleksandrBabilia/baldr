from diffusers  import StableDiffusionXLInpaintPipeline
import torch
from PIL import Image
import numpy as np
# damaged_ground = Image.open("imgs/building_org.jpg").convert("RGB")  
# mask=  Image.open("imgs/building_mask.jpg").convert("L")  

# original_size = mask.size 
# processed_mask = mask.convert("L")
# processed_mask_array = np.array(processed_mask)
# binary_mask_array = np.where(processed_mask_array > 15, 255, 0).astype(np.uint8)
# processed_mask = Image.fromarray(binary_mask_array)
# resized_image = processed_mask.resize(original_size, Image.LANCZOS)
# pipe = StableDiffusionXLInpaintPipeline.from_pretrained(
#     "diffusers/stable-diffusion-xl-1.0-inpainting-0.1",
#     torch_dtype=torch.float16,
# ).to(device)  
# p_prompt = "A beautifully restored historical 1930s university building with elegant neoclassical architecture, clean beige and white stone facade, intricate decorative details, large restored windows, a grand entrance, and a fresh coat of paint."
# n_prompt = "burnt walls, fire damage, broken windows, shattered glass, debris, cracks, ruined structure, destroyed facade, bullet holes, dust, rubble"
# restored_landscape = pipe(
#     prompt=p_prompt,
#     image=damaged_ground,
#     mask_image=processed_mask,
#     negative_prompt=n_prompt,
# ).images[0]

# restored_landscape.save("imgs/building_result.png")



from fastapi import FastAPI
from API.schemas import ChatData
from API.utils.preprocess import preprocess_masks, decode_base64_to_pil, pil_to_data_uri
from API.utils.device import get_device 
from API.utils.s3 import download_masks 

pipe = StableDiffusionXLInpaintPipeline.from_pretrained(
    "diffusers/stable-diffusion-xl-1.0-inpainting-0.1",
    torch_dtype=torch.float16,
).to(get_device())  

app = FastAPI()

@app.get("/")
def home():
    return {"message": "succsess"}

@app.post("/inpaint")
def chat(data: ChatData):
    masks = download_masks(data.chat_id, data.user_id, data.image_name)
    result: Image = pipe(
        prompt=data.positive_prompt,
        image=decode_base64_to_pil(data.image),
        mask_image=masks,
        negative_prompt=data.negative_prompt,
    ).images[0]
    result.save("imgs/api_building_result.png")
    result_data_uri = pil_to_data_uri(result)
    print(result)
    return {"img": result_data_uri} 