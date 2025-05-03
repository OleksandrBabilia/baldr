import base64
from io import BytesIO
import requests
from PIL import Image 


def pil_to_data_uri(image, image_format="JPEG"):
    buffered = BytesIO()
    image.save(buffered, format=image_format)
    image_bytes = buffered.getvalue()
    base64_encoded = base64.b64encode(image_bytes).decode("utf-8")
    return f"data:image/{image_format.lower()};base64,{base64_encoded}"

p_prompt = "A beautifully restored historical 1930s university building with elegant neoclassical architecture, clean beige and white stone facade, intricate decorative details, large restored windows, a grand entrance, and a fresh coat of paint."
n_prompt = "burnt walls, fire damage, broken windows, shattered glass, debris, cracks, ruined structure, destroyed facade, bullet holes, dust, rubble"

img = Image.open("imgs/building_org.jpg").convert("RGB")  
mask = Image.open("imgs/building_mask.jpg").convert("L")  

img_base64 = pil_to_data_uri(img)
mask_base64 = pil_to_data_uri(mask)

payload = {
    "prompt": p_prompt,
    "negative_prompt": n_prompt,
    "image": img_base64,
    "masks": [mask_base64]
}
response = response = requests.post("http://localhost:8000/inpaint", json=payload)
print(response.status_code, response.text)
