import base64
from io import BytesIO
import numpy as np
from PIL import Image

def decode_base64_to_pil(base64_str):
    image_data = base64.b64decode(base64_str)
    image = Image.open(BytesIO(image_data))
    return image

def decode_base64_uri_to_pil(base64_data):
    image_data = base64.b64decode(base64_data.split(",")[1])  # Remove 'data:image/...;base64,' part
    return Image.open(BytesIO(image_data))

def pil_to_data_uri(image: Image, image_format="JPEG"):
    buffered = BytesIO()
    image.save(buffered, format=image_format)
    img_bytes = buffered.getvalue()
    base64_encoded = base64.b64encode(img_bytes).decode("utf-8")
    return f"data:image/{image_format.lower()};base64,{base64_encoded}"

def preprocess_masks(mask_base64_list):
    if not mask_base64_list:
        raise ValueError("mask_list is empty")
    print("Het we are on the preprocess_masks function")
    base = decode_base64_to_pil(mask_base64_list[0])
    base_array = np.array(base.convert("L"))
    aggregate = np.where(base_array > 15, 255, 0).astype(np.uint8)

    for mask_base64 in mask_base64_list[1:]:
        mask = mask_base64
        gray = mask.convert("L")
        arr = np.array(gray)
        binary = np.where(arr > 15, 255, 0).astype(np.uint8)
        aggregate = np.maximum(aggregate, binary)  

    processed_mask = Image.fromarray(aggregate)
    print(type(processed_mask))
    return processed_mask
