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

    def decode_and_binarize(base64_str):
        image = decode_base64_to_pil(base64_str).convert("L")  # grayscale
        image = image.resize((1024, 1024), Image.Resampling.BILINEAR)  # ensure smooth up/downscale
        arr = np.array(image)
        binary = np.where(arr > 15, 255, 0).astype(np.uint8)
        return binary

    # Decode and binarize the first mask
    aggregate = decode_and_binarize(mask_base64_list[0])

    # Combine the rest
    for mask_base64 in mask_base64_list[1:]:
        binary = decode_and_binarize(mask_base64)
        aggregate = np.maximum(aggregate, binary)

    processed_mask = Image.fromarray(aggregate).convert("L")  # keep grayscale
    return processed_mask
