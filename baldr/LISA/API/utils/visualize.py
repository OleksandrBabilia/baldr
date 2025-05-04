import os
import cv2
import numpy as np
import base64
import boto3
import uuid
from io import BytesIO
from dotenv import load_dotenv
from datetime import datetime

load_dotenv()

s3 = boto3.client(
    's3',
    endpoint_url='https://s3.eu-central-2.wasabisys.com', 
    aws_access_key_id=os.getenv('WASABI_ACCESS_KEY'),
    aws_secret_access_key=os.getenv('WASABI_SECRET_KEY'),
)
print(os.getenv('WASABI_ACCESS_KEY'), os.getenv('WASABI_SECRET_KEY'))

BUCKET_NAME = 'greenly'

def visualize(pred_masks, image_np, data_user_id, data_chat_id):
    def hex_to_rgb(hex_color):
        hex_color = hex_color.lstrip('#')
        lv = len(hex_color)
        return tuple(int(hex_color[i:i + lv // 3], 16) for i in range(0, lv, lv // 3))

    colormap = [
        "#1f77b4", "#d62728", "#ff7f0e", "#2ca02c", "#9467bd", "#8c564b", "#e377c2",
        "#7f7f7f", "#bcbd22", "#17becf", "#9467bd", "#8c564b", "#e377c2", "#7f7f7f",
        "#bcbd22", "#17becf", "#1f77b4", "#41340e", "#2ca02c", "#b62728", "#9467bd",
        "#8ff64b", "#e377c2", "#7f7faa", "#bcbd22", "#17baaf"
    ]
    colormap_rgb = [hex_to_rgb(color) for color in colormap]

    save_img = image_np.copy()

    unique_id = str(uuid.uuid4())
    final_image_name = f"{unique_id}.jpg"
    final_image_path = f"{data_user_id}/{data_chat_id}/{final_image_name}"
    mask_folder_path = f"{data_user_id}/{data_chat_id}/masks"

    for i, pred_mask in enumerate(pred_masks[0]):
        if pred_mask.shape[0] == 0:
            continue

        pred_mask_np = pred_mask.detach().cpu().numpy()
        binary_mask = pred_mask_np > 0

        overlay_color = np.array(colormap_rgb[i % len(colormap_rgb)])
        blended = image_np * 0.5 + binary_mask[:, :, None] * overlay_color * 0.5
        save_img = np.where(binary_mask[:, :, None], blended, save_img)

        # Save mask
        mask_img = (binary_mask.astype(np.uint8)) * 255
        mask_img = cv2.cvtColor(mask_img, cv2.COLOR_GRAY2BGR)
        _, mask_encoded = cv2.imencode('.jpg', mask_img)
        mask_key = f"{mask_folder_path}/{unique_id}_mask_{i+1}.jpg"
        s3.upload_fileobj(BytesIO(mask_encoded.tobytes()), BUCKET_NAME, mask_key)


    save_img = np.clip(save_img, 0, 255).astype(np.uint8)
    save_img_bgr = cv2.cvtColor(save_img, cv2.COLOR_RGB2BGR)
    success, encoded_image = cv2.imencode('.jpg', save_img_bgr)
    if not success:
        raise ValueError("Image encoding failed.")

    s3.upload_fileobj(BytesIO(encoded_image.tobytes()), BUCKET_NAME, final_image_path)

    jpg_bytes = encoded_image.tobytes()
    base64_str = base64.b64encode(jpg_bytes).decode('utf-8')

    return f"data:image/jpeg;base64,{base64_str}", final_image_name
