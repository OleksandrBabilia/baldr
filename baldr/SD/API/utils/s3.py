import base64
import boto3
from dotenv import load_dotenv
import os
from io import BytesIO
from PIL import Image
import numpy as np

load_dotenv()

s3 = boto3.client(
    's3',
    endpoint_url='https://s3.eu-central-2.wasabisys.com',  
    aws_access_key_id=os.getenv('WASABI_ACCESS_KEY'),
    aws_secret_access_key=os.getenv('WASABI_SECRET_KEY'),
)

BUCKET_NAME = 'greenly'

def download_masks(user_id, chat_id, image_name):
    mask_folder_path = f"{user_id}/{chat_id}/masks"
    mask_base64_list = []

    try:
        response = s3.list_objects_v2(Bucket=BUCKET_NAME, Prefix=mask_folder_path)

        if 'Contents' not in response:
            print(f"No masks found for {image_name}.")
            return []

        for obj in response['Contents']:
            mask_key = obj['Key']

            if image_name in mask_key and mask_key.endswith('.jpg'):
                mask_object = s3.get_object(Bucket=BUCKET_NAME, Key=mask_key)
                mask_data = mask_object['Body'].read()

                mask_base64 = base64.b64encode(mask_data).decode('utf-8')
                mask_base64_list.append(mask_base64)

    except Exception as e:
        print(f"Error downloading masks: {e}")
        return []

    return mask_base64_list
