import numpy as np
import base64
from PIL import Image
from io import BytesIO
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, BitsAndBytesConfig, CLIPImageProcessor
from config import PlusConfig
from utils.utils import (DEFAULT_IM_END_TOKEN, DEFAULT_IM_START_TOKEN,
                         DEFAULT_IMAGE_TOKEN)
from model.llava import conversation as conversation_lib
from model.segment_anything.utils.transforms import ResizeLongestSide
from model.llava.mm_utils import tokenizer_image_token

config = PlusConfig()

def prompt_preproccess(prompt, tokenizer):
    conv = conversation_lib.conv_templates[config.conv_type].copy()
    conv.messages = []

    prompt = DEFAULT_IMAGE_TOKEN + "\n" + prompt
    if config.use_mm_start_end:
        replace_token = (
            DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN
        )
        prompt = prompt.replace(DEFAULT_IMAGE_TOKEN, replace_token)

    conv.append_message(conv.roles[0], prompt)
    conv.append_message(conv.roles[1], "")
    prompt = conv.get_prompt()
    input_ids = tokenizer_image_token(prompt, tokenizer, return_tensors="pt")
    input_ids = input_ids.unsqueeze(0).cuda()

    return input_ids


def image_preprocess(base64_image_str):
    header, encoded = base64_image_str.split(",", 1)
    image_bytes = base64.b64decode(encoded)
    pil_image = Image.open(BytesIO(image_bytes)).convert("RGB")

    image_np = np.array(pil_image)
    original_size_list = [image_np.shape[:2]]

    clip_image_processor = CLIPImageProcessor.from_pretrained(config.vision_tower)
    image_clip = (
        clip_image_processor.preprocess(image_np, return_tensors="pt")["pixel_values"][0]
        .unsqueeze(0)
        .cuda()
    )

    if config.precision == "bf16":
        image_clip = image_clip.bfloat16()
    elif config.precision == "fp16":
        image_clip = image_clip.half()
    else:
        image_clip = image_clip.float()

    transform = ResizeLongestSide(config.image_size)
    image_resized_np = transform.apply_image(image_np)
    resize_list = [image_resized_np.shape[:2]]

    image = (
        preprocess(torch.from_numpy(image_resized_np).permute(2, 0, 1).contiguous())
        .unsqueeze(0)
        .cuda()
    )

    if config.precision == "bf16":
        image = image.bfloat16()
    elif config.precision == "fp16":
        image = image.half()
    else:
        image = image.float()

    return image_clip, image, resize_list, original_size_list, image_np


def preprocess(
    x,
    pixel_mean=torch.Tensor([123.675, 116.28, 103.53]).view(-1, 1, 1),
    pixel_std=torch.Tensor([58.395, 57.12, 57.375]).view(-1, 1, 1),
    img_size=1024,
) -> torch.Tensor:
    x = (x - pixel_mean) / pixel_std
    h, w = x.shape[-2:]
    padh = img_size - h
    padw = img_size - w
    x = F.pad(x, (0, padw, 0, padh))
    return x