import argparse
import os
import sys

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, BitsAndBytesConfig, CLIPImageProcessor

from model.LISA import LISAForCausalLM
from model.llava import conversation as conversation_lib
from model.llava.mm_utils import tokenizer_image_token
from model.segment_anything.utils.transforms import ResizeLongestSide
from utils.utils import (DEFAULT_IM_END_TOKEN, DEFAULT_IM_START_TOKEN,
                         DEFAULT_IMAGE_TOKEN, IMAGE_TOKEN_INDEX)

from config import PlusConfig
from typing import Optional

def preprocess(
    x,
    pixel_mean=torch.Tensor([123.675, 116.28, 103.53]).view(-1, 1, 1),
    pixel_std=torch.Tensor([58.395, 57.12, 57.375]).view(-1, 1, 1),
    img_size=1024,
) -> torch.Tensor:
    """Normalize pixel values and pad to a square input."""
    # Normalize colors
    x = (x - pixel_mean) / pixel_std
    # Pad
    h, w = x.shape[-2:]
    padh = img_size - h
    padw = img_size - w
    x = F.pad(x, (0, padw, 0, padh))
    return x


def load_model(tokenizer, config: Optional[PlusConfig] = None):
    config = config or PlusConfig()

    os.makedirs(config.vis_save_path, exist_ok=True)

    config.seg_token_idx = tokenizer("[SEG]", add_special_tokens=False).input_ids[0]


    torch_dtype = torch.float32
    if config.precision == "bf16":
        torch_dtype = torch.bfloat16
    elif config.precision == "fp16":
        torch_dtype = torch.half

    kwargs = {"torch_dtype": torch_dtype}
    if config.load_in_4bit:
        kwargs.update(
            {
                "torch_dtype": torch.half,
                "load_in_4bit": True,
                "quantization_config": BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_compute_dtype=torch.float16,
                    bnb_4bit_use_double_quant=True,
                    bnb_4bit_quant_type="nf4",
                    llm_int8_skip_modules=["visual_model"],
                ),
            }
        )
    elif config.load_in_8bit:
        kwargs.update(
            {
                "torch_dtype": torch.half,
                "quantization_config": BitsAndBytesConfig(
                    llm_int8_skip_modules=["visual_model"],
                    load_in_8bit=True,
                ),
            }
        )

    model = LISAForCausalLM.from_pretrained(
        config.version, low_cpu_mem_usage=True, vision_tower=config.vision_tower, seg_token_idx=config.seg_token_idx, **kwargs
    )

    model.config.eos_token_id = tokenizer.eos_token_id
    model.config.bos_token_id = tokenizer.bos_token_id
    model.config.pad_token_id = tokenizer.pad_token_id

    model.get_model().initialize_vision_modules(model.get_model().config)
    vision_tower = model.get_model().get_vision_tower()
    vision_tower.to(dtype=torch_dtype)

    if config.precision == "bf16":
        model = model.bfloat16().cuda()
    elif (
        config.precision == "fp16" and (not config.load_in_4bit) and (not config.load_in_8bit)
    ):
        vision_tower = model.get_model().get_vision_tower()
        model.model.vision_tower = None
        import deepspeed

        model_engine = deepspeed.init_inference(
            model=model,
            dtype=torch.half,
            replace_with_kernel_inject=True,
            replace_method="auto",
        )
        model = model_engine.module
        model.model.vision_tower = vision_tower.half().cuda()
    elif config.precision == "fp32":
        model = model.float().cuda()

    vision_tower = model.get_model().get_vision_tower()
    vision_tower.to(device=config.local_rank)

    model.eval()
    return model

    clip_image_processor = CLIPImageProcessor.from_pretrained(model.config.vision_tower)
    transform = ResizeLongestSide(config.image_size)

    model.eval()

    while True:
        conv = conversation_lib.conv_templates[config.conv_type].copy()
        conv.messages = []

        prompt = input("Please input your prompt: ")
        prompt = DEFAULT_IMAGE_TOKEN + "\n" + prompt
        if config.use_mm_start_end:
            replace_token = (
                DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN
            )
            prompt = prompt.replace(DEFAULT_IMAGE_TOKEN, replace_token)

        conv.append_message(conv.roles[0], prompt)
        conv.append_message(conv.roles[1], "")
        prompt = conv.get_prompt()

        image_path = input("Please input the image path: ")
        if not os.path.exists(image_path):
            print("File not found in {}".format(image_path))
            continue

        image_np = cv2.imread(image_path)
        image_np = cv2.cvtColor(image_np, cv2.COLOR_BGR2RGB)
        original_size_list = [image_np.shape[:2]]

        image_clip = (
            clip_image_processor.preprocess(image_np, return_tensors="pt")[
                "pixel_values"
            ][0]
            .unsqueeze(0)
            .cuda()
        )
        if config.precision == "bf16":
            image_clip = image_clip.bfloat16()
        elif config.precision == "fp16":
            image_clip = image_clip.half()
        else:
            image_clip = image_clip.float()

        image = transform.apply_image(image_np)
        resize_list = [image.shape[:2]]

        image = (
            preprocess(torch.from_numpy(image).permute(2, 0, 1).contiguous())
            .unsqueeze(0)
            .cuda()
        )
        if config.precision == "bf16":
            image = image.bfloat16()
        elif config.precision == "fp16":
            image = image.half()
        else:
            image = image.float()

        input_ids = tokenizer_image_token(prompt, tokenizer, return_tensors="pt")
        input_ids = input_ids.unsqueeze(0).cuda()

        output_ids, pred_masks = model.evaluate(
            image_clip,
            image,
            input_ids,
            resize_list,
            original_size_list,
            max_new_tokens=512,
            tokenizer=tokenizer,
        )
        output_ids = output_ids[0][output_ids[0] != IMAGE_TOKEN_INDEX]

        text_output = tokenizer.decode(output_ids, skip_special_tokens=False)
        text_output = text_output.replace("\n", "").replace("  ", " ")
        print("text_output: ", text_output)

        print("len(pred_masks): ", len(pred_masks))
        print("[x.shape for x in pred_masks]: ", [x.shape for x in pred_masks])
        
        visualize(config, pred_masks, image_path, image_np)

def save_mask_as_image(pred_mask, hex_color, file_path):
    """
    Save a boolean mask as an image, coloring True values with the specified color.

    :param pred_mask: A 2D numpy array of boolean values.
    :param hex_color: String of the color in hexadecimal format (e.g., '#FF5733').
    :param file_path: Path to save the image file.
    """
    # Convert hex color to BGR
    hex_color = hex_color.lstrip('#')
    rgb_color = tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))
    bgr_color = rgb_color[::-1]

    # Create an image from the mask
    image = np.zeros((pred_mask.shape[0], pred_mask.shape[1], 3), dtype=np.uint8)
    for i, color in enumerate(bgr_color):
        image[pred_mask, i] = color

    # Save the image
    cv2.imwrite(file_path, image)

def visualize(args, pred_masks, image_path, image_np):
    def hex_to_rgb(hex_color):
        hex_color = hex_color.lstrip('#')
        lv = len(hex_color)
        return tuple(int(hex_color[i:i + lv // 3], 16) for i in range(0, lv, lv // 3))
    colormap = ["#1f77b4","#d62728","#ff7f0e","#2ca02c","#9467bd","#8c564b","#e377c2",\
                "#7f7f7f","#bcbd22","#17becf","#9467bd","#8c564b","#e377c2","#7f7f7f","#bcbd22","#17becf", \
                "#1f77b4","#41340e","#2ca02c","#b62728","#9467bd","#8ff64b","#e377c2","#7f7faa","#bcbd22","#17baaf"]
    colormap_rgb = [hex_to_rgb(color) for color in colormap]
    save_path_tot = "{}/{}_masked_img.jpg".format(
        args.vis_save_path, image_path.split("/")[-1].split(".")[0]
    )
    save_img = image_np.copy()
    
    for i, pred_mask in enumerate(pred_masks[0]):
        if pred_mask.shape[0] == 0:
            continue

        pred_mask = pred_mask.detach().cpu().numpy()#[0]
        pred_mask = pred_mask > 0

        save_path = "{}/{}_mask_{}.jpg".format(
            args.vis_save_path, image_path.split("/")[-1].split(".")[0], i
        )
        cv2.imwrite(save_path, pred_mask * 100)
        print("{} has been saved.".format(save_path))
        
        save_img[pred_mask] = (
            image_np * 0.5
            + pred_mask[:, :, None].astype(np.uint8) * np.array(colormap_rgb[i % len(colormap_rgb)]) * 0.5
        )[pred_mask]
    save_img = cv2.cvtColor(save_img, cv2.COLOR_RGB2BGR)
    cv2.imwrite(save_path_tot, save_img)
    print("{} has been saved.".format(save_path_tot))
    
    
    
if __name__ == "__main__":
    config = PlusConfig()
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(
            config.version,
            cache_dir=None,
            model_max_length=config.model_max_length,
            padding_side="right",
            use_fast=False,
        )
    tokenizer.pad_token = tokenizer.unk_token
    model = load_model(tokenizer, config)
