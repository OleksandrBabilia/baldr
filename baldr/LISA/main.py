from fastapi import FastAPI
from transformers import AutoTokenizer
from config import PlusConfig
import re
from API.schemas import ChatData
from load_model import load_model
from API.utils.preproccess import prompt_preproccess, image_preprocess
from API.utils.visualize import visualize
from utils.utils import IMAGE_TOKEN_INDEX
from pprint import pprint

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

app = FastAPI()

@app.get("/")
def home():
    return {"message": "succsess"}

@app.post("/chat")
def chat(data: ChatData):
    prompt = prompt_preproccess(data.prompt, tokenizer)
    image_clip, image, resize_list, original_size_list, image_np = image_preprocess(data.image)
    output_ids, pred_masks = model.evaluate(
        image_clip,
        image,
        prompt,
        resize_list,
        original_size_list,
        max_new_tokens=512,
        tokenizer=tokenizer,
    )
    output_ids = output_ids[0][output_ids[0] != IMAGE_TOKEN_INDEX]

    text_output = tokenizer.decode(output_ids, skip_special_tokens=True)
    # text_output = text_output.replace("\n", "").replace("  ", " ")
    match = re.search(r'ASSISTANT:\s*(.*)', text_output, re.DOTALL)
    assistant_response = match.group(1) if match else None
    print("text_output: ", text_output)
    response = {"message": assistant_response}

    print("len(pred_masks): ", len(pred_masks))
    if len(pred_masks) > 0:
        print("[x.shape for x in pred_masks]: ", [x.shape for x in pred_masks])
        img = visualize(config, pred_masks, data.image, image_np)
        response["img"] = img
   
    return response