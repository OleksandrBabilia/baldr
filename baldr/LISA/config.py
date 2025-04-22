from dataclasses import dataclass


@dataclass
class BaseConfig:
    version: str = "xinlai/LISA-13B-llama2-v1"
    vis_save_path: str = "./vis_output"
    precision: str = "bf16"
    image_size: int = 1024
    model_max_length: int = 512
    lora_r: int = 8
    vision_tower: str = "openai/clip-vit-large-patch14"
    local_rank: int = 0
    load_in_8bit: bool = False
    load_in_4bit: bool = False
    use_mm_start_end: bool = False
    conv_type: str = "llava_v1"

@dataclass
class PlusConfig:
    version: str = "Senqiao/LISA_Plus_7b"
    vis_save_path: str = "./vis_output"
    precision: str = "bf16"
    image_size: int = 1024
    model_max_length: int = 512
    lora_r: int = 8
    vision_tower: str = "openai/clip-vit-large-patch14"
    local_rank: int = 0
    load_in_8bit: bool = False
    load_in_4bit: bool = False
    use_mm_start_end: bool = False
    conv_type: str = "llava_v1"