from pydantic import BaseModel, field_validator
from typing import Optional, List
import base64

class ChatData(BaseModel):
    positive_prompt: str
    image: str
    user_id: str
    chat_id: str
    image_name: str
    masks: Optional[str] = None
    negative_prompt: Optional[str] = None

    @field_validator("image")
    @classmethod
    def validate_base64_image(cls, v):
        return cls._validate_base64_image_string(v, field_name="image")

    @field_validator("masks")
    @classmethod
    def validate_base64_mask_list(cls, v):
        if not isinstance(v, list):
            raise ValueError("Mask must be a list of base64-encoded image strings")
        for i, item in enumerate(v):
            cls._validate_base64_image_string(item, field_name=f"masks[{i}]")
        return v

    @staticmethod
    def _validate_base64_image_string(v, field_name="field"):
        if not v.startswith("data:image/"):
            raise ValueError(f"{field_name} must be a base64-encoded data URI starting with 'data:image/'")
        try:
            header, encoded = v.split(",", 1)
            base64.b64decode(encoded)
        except Exception:
            raise ValueError(f"Invalid base64 encoding in {field_name}")
        return v
