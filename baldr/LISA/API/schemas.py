from pydantic import BaseModel, field_validator
from typing import Optional
import base64

class ChatData(BaseModel):
    prompt: str
    user_id: str
    chat_id: str
    image: Optional[str] = None
    object: Optional[str] = None

    @field_validator("image")
    @classmethod
    def validate_base64_image(cls, v):
        if v is None:
            return v
        if not v.startswith("data:image/"):
            raise ValueError("Image must be a base64-encoded data URI starting with 'data:image/'")
        try:
            header, encoded = v.split(",", 1)
            base64.b64decode(encoded)
        except Exception:
            raise ValueError("Invalid base64 encoding in image")
        return v
