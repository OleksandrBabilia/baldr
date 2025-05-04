from pydantic import BaseModel, field_validator
from typing import Optional, List
import base64

class ChatData(BaseModel):
    positive_prompt: str
    original_image: str
    user_id: str
    chat_id: str
    image_name: str
    current_image: str  
    negative_prompt: Optional[str] = None


