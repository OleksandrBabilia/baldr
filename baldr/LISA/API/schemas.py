from pydantic import BaseModel

class ChatData(BaseModel):
    prompt: str
    image: str