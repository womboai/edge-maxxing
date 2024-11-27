from pydantic import BaseModel

class TextToImageRequest(BaseModel):
    prompt: str
    negative_prompt: str | None = None

    width: int | None = None
    height: int | None = None

    seed: int | None = None
