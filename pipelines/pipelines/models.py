from pydantic import BaseModel


SUBMISSION_SPEC_VERSION = 6


class TextToImageRequest(BaseModel):
    prompt: str
    negative_prompt: str | None = None

    width: int | None = None
    height: int | None = None

    seed: int | None = None
