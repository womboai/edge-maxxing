from PIL.Image import Image
from diffusers import StableDiffusionXLPipeline
from pydantic import BaseModel
from torch import Generator


class GenerationRequest(BaseModel):
    prompt: str
    negative_prompt: str | None = None

    width: int | None = None
    height: int | None = None

    seed: int | None = None


def load_pipeline() -> StableDiffusionXLPipeline:
    return StableDiffusionXLPipeline.from_pretrained(
        "stablediffusionapi/newdream-sdxl-20",
        cache_dir="./models",
        local_files_only=True,
    )


def infer(request: GenerationRequest, pipeline: StableDiffusionXLPipeline) -> Image:
    generator = Generator(pipeline.device).manual_seed(request.seed) if request.seed else None

    return pipeline(
        prompt=request.prompt,
        negative_prompt=request.negative_prompt,
        width=request.width,
        height=request.height,
        generator=generator,
    ).images
