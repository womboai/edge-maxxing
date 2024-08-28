from PIL.Image import Image
from diffusers import StableDiffusionXLPipeline
from torch import Generator


def load_pipeline() -> StableDiffusionXLPipeline:
    pipeline = StableDiffusionXLPipeline.from_pretrained(
        "stablediffusionapi/newdream-sdxl-20",
        cache_dir="./models",
        local_files_only=True,
    ).to("cuda")

    pipeline(prompt="")

    return pipeline


def infer(request: GenerationRequest, pipeline: StableDiffusionXLPipeline) -> Image:
    generator = Generator(pipeline.device).manual_seed(request.seed) if request.seed else None

    return pipeline(
        prompt=request.prompt,
        negative_prompt=request.negative_prompt,
        width=request.width,
        height=request.height,
        generator=generator,
    ).images
