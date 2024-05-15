from diffusers import DiffusionPipeline
from diffusers.schedulers import KarrasDiffusionSchedulers
from python_coreml_stable_diffusion.pipeline import CoreMLStableDiffusionPipeline
from transformers import CLIPTokenizer, CLIPImageProcessor


class StableDiffusionXLMinimalPipeline(DiffusionPipeline):
    f"""
    A minimal SDXL pipeline that includes only what's needed for {CoreMLStableDiffusionPipeline}
    """

    _optional_components = [
        "tokenizer",
        "tokenizer_2",
        "feature_extractor",
    ]

    def __init__(
        self,
        tokenizer: CLIPTokenizer,
        tokenizer_2: CLIPTokenizer,
        scheduler: KarrasDiffusionSchedulers,
        feature_extractor: CLIPImageProcessor = None,
        force_zeros_for_empty_prompt: bool = True,
    ):
        super().__init__()

        self.register_modules(
            tokenizer=tokenizer,
            tokenizer_2=tokenizer_2,
            scheduler=scheduler,
            feature_extractor=feature_extractor,
        )

        self.register_to_config(force_zeros_for_empty_prompt=force_zeros_for_empty_prompt)


class CoreMLPipelines:
    base_minimal_pipeline: StableDiffusionXLMinimalPipeline
    coreml_sdxl_pipeline: CoreMLStableDiffusionPipeline
    coreml_models_path: str

    def __init__(
        self,
        base_minimal_pipeline: StableDiffusionXLMinimalPipeline,
        coreml_sdxl_pipeline: CoreMLStableDiffusionPipeline,
        coreml_models_path: str,
    ):
        self.base_minimal_pipeline = base_minimal_pipeline
        self.coreml_sdxl_pipeline = coreml_sdxl_pipeline
        self.coreml_models_path = coreml_models_path
