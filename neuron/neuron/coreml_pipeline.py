from os import PathLike
from os.path import isdir, join

from coremltools import ComputeUnit
from diffusers import (
    DDIMScheduler, DPMSolverMultistepScheduler, EulerAncestralDiscreteScheduler,
    EulerDiscreteScheduler, LMSDiscreteScheduler, PNDMScheduler,
)
from huggingface_hub import snapshot_download
from python_coreml_stable_diffusion.coreml_model import CoreMLModel
from python_coreml_stable_diffusion.pipeline import CoreMLStableDiffusionPipeline, get_coreml_pipe
from transformers import CLIPTokenizer, CLIPFeatureExtractor

from .pipeline import StableDiffusionXLMinimalPipeline


class CoreMLStableDiffusionXLPipeline(CoreMLStableDiffusionPipeline):
    def __init__(
        self,
        text_encoder: CoreMLModel,
        text_encoder_2: CoreMLModel,
        unet: CoreMLModel,
        vae_decoder: CoreMLModel,
        scheduler: (
            DDIMScheduler |
            DPMSolverMultistepScheduler |
            EulerAncestralDiscreteScheduler |
            EulerDiscreteScheduler |
            LMSDiscreteScheduler |
            PNDMScheduler
        ),
        tokenizer: CLIPTokenizer,
        tokenizer_2: CLIPTokenizer,
        force_zeros_for_empty_prompt: bool | None = True,
        feature_extractor: CLIPFeatureExtractor | None = None,
    ):
        super().__init__(
            text_encoder,
            unet,
            vae_decoder,
            scheduler,
            tokenizer,
            None,
            True,
            force_zeros_for_empty_prompt,
            feature_extractor,
            None,
            text_encoder_2,
            tokenizer_2,
        )

    def save_pretrained(
        self,
        save_directory: str | PathLike,
        safe_serialization: bool = True,
        variant: str | None = None,
        push_to_hub: bool = False,
        **kwargs
    ):
        coreml_dir = join(save_directory, "mlpackages")

        self.text_encoder.model.save(join(coreml_dir, "TextEncoder"))
        self.text_encoder_2.model.save(join(coreml_dir, "TextEncoder2"))
        self.unet.model.save(join(coreml_dir, "Unet"))
        self.vae_decoder.model.save(join(coreml_dir, "VAEDecoder"))

        StableDiffusionXLMinimalPipeline.save_pretrained(
            self,
            save_directory,
            safe_serialization,
            variant,
            push_to_hub,
            **kwargs,
        )

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path: str | PathLike | None, **kwargs):
        base_pipeline = StableDiffusionXLMinimalPipeline.from_pretrained(pretrained_model_name_or_path, **kwargs)

        if isdir(pretrained_model_name_or_path):
            directory = pretrained_model_name_or_path
        else:
            directory = snapshot_download(pretrained_model_name_or_path)

        coreml_dir = join(directory, "mlpackages")

        pipeline = get_coreml_pipe(
            pytorch_pipe=base_pipeline,
            mlpackages_dir=coreml_dir,
            model_version="xl",
            compute_unit=ComputeUnit.CPU_AND_GPU.name,
            delete_original_pipe=False,
            force_zeros_for_empty_prompt=base_pipeline.force_zeros_for_empty_prompt,
        )

        return cls(
            text_encoder=pipeline.text_encoder,
            text_encoder_2=pipeline.text_encoder_2,
            unet=pipeline.unet,
            vae_decoder=pipeline.vae_decoder,
            scheduler=pipeline.scheduler,
            tokenizer=pipeline.tokenizer,
            tokenizer_2=pipeline.tokenizer_2,
            force_zeros_for_empty_prompt=pipeline.force_zeros_for_empty_prompt,
            feature_extractor=pipeline.feature_extractor,
        )
